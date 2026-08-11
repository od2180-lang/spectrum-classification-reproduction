#!/usr/bin/env python3
"""Prepare a leakage-aware, paper-style feature dataset.

The released dataset stores PSD matrices as ``time x frequency`` arrays.  This
module deliberately keeps detection separate from the technology labels: the
filename label is used only as ground truth and never to tune or gate the
detector.

The detector follows the paper's fixed TDS settings:

* 215-bin linear-amplitude noise blocks, with the quietest block selected per
  sensor and a ``mean + 3 * std`` noise level;
* at most 360 time rows for the occupancy/boundary calculation;
* 5 dB occupancy, CV=1 on the linear row-average, and 3 dB peak edges; and
* fixed SciPy peak settings (distance=10, width=4, prominence=mean +
  0.2*std after an 8-bin moving average) so the paper's unspecified peak
  operation remains reproducible.

No technology-specific width, frequency, SNR, or entropy gate is applied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import shutil
import tarfile
import tempfile
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

try:
    from scipy.signal import find_peaks
except ImportError:  # Keep split/archive unit tests usable without SciPy.
    find_peaks = None


try:
    # tsfresh is an optional runtime dependency in this repository.  Import it
    # lazily so detector and split tests do not require the feature stack.
    from feature_extraction import extract_33_features
except ModuleNotFoundError:
    extract_33_features = None


LABELS = ("dab", "dvbt", "fm", "gsm", "lte", "tetra")
LABEL_PATTERN = re.compile(r"_(dab|dvbt|fm|gsm|lte|tetra)(?:_|\.|$)", re.IGNORECASE)

NOISE_BLOCK_BINS = 215
BOUNDARY_ROWS = 360
CHUNK_BINS = 215
OCCUPANCY_DB = 5.0
CV_THRESHOLD = 1.0
EDGE_DB = 3.0

# These choices are fixed for every class and band. They resolve details that
# the paper leaves unspecified and avoid the adaptive, class-aware parameters
# used by deployment experiments elsewhere in this repository.
PEAK_DISTANCE = 10
PEAK_WIDTH = 4
PEAK_PROMINENCE_FACTOR = 0.2
PEAK_SMOOTHING_BINS = 8

RANDOM_SEED = 42
TRAIN_FRACTION = 0.8
SENSOR_SEARCH_TRIALS = 4096
SMOKE_FILES_PER_CLASS = 2
SMOKE_MAX_ROWS_PER_FILE = 12
SMOKE_MAX_SAMPLES_PER_CLASS = 120


@dataclass(frozen=True)
class InputFile:
    """A discovered input file and its dataset-level identifiers."""

    path: Path
    source_file: str
    technology: str | None
    sensor: str
    payload_hash: str | None = None


def technology_from_filename(filename: str | Path) -> str | None:
    """Return the technology token embedded in a filename, if present."""

    match = LABEL_PATTERN.search(Path(filename).name)
    return match.group(1).lower() if match else None


def sensor_from_path(path: str | Path) -> str:
    """Return the sensor directory for ``sensor/date/file.npy`` paths."""

    path = Path(path)
    return path.parent.parent.name


def discover_files(data_dir: str | Path) -> list[InputFile]:
    """Discover sorted NPY files below an extracted dataset root."""

    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    files: list[InputFile] = []
    for path in sorted(root.rglob("*.npy")):
        files.append(
            InputFile(
                path=path,
                source_file=path.relative_to(root).as_posix(),
                technology=technology_from_filename(path.name),
                sensor=sensor_from_path(path),
            )
        )
    return files


def _input_shape(info: InputFile) -> tuple[int, ...] | None:
    """Read an NPY header when selecting smoke inputs, without loading data."""

    try:
        data = np.load(info.path, mmap_mode="r", allow_pickle=False)
    except (OSError, TypeError, ValueError):
        return None
    return tuple(int(value) for value in data.shape)


def select_smoke_files(
    files: Sequence[InputFile],
    *,
    max_files_per_class: int = SMOKE_FILES_PER_CLASS,
    max_files: int | None = None,
) -> list[InputFile]:
    """Select a small deterministic, sensor-stratified input subset.

    Two files per class are selected from a pair of sensors when possible.  A
    pair is scored by total frequency width so smoke runs avoid accidentally
    selecting an enormous full-spectrum file.  The fallback selects two
    sensors per class independently, which also keeps the sensor split valid.
    """

    if max_files_per_class < 2:
        raise ValueError("Smoke/source selection needs at least two files per class")
    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be positive")

    ordered = sorted(files, key=lambda info: info.source_file)
    by_sensor: dict[str, dict[str, list[InputFile]]] = defaultdict(lambda: defaultdict(list))
    for info in ordered:
        if info.technology not in LABELS:
            continue
        shape = _input_shape(info)
        if shape is not None and (
            len(shape) != 2 or 0 in shape or shape[1] < NOISE_BLOCK_BINS
        ):
            continue
        by_sensor[info.sensor][str(info.technology)].append(info)

    if not by_sensor:
        raise ValueError("No usable labelled files available for smoke/source selection")

    complete_sensors = [
        sensor for sensor, by_label in by_sensor.items() if all(label in by_label for label in LABELS)
    ]
    selected: list[InputFile] = []
    if len(complete_sensors) >= 2:
        pair_candidates: list[tuple[int, str, str]] = []
        for left, right in itertools.combinations(sorted(complete_sensors), 2):
            width = 0
            for label in LABELS:
                for sensor in (left, right):
                    shape = _input_shape(by_sensor[sensor][label][0])
                    width += shape[1] if shape is not None and len(shape) == 2 else 0
            pair_candidates.append((width, left, right))
        _, left, right = min(pair_candidates)
        for label in LABELS:
            selected.extend((by_sensor[left][label][0], by_sensor[right][label][0]))
    else:
        for label in LABELS:
            candidates = [
                info
                for sensor in sorted(by_sensor)
                for info in by_sensor[sensor].get(label, [])
            ]
            sensors = sorted({info.sensor for info in candidates})
            if len(sensors) < 2:
                raise ValueError(f"Need at least two sensors containing {label}")
            selected.extend(
                next(info for info in candidates if info.sensor == sensor) for sensor in sensors[:2]
            )

    required_selection = {info.source_file: info for info in selected}
    if max_files_per_class > 2:
        selected_paths = {info.source_file for info in selected}
        usable_paths = {
            info.source_file
            for by_label in by_sensor.values()
            for label_files in by_label.values()
            for info in label_files
        }
        for label in LABELS:
            candidates = [
                info
                for info in ordered
                if info.technology == label and info.source_file in usable_paths
            ]
            for info in candidates:
                if len([item for item in selected if item.technology == label]) >= max_files_per_class:
                    break
                if info.source_file not in selected_paths:
                    selected.append(info)
                    selected_paths.add(info.source_file)

    selected = sorted(
        required_selection.values(), key=lambda info: info.source_file
    ) + sorted(
        {
            info.source_file: info
            for info in selected
            if info.source_file not in required_selection
        }.values(),
        key=lambda info: info.source_file,
    )
    if max_files is not None and len(selected) > max_files:
        minimum = len(LABELS) * 2
        if max_files < minimum:
            raise ValueError(f"max_files must be at least {minimum} to preserve both sensor partitions")
        selected = selected[:max_files]

    selected_labels = {info.technology for info in selected}
    selected_sensors = {info.sensor for info in selected}
    if selected_labels != set(LABELS) or len(selected_sensors) < 2:
        raise ValueError("Selected files must contain all six classes and at least two sensors")
    if any(
        len({info.sensor for info in selected if info.technology == label}) < 2
        for label in LABELS
    ):
        raise ValueError("Selected files must contain every class on at least two sensors")
    return selected


def count_archive_files(archive: str | Path) -> Counter[str]:
    """Count labelled NPY members without extracting or loading features."""

    counts: Counter[str] = Counter()
    with tarfile.open(archive, mode="r:*") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.lower().endswith(".npy"):
                label = technology_from_filename(member.name)
                if label is not None:
                    counts[label] += 1
    return counts


def _safe_archive_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return target


@contextmanager
def extracted_archive(archive: str | Path) -> Iterator[Path]:
    """Safely extract NPY archive members into a temporary directory."""

    with tempfile.TemporaryDirectory(prefix="spectrum_bands_") as temporary:
        root = Path(temporary)
        with tarfile.open(archive, mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".npy"):
                    continue
                target = _safe_archive_target(root, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        yield root


def _linear_amplitude(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.power(10.0, np.asarray(values, dtype=np.float64) / 20.0)


def _fixed_noise_blocks(data: np.ndarray, block_bins: int = NOISE_BLOCK_BINS) -> Iterator[tuple[float, np.ndarray]]:
    """Yield mean amplitude and block values for complete fixed-size blocks."""

    if data.ndim != 2 or data.shape[1] < block_bins:
        return

    # Noise selection is site-level and uses every available row.  The 360-row
    # limit belongs only to the later occupancy/boundary calculation.
    rows = data.shape[0]
    block_count = data.shape[1] // block_bins
    for row_index in range(rows):
        row = _linear_amplitude(data[row_index, : block_count * block_bins])
        if not np.all(np.isfinite(row)):
            continue
        blocks = row.reshape(block_count, block_bins)
        means = blocks.mean(axis=1)
        for block_index, mean_value in enumerate(means):
            if np.isfinite(mean_value):
                yield float(mean_value), blocks[block_index].copy()


def _quietest_noise_candidate(
    data: np.ndarray, block_bins: int = NOISE_BLOCK_BINS
) -> tuple[float, float] | None:
    """Return ``(quietest_mean_amplitude, derived_noise_db)`` for one file."""

    best_mean = None
    best_block = None
    for mean_value, block in _fixed_noise_blocks(data, block_bins):
        if best_mean is None or mean_value < best_mean:
            best_mean = mean_value
            best_block = block
    if best_block is None or best_mean is None:
        return None

    noise_amplitude = float(best_block.mean() + 3.0 * best_block.std())
    if not np.isfinite(noise_amplitude) or noise_amplitude <= 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        noise_db = 20.0 * np.log10(noise_amplitude)
    return (float(best_mean), float(noise_db)) if np.isfinite(noise_db) else None


def quietest_noise_level(data: np.ndarray, block_bins: int = NOISE_BLOCK_BINS) -> float | None:
    """Estimate a file candidate's noise level in dB from its quietest block."""

    candidate = _quietest_noise_candidate(data, block_bins)
    return candidate[1] if candidate is not None else None


def build_site_noise(files: Sequence[InputFile]) -> dict[str, float]:
    """Select the quietest available 215-bin noise block for every sensor."""

    candidates: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for info in files:
        try:
            data = np.load(info.path, mmap_mode="r", allow_pickle=False)
            if data.ndim != 2 or not np.all(np.isfinite(data)):
                continue
            candidate = _quietest_noise_candidate(data)
        except (OSError, TypeError, ValueError):
            continue
        if candidate is not None:
            candidates[info.sensor].append(candidate)

    return {
        sensor: min(levels, key=lambda item: item[0])[1]
        for sensor, levels in candidates.items()
        if levels
    }


def _moving_average(values: np.ndarray, width: int = PEAK_SMOOTHING_BINS) -> np.ndarray:
    if len(values) <= 1 or width <= 1:
        return values.astype(np.float64, copy=True)
    return np.convolve(values, np.ones(width, dtype=np.float64) / width, mode="same")


def _drop_db(peak_value: float, value: float) -> float:
    if value <= 0 or not np.isfinite(value):
        return np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(20.0 * np.log10(peak_value / value))


def _peak_edges(values: np.ndarray, offset: int) -> list[tuple[int, int]]:
    """Split one occupied interval using the fixed paper peak settings."""

    if find_peaks is None:
        raise RuntimeError("scipy is required for peak-based transmission detection")

    smoothed = _moving_average(values)
    prominence = float(np.mean(smoothed) + PEAK_PROMINENCE_FACTOR * np.std(smoothed))
    peaks, _ = find_peaks(
        smoothed,
        distance=PEAK_DISTANCE,
        width=PEAK_WIDTH,
        prominence=prominence,
    )
    if len(peaks) == 0:
        return [(offset, offset + len(values) - 1)]

    intervals: list[tuple[int, int]] = []
    for peak in peaks:
        peak_value = float(smoothed[peak])
        left = int(peak)
        while left > 0:
            candidate = left - 1
            if _drop_db(peak_value, float(smoothed[candidate])) >= EDGE_DB:
                left = candidate
                break
            left = candidate

        right = int(peak)
        last = len(smoothed) - 1
        while right < last:
            candidate = right + 1
            if _drop_db(peak_value, float(smoothed[candidate])) >= EDGE_DB:
                right = candidate
                break
            right = candidate
        intervals.append((offset + left, offset + right))

    # Peaks can have touching 3-dB regions.  Return ordered non-overlapping
    # boundaries rather than emitting duplicate samples from the overlap.
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _occupied_intervals(values: np.ndarray, threshold: float) -> Iterator[tuple[int, int]]:
    active = values >= threshold
    start = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            yield start, index - 1
            start = None
    if start is not None:
        yield start, len(values) - 1


def detect_transmissions(
    data: np.ndarray,
    noise_db: float,
    boundary_rows: int = BOUNDARY_ROWS,
) -> np.ndarray:
    """Detect inclusive ``[start, end]`` frequency boundaries.

    Occupancy is computed from the linear-amplitude average of up to 360 time
    rows.  CV and peak processing use that same averaged linear vector.
    """

    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        return np.empty((0, 2), dtype=np.int64)
    if not np.all(np.isfinite(data)) or not np.isfinite(noise_db):
        return np.empty((0, 2), dtype=np.int64)

    # K is the number of rows actually available to this file.  Smoke runs
    # may intentionally provide fewer than the paper's 360-row boundary.
    rows = min(int(boundary_rows), data.shape[0])
    if rows <= 0:
        return np.empty((0, 2), dtype=np.int64)
    averaged = _linear_amplitude(data[:rows, :]).mean(axis=0)
    if not np.all(np.isfinite(averaged)):
        return np.empty((0, 2), dtype=np.int64)

    noise_amplitude = float(_linear_amplitude(np.asarray([noise_db]))[0])
    if not np.isfinite(noise_amplitude) or noise_amplitude <= 0:
        return np.empty((0, 2), dtype=np.int64)
    occupancy_threshold = noise_amplitude * float(10.0 ** (OCCUPANCY_DB / 20.0))

    detections: list[tuple[int, int]] = []
    for start, end in _occupied_intervals(averaged, occupancy_threshold):
        interval = averaged[start : end + 1]
        mean_value = float(np.mean(interval))
        if mean_value <= 0 or not np.isfinite(mean_value):
            continue
        coefficient = float(np.std(interval) / mean_value)
        if coefficient < CV_THRESHOLD:
            detections.append((start, end))
        else:
            detections.extend(_peak_edges(interval, start))

    return np.asarray(detections, dtype=np.int64).reshape(-1, 2)


def iter_hopping_chunks(
    transmission: np.ndarray,
    chunk_bins: int = CHUNK_BINS,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield sequential ``[start, end)`` chunks from a detected transmission.

    Narrow transmissions are preserved unchanged.  For a wide transmission,
    only complete 215-bin chunks are yielded; the final partial tail is
    intentionally discarded as required by the paper-style preparation.
    """

    if transmission.ndim != 2:
        raise ValueError(f"Expected a time x frequency matrix, got {transmission.shape}")
    width = transmission.shape[1]
    if width == 0:
        return
    if width <= chunk_bins:
        yield 0, width, transmission
        return
    for start in range(0, width - chunk_bins + 1, chunk_bins):
        end = start + chunk_bins
        yield start, end, transmission[:, start:end]


def array_payload_hash(path: str | Path) -> str:
    """Hash NPY array payload, shape, and dtype without loading it all at once."""

    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {data.shape}")
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode("ascii"))
    digest.update(repr(tuple(data.shape)).encode("ascii"))
    for start in range(0, data.shape[0], 32):
        block = np.ascontiguousarray(data[start : start + 32])
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def sensor_groups(files: Sequence[InputFile]) -> dict[str, str]:
    """Union sensors with identical payloads and return stable group IDs."""

    sensors = sorted({info.sensor for info in files})
    union_find = _UnionFind(sensors)
    by_hash: dict[str, str] = {}
    for info in files:
        if info.payload_hash is None:
            continue
        previous = by_hash.get(info.payload_hash)
        if previous is not None:
            union_find.union(previous, info.sensor)
        else:
            by_hash[info.payload_hash] = info.sensor

    members: dict[str, list[str]] = defaultdict(list)
    for sensor in sensors:
        members[union_find.find(sensor)].append(sensor)
    group_ids = {root: "+".join(sorted(values)) for root, values in members.items()}
    return {sensor: group_ids[union_find.find(sensor)] for sensor in sensors}


def _train_size(count: int, train_fraction: float) -> int:
    if count < 2:
        raise ValueError(f"Each class needs at least two samples, got {count}")
    return min(count - 1, max(1, int(round(count * train_fraction))))


def random_class_split(
    records: Sequence[Mapping[str, object]],
    seed: int = RANDOM_SEED,
    train_fraction: float = TRAIN_FRACTION,
) -> np.ndarray:
    """Return deterministic class-stratified random train/test assignments."""

    assignments = np.full(len(records), "", dtype="U5")
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_class[str(record["technology"])].append(index)
    rng = np.random.default_rng(seed)

    # Only payload hashes shared by multiple source files are grouped. Rows
    # from an ordinary source file remain independent samples in this split.
    # This preserves the paper-style segment split while keeping byte-identical
    # duplicate files out of both partitions.
    by_payload: dict[str, list[int]] = defaultdict(list)
    payload_sources: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        payload_hash = record.get("payload_hash")
        if payload_hash:
            key = str(payload_hash)
            by_payload[key].append(index)
            payload_sources[key].add(str(record.get("source_file", "")))
    duplicate_payloads = {
        payload_hash
        for payload_hash, sources in payload_sources.items()
        if len(sources) > 1
    }

    for label in LABELS:
        indices = by_class.get(label, [])
        if len(indices) < 2:
            raise ValueError(f"Random split cannot contain {label} in both partitions: {len(indices)} samples")
        units: list[list[int]] = []
        used: set[int] = set()
        for index in indices:
            payload_hash = records[index].get("payload_hash")
            unit = (
                by_payload[str(payload_hash)]
                if payload_hash and str(payload_hash) in duplicate_payloads
                else [index]
            )
            if any(item in used for item in unit):
                continue
            used.update(unit)
            units.append([item for item in unit if item in indices])
        order = rng.permutation(len(units))
        units = [units[index] for index in order]
        target = _train_size(len(indices), train_fraction)
        selected = 0
        for unit in units:
            if selected < target and (
                selected + len(unit) <= target or selected == 0 and len(units) == 1
            ):
                assignments[unit] = "train"
                selected += len(unit)
            else:
                assignments[unit] = "test"
        # If a duplicate unit crossed the class boundary, preserve its single
        # assignment while still assigning any unassigned class rows safely.
        unassigned = [index for index in indices if assignments[index] == ""]
        assignments[unassigned] = "test"

    _validate_assignment_classes(records, assignments, "random")
    return assignments


def _sensor_objective(
    test_counts: np.ndarray,
    totals: np.ndarray,
    test_fraction: float,
) -> float:
    if np.any(test_counts <= 0) or np.any(test_counts >= totals):
        return 1_000_000.0 + float(np.sum(test_counts <= 0) + np.sum(test_counts >= totals))
    ratios = test_counts / totals
    return float(np.mean(np.abs(ratios - test_fraction)) + 0.1 * abs(float(ratios.mean() - test_fraction)))


def sensor_disjoint_split(
    records: Sequence[Mapping[str, object]],
    seed: int = RANDOM_SEED,
    train_fraction: float = TRAIN_FRACTION,
    trials: int = SENSOR_SEARCH_TRIALS,
) -> np.ndarray:
    """Find a deterministic approximately-stratified, sensor-disjoint split."""

    sensors = sorted({str(record["sensor"]) for record in records})
    union_find = _UnionFind(sensors)
    explicit_groups: dict[str, str] = {}
    payload_sensors: dict[str, str] = {}
    for record in records:
        sensor = str(record["sensor"])
        explicit = record.get("sensor_group")
        if explicit:
            explicit_groups.setdefault(str(explicit), sensor)
            union_find.union(explicit_groups[str(explicit)], sensor)
        payload_hash = record.get("payload_hash")
        if payload_hash:
            payload_hash = str(payload_hash)
            if payload_hash in payload_sensors:
                union_find.union(payload_sensors[payload_hash], sensor)
            else:
                payload_sensors[payload_hash] = sensor
    members: dict[str, list[str]] = defaultdict(list)
    for sensor in sensors:
        members[union_find.find(sensor)].append(sensor)
    derived_groups = {
        sensor: "+".join(sorted(values))
        for root, values in members.items()
        for sensor in values
    }

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        group = derived_groups[str(record["sensor"])]
        group_to_indices[group].append(index)
    groups = sorted(group_to_indices)
    group_counts = np.zeros((len(groups), len(LABELS)), dtype=np.int64)
    label_index = {label: index for index, label in enumerate(LABELS)}
    for group_index, group in enumerate(groups):
        for sample_index in group_to_indices[group]:
            label = str(records[sample_index]["technology"])
            if label in label_index:
                group_counts[group_index, label_index[label]] += 1
    totals = group_counts.sum(axis=0)
    if np.any(totals < 2):
        raise ValueError("Sensor split requires at least two sensor groups per class")
    if any(np.count_nonzero(group_counts[:, index]) < 2 for index in range(len(LABELS))):
        raise ValueError("At least one class occurs in only one sensor group")

    test_fraction = 1.0 - train_fraction
    rng = np.random.default_rng(seed + 1009)
    best_score = float("inf")
    best_mask: np.ndarray | None = None

    def consider(mask: np.ndarray) -> None:
        nonlocal best_score, best_mask
        score = _sensor_objective(group_counts[mask].sum(axis=0), totals, test_fraction)
        if score < best_score:
            best_score = score
            best_mask = mask.copy()

    # The first candidates are stratified five-fold-like assignments.  The
    # remaining candidates are a deterministic robust search over group sets.
    for _ in range(max(1, min(256, trials))):
        order = rng.permutation(len(groups))
        fold_counts = np.zeros((5, len(LABELS)), dtype=np.int64)
        fold_groups: list[list[int]] = [[] for _ in range(5)]
        for group_index in order:
            scores = []
            for fold in range(5):
                candidate = fold_counts[fold] + group_counts[group_index]
                scores.append(float(np.mean(np.abs(candidate / np.maximum(totals, 1) - 0.2))))
            fold = int(np.argmin(scores))
            fold_groups[fold].append(int(group_index))
            fold_counts[fold] += group_counts[group_index]
        for fold in range(5):
            mask = np.zeros(len(groups), dtype=bool)
            mask[fold_groups[fold]] = True
            consider(mask)

    for _ in range(max(1, trials)):
        mask = rng.random(len(groups)) < test_fraction
        for label_index_value in range(len(LABELS)):
            if group_counts[mask, label_index_value].sum() == 0:
                candidates = np.flatnonzero(~mask & (group_counts[:, label_index_value] > 0))
                if len(candidates):
                    candidate_scores = []
                    for candidate in candidates:
                        candidate_mask = mask.copy()
                        candidate_mask[candidate] = True
                        candidate_scores.append(
                            _sensor_objective(
                                group_counts[candidate_mask].sum(axis=0), totals, test_fraction
                            )
                        )
                    mask[candidates[int(np.argmin(candidate_scores))]] = True
        for label_index_value in range(len(LABELS)):
            if group_counts[~mask, label_index_value].sum() == 0:
                candidates = np.flatnonzero(mask & (group_counts[:, label_index_value] > 0))
                candidates = [
                    candidate
                    for candidate in candidates
                    if group_counts[mask & (np.arange(len(groups)) != candidate), label_index_value].sum() > 0
                ]
                if candidates:
                    candidate_scores = []
                    for candidate in candidates:
                        candidate_mask = mask.copy()
                        candidate_mask[candidate] = False
                        candidate_scores.append(
                            _sensor_objective(
                                group_counts[candidate_mask].sum(axis=0), totals, test_fraction
                            )
                        )
                    mask[candidates[int(np.argmin(candidate_scores))]] = False
        consider(mask)

    if best_mask is None or best_score >= 1_000_000:
        raise ValueError("Could not construct a sensor-disjoint split with every class in both partitions")
    assignments = np.full(len(records), "", dtype="U5")
    for group_index, group in enumerate(groups):
        assignments[group_to_indices[group]] = "test" if best_mask[group_index] else "train"
    _validate_assignment_classes(records, assignments, "sensor")
    return assignments


def _validate_assignment_classes(
    records: Sequence[Mapping[str, object]], assignments: np.ndarray, name: str
) -> None:
    for label in LABELS:
        indices = [index for index, record in enumerate(records) if record["technology"] == label]
        values = set(assignments[indices])
        if values != {"train", "test"}:
            raise ValueError(f"{name} split does not contain {label} in both partitions: {values}")


def assign_splits(
    records: list[dict[str, object]],
    seed: int = RANDOM_SEED,
    train_fraction: float = TRAIN_FRACTION,
) -> None:
    """Add ``random_split`` and ``sensor_split`` fields to sample records."""

    random_assignments = random_class_split(records, seed, train_fraction)
    sensor_assignments = sensor_disjoint_split(records, seed, train_fraction)
    for index, record in enumerate(records):
        record["random_split"] = str(random_assignments[index])
        record["sensor_split"] = str(sensor_assignments[index])


def _load_extractor():
    global extract_33_features
    if extract_33_features is None:
        from feature_extraction import extract_33_features as extractor

        extract_33_features = extractor
    return extract_33_features


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return dict(value)
    return value


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "pandas", "tsfresh"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _counts_by(records: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    return dict(Counter(str(record[field]) for record in records))


def prepare_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = RANDOM_SEED,
    source_kind: str = "data-dir",
    source_name: str | None = None,
    smoke: bool = False,
    max_files_per_class: int | None = None,
    max_files: int | None = None,
    max_rows_per_file: int | None = None,
    max_samples_per_class: int | None = None,
) -> dict[str, object]:
    """Prepare all valid feature rows and write the six requested artifacts."""

    root = Path(data_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    discovered = discover_files(root)
    if not discovered:
        raise ValueError(f"No NPY files found below {root}")

    files_discovered = len(discovered)
    if smoke:
        max_files_per_class = (
            SMOKE_FILES_PER_CLASS if max_files_per_class is None else max_files_per_class
        )
        max_rows_per_file = SMOKE_MAX_ROWS_PER_FILE if max_rows_per_file is None else max_rows_per_file
        max_samples_per_class = (
            SMOKE_MAX_SAMPLES_PER_CLASS
            if max_samples_per_class is None
            else max_samples_per_class
        )
    if max_rows_per_file is not None and max_rows_per_file < 1:
        raise ValueError("max_rows_per_file must be positive")
    if max_samples_per_class is not None and max_samples_per_class < 2:
        raise ValueError("max_samples_per_class must be at least two for both splits")
    if smoke or max_files_per_class is not None or max_files is not None:
        discovered = select_smoke_files(
            discovered,
            max_files_per_class=(
                SMOKE_FILES_PER_CLASS if max_files_per_class is None else max_files_per_class
            ),
            max_files=max_files,
        )

    hashed: list[InputFile] = []
    hash_errors: dict[str, str] = {}
    for info in discovered:
        try:
            hashed.append(replace(info, payload_hash=array_payload_hash(info.path)))
        except (OSError, TypeError, ValueError) as error:
            hashed.append(info)
            hash_errors[info.source_file] = str(error)
    discovered = hashed
    groups = sensor_groups(discovered)
    noise_by_sensor = build_site_noise(discovered)
    extractor = _load_extractor()

    required_sensor_groups: dict[str, set[str]] = {
        label: {
            groups.get(info.sensor, info.sensor)
            for info in discovered
            if info.technology == label
        }
        for label in LABELS
    }
    if max_samples_per_class is not None:
        missing = [label for label, values in required_sensor_groups.items() if len(values) < 2]
        if missing:
            raise ValueError(
                "Sample cap cannot preserve sensor splits for: " + ", ".join(missing)
            )

    feature_parts: list[np.ndarray] = []
    segment_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []
    feature_columns: list[str] | None = None
    detection_counts: Counter[str] = Counter()
    chunk_counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    accepted_sensor_groups: dict[str, set[str]] = defaultdict(set)

    def reject(info: InputFile, stage: str, reason: str, detail: str) -> None:
        rejection_rows.append(
            {
                "source_file": info.source_file,
                "technology": info.technology or "",
                "sensor": info.sensor,
                "stage": stage,
                "reason": reason,
                "detail": detail,
            }
        )

    for file_index, info in enumerate(discovered):
        file_row: dict[str, object] = {
            "file_index": file_index,
            "source_file": info.source_file,
            "technology": info.technology or "",
            "sensor": info.sensor,
            "sensor_group": groups.get(info.sensor, info.sensor),
            "payload_hash": info.payload_hash or "",
            "rows": 0,
            "processed_rows": 0,
            "frequency_bins": 0,
            "noise_db": "",
            "detected_transmissions": 0,
            "hopping_chunks": 0,
            "discarded_partial_chunks": 0,
            "samples": 0,
            "status": "",
        }
        file_rows.append(file_row)
        if info.technology is None:
            file_row["status"] = "rejected"
            reject(info, "label", "unrecognized_technology", "No technology token in filename")
            continue

        try:
            data = np.load(info.path, allow_pickle=False)
            file_row["rows"], file_row["frequency_bins"] = data.shape if data.ndim == 2 else (0, 0)
            if data.ndim != 2 or 0 in data.shape:
                file_row["status"] = "rejected"
                reject(info, "load", "invalid_shape", f"Expected non-empty 2-D array, got {data.shape}")
                continue
            if not np.all(np.isfinite(data)):
                file_row["status"] = "rejected"
                reject(info, "load", "nonfinite_data", "Input contains non-finite values")
                continue
            processing_data = data
            if max_rows_per_file is not None:
                processing_data = data[:max_rows_per_file]
            file_row["processed_rows"] = int(processing_data.shape[0])
            if processing_data.shape[0] == 0:
                file_row["status"] = "rejected"
                reject(info, "load", "no_processing_rows", "Row limit left no data")
                continue
            noise_db = noise_by_sensor.get(info.sensor)
            if noise_db is None:
                file_row["status"] = "rejected"
                reject(info, "detect", "noise_unavailable", "Sensor has no complete finite 215-bin noise block")
                continue
            file_row["noise_db"] = noise_db
            detections = detect_transmissions(processing_data, noise_db)
            file_row["detected_transmissions"] = len(detections)
            detection_counts[info.technology] += len(detections)
            sample_start = len(segment_rows)
            for detection_index, (detected_start, detected_end) in enumerate(detections):
                detected_start, detected_end = int(detected_start), int(detected_end)
                transmission = processing_data[:, detected_start : detected_end + 1]
                width = transmission.shape[1]
                if width > CHUNK_BINS and width % CHUNK_BINS:
                    file_row["discarded_partial_chunks"] = int(file_row["discarded_partial_chunks"]) + 1
                for chunk_index, (local_start, local_end, chunk) in enumerate(
                    iter_hopping_chunks(transmission, CHUNK_BINS)
                ):
                    if max_samples_per_class is not None:
                        group = groups.get(info.sensor, info.sensor)
                        accepted = accepted_counts[info.technology]
                        seen_groups = accepted_sensor_groups[info.technology]
                        required = min(2, len(required_sensor_groups[info.technology]))
                        missing_groups = required - len(seen_groups)
                        if accepted >= max_samples_per_class or (
                            group in seen_groups
                            and accepted >= max_samples_per_class - missing_groups
                        ):
                            continue
                    try:
                        # tsfresh emits RuntimeWarning for derivatives and
                        # higher moments on valid narrowband vectors. Those
                        # rows are handled by the finite-feature check below.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            features, columns = extractor(np.asarray(chunk, dtype=np.float32))
                        # The pandas-backed extractor can return an object
                        # array even though every feature value is numeric.
                        features = np.asarray(features, dtype=np.float64)
                        columns = [str(column) for column in columns]
                        if features.ndim != 2 or features.shape != (chunk.shape[0], 33) or len(columns) != 33:
                            raise ValueError(
                                f"Expected ({chunk.shape[0]}, 33) features, got {features.shape} and {len(columns)} columns"
                            )
                        if feature_columns is None:
                            feature_columns = columns
                        elif columns != feature_columns:
                            raise ValueError("Feature column order changed between chunks")
                    except Exception as error:  # A bad feature chunk is auditable, not fabricated.
                        reject(
                            info,
                            "features",
                            "feature_error",
                            f"detection={detection_index}, chunk={chunk_index}: {error}",
                        )
                        continue

                    finite_rows = np.all(np.isfinite(features), axis=1)
                    for row_index in np.flatnonzero(~finite_rows):
                        reject(
                            info,
                            "features",
                            "nonfinite_features",
                            f"detection={detection_index}, chunk={chunk_index}, time_row={int(row_index)}",
                        )
                    if not np.any(finite_rows):
                        continue
                    valid_indices = np.flatnonzero(finite_rows)
                    if max_samples_per_class is not None:
                        group = groups.get(info.sensor, info.sensor)
                        accepted = accepted_counts[info.technology]
                        seen_groups = accepted_sensor_groups[info.technology]
                        required = min(2, len(required_sensor_groups[info.technology]))
                        accepted_indices: list[int] = []
                        for row_index in valid_indices:
                            missing_groups = required - len(seen_groups)
                            if accepted >= max_samples_per_class:
                                break
                            if group in seen_groups and accepted >= max_samples_per_class - missing_groups:
                                continue
                            accepted_indices.append(int(row_index))
                            accepted += 1
                            seen_groups.add(group)
                        valid_indices = np.asarray(accepted_indices, dtype=np.int64)
                    if not len(valid_indices):
                        continue
                    feature_parts.append(features[valid_indices].astype(np.float32, copy=False))
                    accepted_counts[info.technology] += len(valid_indices)
                    accepted_sensor_groups[info.technology].add(groups.get(info.sensor, info.sensor))
                    chunk_counts[info.technology] += 1
                    file_row["hopping_chunks"] = int(file_row["hopping_chunks"]) + 1
                    for row_index in valid_indices:
                        segment_rows.append(
                            {
                                "sample_index": len(segment_rows),
                                "file_index": file_index,
                                "source_file": info.source_file,
                                "technology": info.technology,
                                "sensor": info.sensor,
                                "sensor_group": groups.get(info.sensor, info.sensor),
                                "payload_hash": info.payload_hash or "",
                                "time_row": int(row_index),
                                "detected_start": detected_start,
                                "detected_end": detected_end,
                                "chunk_index": chunk_index,
                                "chunk_start": detected_start + local_start,
                                "chunk_end": detected_start + local_end,
                                "chunk_width": local_end - local_start,
                            }
                        )
                    file_row["samples"] = int(file_row["samples"]) + int(len(valid_indices))
            file_row["status"] = "ok" if int(file_row["samples"]) else "no_samples"
            if sample_start == len(segment_rows) and not len(detections):
                file_row["status"] = "no_detections"
        except (OSError, TypeError, ValueError) as error:
            file_row["status"] = "rejected"
            reject(info, "load", "load_error", str(error))

    if feature_columns is None:
        feature_columns = []
    all_features = (
        np.concatenate(feature_parts, axis=0)
        if feature_parts
        else np.empty((0, len(feature_columns) or 33), dtype=np.float32)
    )
    if len(segment_rows) != all_features.shape[0]:
        raise RuntimeError(f"Feature/segment row mismatch: {all_features.shape} vs {len(segment_rows)}")
    if all_features.size and not np.all(np.isfinite(all_features)):
        raise RuntimeError("Non-finite feature escaped preparation")
    if len(segment_rows) == 0:
        raise ValueError("Detector and feature extraction produced no valid samples")

    for record in segment_rows:
        record["payload_hash"] = record.get("payload_hash", "")
    assign_splits(segment_rows, seed)

    segments_by_file: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in segment_rows:
        segments_by_file[int(record["file_index"])].append(record)
    for file_index, file_row in enumerate(file_rows):
        file_segments = segments_by_file.get(file_index, [])
        random_values = {str(record["random_split"]) for record in file_segments}
        sensor_values = {str(record["sensor_split"]) for record in file_segments}
        file_row["random_split"] = next(iter(random_values)) if len(random_values) == 1 else (
            "mixed" if random_values else ""
        )
        file_row["sensor_split"] = next(iter(sensor_values)) if len(sensor_values) == 1 else (
            "mixed" if sensor_values else ""
        )

    np.save(output / "features.npy", all_features)
    (output / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2) + "\n", encoding="utf-8")
    segment_fields = [
        "sample_index", "file_index", "source_file", "technology", "sensor", "sensor_group",
        "payload_hash", "time_row", "detected_start", "detected_end", "chunk_index",
        "chunk_start", "chunk_end", "chunk_width", "random_split", "sensor_split",
    ]
    _write_csv(output / "segments.csv", segment_rows, segment_fields)
    file_fields = [
        "file_index", "source_file", "technology", "sensor", "sensor_group", "payload_hash",
        "rows", "processed_rows", "frequency_bins", "noise_db", "detected_transmissions", "hopping_chunks",
        "discarded_partial_chunks", "samples", "status",
        "random_split", "sensor_split",
    ]
    _write_csv(output / "files.csv", file_rows, file_fields)
    _write_csv(
        output / "rejections.csv",
        rejection_rows,
        ["source_file", "technology", "sensor", "stage", "reason", "detail"],
    )

    metadata: dict[str, object] = {
        "schema_version": 1,
        "source_kind": source_kind,
        "source": source_name or str(root),
        "orientation": "time x frequency",
        "labels": list(LABELS),
        "seed": seed,
        "sample_count": len(segment_rows),
        "feature_shape": list(all_features.shape),
        "files_discovered": files_discovered,
        "files_selected": len(discovered),
        "smoke": smoke,
        "selected_source_files": [info.source_file for info in discovered],
        "selected_sensors": sorted({info.sensor for info in discovered}),
        "files_by_class": _counts_by([row for row in file_rows if row["technology"]], "technology"),
        "detected_transmissions_by_class": dict(detection_counts),
        "hopping_chunks_by_class": dict(chunk_counts),
        "candidate_samples_by_class": _counts_by(segment_rows, "technology"),
        "random_split_by_class": {
            f"{label}:{split}": count
            for (label, split), count in Counter(
                (row["technology"], row["random_split"]) for row in segment_rows
            ).items()
        },
        "sensor_split_by_class": {
            f"{label}:{split}": count
            for (label, split), count in Counter(
                (row["technology"], row["sensor_split"]) for row in segment_rows
            ).items()
        },
        "rejections_by_reason": dict(Counter(row["reason"] for row in rejection_rows)),
        "parameters": {
            "noise_block_bins": NOISE_BLOCK_BINS,
            "boundary_rows": BOUNDARY_ROWS,
            "occupancy_db": OCCUPANCY_DB,
            "cv_threshold": CV_THRESHOLD,
            "edge_db": EDGE_DB,
            "hopping_chunk_bins": CHUNK_BINS,
            "peak_distance": PEAK_DISTANCE,
            "peak_width": PEAK_WIDTH,
            "peak_prominence": "mean + 0.2 * std",
            "peak_smoothing_bins": PEAK_SMOOTHING_BINS,
            "class_gates": False,
            "max_rows_per_file": max_rows_per_file,
            "max_samples_per_class": max_samples_per_class,
        },
        "selection": {
            "smoke": smoke,
            "max_files_per_class": max_files_per_class,
            "max_files": max_files,
            "max_rows_per_file": max_rows_per_file,
            "max_samples_per_class": max_samples_per_class,
            "source_files": [info.source_file for info in discovered],
            "sensors": sorted({info.sensor for info in discovered}),
        },
        "split_rules": {
            "random": "class-stratified deterministic sample split; repeated exact payload units stay together",
            "sensor": "deterministic robust grouped search, approximately 80/20, linked duplicate sensors stay together",
            "train_fraction": TRAIN_FRACTION,
            "sensor_search_trials": SENSOR_SEARCH_TRIALS,
        },
        "linked_sensor_groups": len(set(groups.values())),
        "duplicate_payload_hashes": len(
            [value for value in Counter(info.payload_hash for info in discovered if info.payload_hash).values() if value > 1]
        ),
        "library_versions": _library_versions(),
    }
    metadata["hash_errors"] = hash_errors
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_value) + "\n", encoding="utf-8")
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="spectrum_bands.tar.gz input")
    source.add_argument("--data-dir", type=Path, help="extracted dataset root")
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_dataset"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a deterministic all-class, multi-sensor subset for a fast pipeline smoke run",
    )
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        help="Limit selected source files per class; at least two are required for sensor splits",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit selected source files while retaining all classes and both sensor partitions",
    )
    parser.add_argument(
        "--max-rows-per-file",
        "--max-time-rows",
        dest="max_rows_per_file",
        type=int,
        help="Limit time rows processed from each source file",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        help="Limit accepted feature samples per class",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.archive is not None:
        with extracted_archive(args.archive) as root:
            metadata = prepare_dataset(
                root,
                args.output_dir,
                seed=args.seed,
                source_kind="archive",
                source_name=str(args.archive),
                smoke=args.smoke,
                max_files_per_class=args.max_files_per_class,
                max_files=args.max_files,
                max_rows_per_file=args.max_rows_per_file,
                max_samples_per_class=args.max_samples_per_class,
            )
    else:
        metadata = prepare_dataset(
            args.data_dir,
            args.output_dir,
            seed=args.seed,
            smoke=args.smoke,
            max_files_per_class=args.max_files_per_class,
            max_files=args.max_files,
            max_rows_per_file=args.max_rows_per_file,
            max_samples_per_class=args.max_samples_per_class,
        )
    print(f"Wrote {metadata['sample_count']:,} samples to {Path(args.output_dir).resolve()}")
    print(f"Feature shape: {tuple(metadata['feature_shape'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
