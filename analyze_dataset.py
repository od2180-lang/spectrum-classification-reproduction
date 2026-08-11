#!/usr/bin/env python3
"""Emit concise Markdown documentation for the published spectrum matrices."""

from __future__ import annotations

import argparse
import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


TECHNOLOGIES = ("dab", "dvbt", "fm", "gsm", "lte", "tetra")
DEFAULT_DATA_DIR = Path("data")
TECHNOLOGY_RE = re.compile(
    r"(?:^|_)(dab|dvbt|fm|gsm|lte|tetra)(?:_|\.|$)", re.IGNORECASE
)
FREQUENCY_PAIR_RE = re.compile(r"(?<!\d)(\d+)[_-](\d+)(?!\d)")


@dataclass(frozen=True)
class MatrixRecord:
    """Metadata for one source matrix, without loading its values into memory."""

    source: str
    technology: str | None
    sensor: str | None
    frequency_pair: tuple[int, int] | None
    rows: int
    frequency_bins: int


def parse_technology(source: str | Path) -> str | None:
    """Read the technology label without relying on frequency metadata."""
    name = Path(source).name
    match = TECHNOLOGY_RE.search(name)
    if match is None:
        match = TECHNOLOGY_RE.search(str(source))
    return match.group(1).lower() if match else None


def parse_frequency_pair(source: str | Path) -> tuple[int, int] | None:
    """Return the first valid pair after the technology label, or ``None``."""
    name = Path(source).name
    technology_match = TECHNOLOGY_RE.search(name)
    if technology_match is None:
        return None

    for match in FREQUENCY_PAIR_RE.finditer(name[technology_match.end():]):
        start, end = (int(value) for value in match.groups())
        if start < end:
            return start, end
    return None


def parse_filename(fname: str | Path) -> tuple[int, int] | None:
    """Compatibility wrapper for callers that used the old parser name."""
    return parse_frequency_pair(fname)


def sensor_from_source(source: str | Path) -> str | None:
    """Get the sensor directory from the published sensor/date/file layout."""
    path = Path(source)
    return path.parent.parent.name if len(path.parts) >= 3 else None


def iter_data_arrays(data_dir: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield arrays from a directory tree in stable path order."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    for path in sorted(data_dir.rglob("*.npy")):
        yield path.as_posix(), np.load(path, mmap_mode="r", allow_pickle=False)


def iter_archive_arrays(archive_path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield arrays from an uncompressed or compressed tar archive."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive_path}")

    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".npy"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted:
                payload = extracted.read()
            yield member.name, np.load(io.BytesIO(payload), allow_pickle=False)


def collect_records(
    arrays: Iterator[tuple[str, np.ndarray]],
) -> tuple[list[MatrixRecord], int]:
    """Collect shape metadata and return the count of non-matrix arrays."""
    records = []
    invalid_arrays = 0
    for source, array in arrays:
        if array.ndim != 2:
            invalid_arrays += 1
            continue
        records.append(
            MatrixRecord(
                source=source,
                technology=parse_technology(source),
                sensor=sensor_from_source(source),
                frequency_pair=parse_frequency_pair(source),
                rows=int(array.shape[0]),
                frequency_bins=int(array.shape[1]),
            )
        )
    return records, invalid_arrays


def number(value: int) -> str:
    return f"{value:,}"


def render_markdown(records: list[MatrixRecord], invalid_arrays: int, source: str) -> str:
    """Render the dataset facts without deriving recording duration."""
    technology_records = {
        technology: [record for record in records if record.technology == technology]
        for technology in TECHNOLOGIES
    }
    stored_rows = sum(record.rows for record in records)
    usable_rows = sum(
        record.rows for record in records if record.frequency_bins > 0
    )
    frequency_bins = sum(record.frequency_bins for record in records)
    empty_frequency = sorted(
        (record for record in records if record.frequency_bins == 0),
        key=lambda record: record.source,
    )
    sensors = {record.sensor for record in records if record.sensor}
    valid_frequency_pairs = sum(
        record.frequency_pair is not None for record in records
    )

    lines = [
        "# Dataset Analysis: Wireless Technology Classification",
        "",
        '**Paper:** "A Framework for Wireless Technology Classification using Crowdsensing Platforms"',
        f"**Analyzed source:** `{source}`",
        "",
        "## Source Matrices",
        "",
        "Each `.npy` file stores a two-dimensional PSD matrix with shape "
        "`(time, frequency)`. The first dimension stores PSD rows; the second "
        "stores frequency bins.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Source matrices | {number(len(records))} |",
        f"| Published sensor directories represented | {number(len(sensors))} |",
        f"| Stored rows | {number(stored_rows)} |",
        f"| Usable rows (`frequency_bins > 0`) | {number(usable_rows)} |",
        f"| Summed frequency-bin widths | {number(frequency_bins)} |",
        f"| Frequency-empty arrays | {number(len(empty_frequency))} |",
        f"| Valid filename frequency pairs | {number(valid_frequency_pairs)} / {number(len(records))} |",
        "",
        "Technology labels come from the technology token in each filename. "
        "Malformed frequency metadata does not remove a file from its technology "
        "count. A frequency pair is nullable when the filename has no valid "
        "increasing pair after that token.",
        "",
        "## Technology Counts",
        "",
        "| Technology | Files | Stored rows | Usable rows | Summed frequency bins |",
        "|---|---:|---:|---:|---:|",
    ]

    for technology in TECHNOLOGIES:
        technology_rows = technology_records[technology]
        lines.append(
            f"| {technology.upper().replace('DVBT', 'DVB-T')} | "
            f"{number(len(technology_rows))} | "
            f"{number(sum(record.rows for record in technology_rows))} | "
            f"{number(sum(record.rows for record in technology_rows if record.frequency_bins > 0))} | "
            f"{number(sum(record.frequency_bins for record in technology_rows))} |"
        )

    lines.extend(
        [
            "",
            "## Three Count Levels",
            "",
            "- **Source matrix:** one published `.npy` file containing time rows and frequency bins.",
            "- **Detected transmission:** a region found by the transmission detector inside a source matrix; its count depends on detector behavior.",
            "- **Classifier sample:** a feature vector made from a PSD row and, where applicable, a hopping chunk of a detected transmission.",
            "",
            "The 98,262 usable source rows are not detected-transmission or classifier-sample counts. Detection can produce zero or multiple transmissions per matrix, and hopping can produce multiple classifier samples from one row.",
            "",
            "## Paper Reference",
            "",
            "The paper reports 108,000 training PSD segments and 26,490 testing PSD segments. Those are paper experiment counts, not counts inferred from the 232 source matrices. The paper also reports 282 sensor-hours; this report repeats that published figure and does not estimate duration from rows, frequency widths, or hopping chunks.",
            "",
            "## Reproduction Pipeline",
            "",
            "`prepare_dataset.py` implements the independent processing path used by this reproduction:",
            "",
            "1. Estimate a sensor noise floor from the quietest available 215-bin block.",
            "2. Detect occupied transmission boundaries with fixed, class-independent parameters.",
            "3. Preserve transmissions up to 215 bins and split wider transmissions into sequential, non-overlapping 215-bin chunks from the first detected bin.",
            "4. Extract one 33-feature classifier sample for each finite time-row/chunk pair.",
            "5. Store a class-stratified segment-level 80/20 split and a sensor-disjoint grouped 80/20 split.",
            "",
            "The detector and hopping implementation does not import or execute the released author code. Technology labels provide ground truth only; detection does not use class-specific width, frequency, SNR, or entropy gates.",
            "",
            "Run `python3 prepare_dataset.py --archive spectrum_bands.tar.gz --output-dir prepared_dataset`. The natural population can contain millions of candidates because the paper selected 134,490 balanced samples rather than using every valid detected row/chunk pair.",
        ]
    )

    if empty_frequency:
        lines.extend(
            [
                "",
                "## Frequency-Empty Arrays",
                "",
            ]
        )
        for record in empty_frequency:
            lines.append(f"- `{Path(record.source).name}` (`{record.rows} x 0`)")

    if invalid_arrays:
        lines.extend(
            [
                "",
                f"Skipped non-2D arrays: {number(invalid_arrays)}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Analysis Script",
            "",
            "Run `analyze_dataset.py --archive spectrum_bands.tar.gz` for the archive or `analyze_dataset.py --data-dir <directory>` for an extracted tree.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--archive", type=Path, help="path to a .tar or .tar.gz archive")
    source_group.add_argument("--data-dir", type=Path, help="directory containing extracted .npy files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.archive:
        source = args.archive
        arrays = iter_archive_arrays(source)
    else:
        source = args.data_dir or DEFAULT_DATA_DIR
        arrays = iter_data_arrays(source)

    records, invalid_arrays = collect_records(arrays)
    print(render_markdown(records, invalid_arrays, str(source)), end="")


if __name__ == "__main__":
    main()
