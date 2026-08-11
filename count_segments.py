#!/usr/bin/env python3
"""Count paper-style PSD samples without feature extraction or training.

The report separates the unit before hopping from the unit after hopping:

* pre-hopping segment: one time row from one detected transmission;
* post-hopping candidate: one time row from one sequential frequency chunk.

This script uses the independent detector in ``prepare_dataset.py`` and does
not import the released author implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from prepare_dataset import (
    BOUNDARY_ROWS,
    CHUNK_BINS,
    LABELS,
    OCCUPANCY_DB,
    InputFile,
    _linear_amplitude,
    _occupied_intervals,
    build_site_noise,
    detect_transmissions,
    discover_files,
    extracted_archive,
    iter_hopping_chunks,
    sensor_groups,
)


PAPER_TOTAL = 134_490
PAPER_PER_CLASS = 22_415


def empty_counts() -> dict[str, int]:
    return {
        "files": 0,
        "files_with_detections": 0,
        "coarse_occupancy_intervals": 0,
        "detected_transmissions": 0,
        "pre_hop_rows_360": 0,
        "pre_hop_rows_all": 0,
        "pre_hop_rows_gt1_360": 0,
        "pre_hop_rows_gt1_all": 0,
        "hopping_chunks": 0,
        "post_hop_rows_360": 0,
        "post_hop_rows_all": 0,
        "post_hop_rows_gt1_360": 0,
        "post_hop_rows_gt1_all": 0,
        "partial_chunks": 0,
    }


def count_segments(files: Sequence[InputFile]) -> dict[str, object]:
    """Count detections and row/chunk units using the preparation detector."""

    noise_by_sensor = build_site_noise(files)
    per_class = {label: empty_counts() for label in LABELS}
    totals = empty_counts()
    file_rows: list[dict[str, object]] = []
    statuses = Counter()
    widths = Counter()

    for info in files:
        label = info.technology
        if label not in per_class:
            statuses["unrecognized_technology"] += 1
            continue
        per_class[label]["files"] += 1
        totals["files"] += 1
        file_row = {
            "source_file": info.source_file,
            "technology": label,
            "sensor": info.sensor,
            "status": "",
            "rows": 0,
            "frequency_bins": 0,
            "coarse_occupancy_intervals": 0,
            "detected_transmissions": 0,
            "pre_hop_rows_360": 0,
            "pre_hop_rows_all": 0,
            "hopping_chunks": 0,
            "post_hop_rows_360": 0,
            "post_hop_rows_all": 0,
        }
        file_rows.append(file_row)

        try:
            data = np.load(info.path, allow_pickle=False)
            file_row["rows"], file_row["frequency_bins"] = map(int, data.shape)
            if data.ndim != 2 or 0 in data.shape:
                file_row["status"] = "invalid_shape"
                statuses["invalid_shape"] += 1
                continue
            if not np.all(np.isfinite(data)):
                file_row["status"] = "nonfinite_data"
                statuses["nonfinite_data"] += 1
                continue
            noise_db = noise_by_sensor.get(info.sensor)
            if noise_db is None:
                file_row["status"] = "noise_unavailable"
                statuses["noise_unavailable"] += 1
                continue

            rows_for_detection = min(BOUNDARY_ROWS, data.shape[0])
            averaged = _linear_amplitude(data[:rows_for_detection, :]).mean(axis=0)
            noise_amplitude = float(_linear_amplitude(np.asarray([noise_db]))[0])
            occupancy_threshold = noise_amplitude * float(10.0 ** (OCCUPANCY_DB / 20.0))
            coarse_intervals = list(_occupied_intervals(averaged, occupancy_threshold))
            file_row["coarse_occupancy_intervals"] = len(coarse_intervals)
            per_class[label]["coarse_occupancy_intervals"] += len(coarse_intervals)
            totals["coarse_occupancy_intervals"] += len(coarse_intervals)

            detections = detect_transmissions(data, noise_db, BOUNDARY_ROWS)
            file_row["detected_transmissions"] = len(detections)
            per_class[label]["detected_transmissions"] += len(detections)
            totals["detected_transmissions"] += len(detections)
            if len(detections):
                per_class[label]["files_with_detections"] += 1
                totals["files_with_detections"] += 1

            for start, end in detections:
                transmission = data[:, int(start) : int(end) + 1]
                rows_all = int(transmission.shape[0])
                rows_360 = min(BOUNDARY_ROWS, rows_all)
                width = int(transmission.shape[1])
                widths[width] += 1
                for counts in (per_class[label], totals):
                    counts["pre_hop_rows_all"] += rows_all
                    counts["pre_hop_rows_360"] += rows_360
                    if width > 1:
                        counts["pre_hop_rows_gt1_all"] += rows_all
                        counts["pre_hop_rows_gt1_360"] += rows_360

                file_row["pre_hop_rows_all"] += rows_all
                file_row["pre_hop_rows_360"] += rows_360

                chunks = list(iter_hopping_chunks(transmission, CHUNK_BINS))
                if width > CHUNK_BINS and width % CHUNK_BINS:
                    per_class[label]["partial_chunks"] += 1
                    totals["partial_chunks"] += 1
                for _, _, chunk in chunks:
                    chunk_width = int(chunk.shape[1])
                    chunk_rows_all = int(chunk.shape[0])
                    chunk_rows_360 = min(BOUNDARY_ROWS, chunk_rows_all)
                    for counts in (per_class[label], totals):
                        counts["hopping_chunks"] += 1
                        counts["post_hop_rows_all"] += chunk_rows_all
                        counts["post_hop_rows_360"] += chunk_rows_360
                        if chunk_width > 1:
                            counts["post_hop_rows_gt1_all"] += chunk_rows_all
                            counts["post_hop_rows_gt1_360"] += chunk_rows_360
                    file_row["hopping_chunks"] += 1
                    file_row["post_hop_rows_all"] += chunk_rows_all
                    file_row["post_hop_rows_360"] += chunk_rows_360

            file_row["status"] = "ok" if len(detections) else "no_detections"
            statuses[file_row["status"]] += 1
        except (OSError, TypeError, ValueError) as error:
            file_row["status"] = f"error:{type(error).__name__}"
            statuses[file_row["status"]] += 1

    return {
        "parameters": {
            "boundary_rows": BOUNDARY_ROWS,
            "hopping_chunk_bins": CHUNK_BINS,
            "pre_hop_definition": "one time row per detected transmission",
            "post_hop_definition": "one time row per sequential hopping chunk",
        },
        "paper_reference": {
            "total_samples": PAPER_TOTAL,
            "samples_per_class": PAPER_PER_CLASS,
            "training_samples": 108_000,
            "testing_samples": 26_490,
        },
        "totals": totals,
        "per_class": per_class,
        "statuses": dict(statuses),
        "detected_widths": {str(width): count for width, count in sorted(widths.items())},
        "files": file_rows,
    }


def _comparison(count: int) -> dict[str, float | int]:
    return {
        "count": count,
        "paper_total": PAPER_TOTAL,
        "difference": count - PAPER_TOTAL,
        "ratio_to_paper": count / PAPER_TOTAL if PAPER_TOTAL else 0.0,
    }


def write_report(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = report["totals"]
    assert isinstance(totals, dict)
    report["comparisons"] = {
        key: _comparison(int(totals[key]))
        for key in (
            "pre_hop_rows_360",
            "pre_hop_rows_all",
            "pre_hop_rows_gt1_360",
            "pre_hop_rows_gt1_all",
            "post_hop_rows_360",
            "post_hop_rows_all",
            "post_hop_rows_gt1_360",
            "post_hop_rows_gt1_all",
        )
    }
    (output_dir / "segment_count_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    fields = [
        "technology",
        "files",
        "files_with_detections",
        "coarse_occupancy_intervals",
        "detected_transmissions",
        "pre_hop_rows_360",
        "pre_hop_rows_all",
        "pre_hop_rows_gt1_360",
        "pre_hop_rows_gt1_all",
        "hopping_chunks",
        "post_hop_rows_360",
        "post_hop_rows_all",
        "post_hop_rows_gt1_360",
        "post_hop_rows_gt1_all",
        "partial_chunks",
    ]
    with (output_dir / "segment_count_by_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for technology in LABELS:
            writer.writerow({"technology": technology, **report["per_class"][technology]})
        writer.writerow({"technology": "TOTAL", **report["totals"]})

    with (output_dir / "segment_count_by_file.csv").open("w", newline="", encoding="utf-8") as stream:
        file_rows = report["files"]
        fields = list(file_rows[0]) if file_rows else []
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(file_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("segment_counts"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.archive:
        with extracted_archive(args.archive) as root:
            files = discover_files(root)
            report = count_segments(files)
            report["source"] = str(args.archive)
            write_report(report, args.output_dir)
    else:
        files = discover_files(args.data_dir)
        report = count_segments(files)
        report["source"] = str(args.data_dir)
        write_report(report, args.output_dir)

    totals = report["totals"]
    print(f"Detected transmissions: {totals['detected_transmissions']:,}")
    print(f"Pre-hopping rows (360): {totals['pre_hop_rows_360']:,}")
    print(f"Pre-hopping rows (all): {totals['pre_hop_rows_all']:,}")
    print(f"Post-hopping rows (360): {totals['post_hop_rows_360']:,}")
    print(f"Post-hopping rows (all): {totals['post_hop_rows_all']:,}")
    print(f"Paper reference total: {PAPER_TOTAL:,}")
    print(f"Reports written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
