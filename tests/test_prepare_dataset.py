import csv
import json
import tarfile
from pathlib import Path

import numpy as np

import prepare_dataset as preparation


def _records(per_sensor=4):
    records = []
    labels = preparation.LABELS
    for sensor_index in range(per_sensor):
        for label in labels:
            records.append(
                {
                    "technology": label,
                    "sensor": f"sensor-{sensor_index}",
                    "sensor_group": f"sensor-{sensor_index}",
                }
            )
    return records


def test_hopping_is_sequential_and_discards_wide_tail():
    transmission = np.arange(2 * 431, dtype=np.float32).reshape(2, 431)
    chunks = list(preparation.iter_hopping_chunks(transmission))

    assert [(start, end) for start, end, _ in chunks] == [(0, 215), (215, 430)]
    assert [chunk.shape for _, _, chunk in chunks] == [(2, 215), (2, 215)]
    assert np.array_equal(chunks[1][2], transmission[:, 215:430])

    narrow = np.ones((3, 17), dtype=np.float32)
    narrow_chunks = list(preparation.iter_hopping_chunks(narrow))
    assert [(start, end, chunk.shape) for start, end, chunk in narrow_chunks] == [
        (0, 17, (3, 17))
    ]


def test_splits_are_deterministic_stratified_and_sensor_disjoint():
    records = _records()
    first_random = preparation.random_class_split(records, seed=7)
    second_random = preparation.random_class_split(records, seed=7)
    sensor_split = preparation.sensor_disjoint_split(records, seed=7)

    assert np.array_equal(first_random, second_random)
    for label in preparation.LABELS:
        indices = [i for i, record in enumerate(records) if record["technology"] == label]
        assert set(first_random[indices]) == {"train", "test"}
        assert set(sensor_split[indices]) == {"train", "test"}
    train_sensors = {records[i]["sensor"] for i, value in enumerate(sensor_split) if value == "train"}
    test_sensors = {records[i]["sensor"] for i, value in enumerate(sensor_split) if value == "test"}
    assert train_sensors.isdisjoint(test_sensors)


def test_random_split_is_segment_level_except_for_duplicate_sources():
    records = []
    for label in preparation.LABELS:
        for index in range(10):
            records.append(
                {
                    "technology": label,
                    "sensor": "sensor-a",
                    "source_file": f"{label}/ordinary.npy",
                    "payload_hash": f"unique-{label}",
                }
            )
        for source in ("duplicate-a.npy", "duplicate-b.npy"):
            records.append(
                {
                    "technology": label,
                    "sensor": source,
                    "source_file": source,
                    "payload_hash": f"duplicate-{label}",
                }
            )

    assignments = preparation.random_class_split(records, seed=19)
    for label in preparation.LABELS:
        ordinary = [
            index
            for index, record in enumerate(records)
            if record["technology"] == label and record["source_file"].endswith("ordinary.npy")
        ]
        duplicates = [
            index
            for index, record in enumerate(records)
            if record["technology"] == label and "duplicate-" in record["source_file"]
        ]
        assert set(assignments[ordinary]) == {"train", "test"}
        assert len(set(assignments[duplicates])) == 1


def test_linked_duplicate_sensors_share_sensor_partition():
    records = _records()
    for record in records:
        if record["sensor"] in {"sensor-0", "sensor-1"}:
            record["sensor_group"] = "linked-0-1"
    assignments = preparation.sensor_disjoint_split(records, seed=11)
    linked = [i for i, record in enumerate(records) if record["sensor_group"] == "linked-0-1"]
    assert len(set(assignments[linked])) == 1


def test_identical_payloads_link_sensors():
    first = preparation.InputFile(Path("a/sensor-a/day/a.npy"), "a.npy", "fm", "sensor-a", "same")
    second = preparation.InputFile(Path("b/sensor-b/day/b.npy"), "b.npy", "fm", "sensor-b", "same")
    groups = preparation.sensor_groups([first, second])
    assert groups["sensor-a"] == groups["sensor-b"]


def test_smoke_selection_has_all_classes_on_two_sensors(tmp_path):
    data_root = tmp_path / "data"
    for sensor in ("sensor-a", "sensor-b", "sensor-c"):
        date_dir = data_root / sensor / "day"
        date_dir.mkdir(parents=True)
        for label in preparation.LABELS:
            np.save(date_dir / f"SpectrumBands_1_2_{label}_{sensor}.npy", np.zeros((2, 215)))

    selected = preparation.select_smoke_files(preparation.discover_files(data_root))

    assert len(selected) == 12
    assert {info.technology for info in selected} == set(preparation.LABELS)
    assert len({info.sensor for info in selected}) == 2
    for label in preparation.LABELS:
        assert len({info.sensor for info in selected if info.technology == label}) == 2


def test_archive_counts_without_feature_generation(tmp_path):
    source = tmp_path / "source"
    for sensor in ("one", "two"):
        date_dir = source / sensor / "day"
        date_dir.mkdir(parents=True)
        np.save(date_dir / f"SpectrumBands_1_2_fm_x_{sensor}.npy", np.zeros((2, 215)))
    np.save(source / "one" / "day" / "SpectrumBands_1_2_dab_x.npy", np.zeros((2, 215)))
    archive = tmp_path / "spectrum_bands.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="opt/shared/spectrum_bands_2")

    assert preparation.count_archive_files(archive) == {"dab": 1, "fm": 2}


def test_preparation_writes_aligned_feature_and_segment_shapes(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    for sensor_index in range(4):
        date_dir = data_root / f"sensor-{sensor_index}" / "day"
        date_dir.mkdir(parents=True)
        for label_index, label in enumerate(preparation.LABELS):
            values = np.full((3, 430), -60.0 + sensor_index * 0.01 + label_index * 0.001)
            values[:, 300:340] = -20.0 + sensor_index * 0.01
            np.save(date_dir / f"SpectrumBands_1_2_{label}_x.npy", values)

    def fake_features(data):
        values = np.mean(data, axis=1, keepdims=True)
        features = np.repeat(values, 33, axis=1).astype(object)
        return features, [f"f{i}" for i in range(33)]

    monkeypatch.setattr(preparation, "extract_33_features", fake_features)
    output = tmp_path / "prepared"
    metadata = preparation.prepare_dataset(
        data_root,
        output,
        smoke=True,
        max_rows_per_file=2,
        max_samples_per_class=4,
    )

    features = np.load(output / "features.npy")
    with (output / "segments.csv").open(newline="") as stream:
        segments = list(csv.DictReader(stream))
    with (output / "feature_columns.json").open() as stream:
        columns = json.load(stream)
    with (output / "files.csv").open(newline="") as stream:
        files = list(csv.DictReader(stream))

    assert features.shape == (len(segments), 33)
    assert len(columns) == 33
    assert all(int(row["chunk_width"]) == 40 for row in segments)
    assert all(int(row["time_row"]) < 2 for row in segments)
    assert all(int(row["processed_rows"]) <= 2 for row in files)
    assert metadata["feature_shape"] == [len(segments), 33]
    assert metadata["smoke"] is True
    assert metadata["files_selected"] == 12
    assert len(metadata["selected_sensors"]) == 2
    assert {row["technology"] for row in segments} == set(preparation.LABELS)
    for label in preparation.LABELS:
        class_rows = [row for row in segments if row["technology"] == label]
        assert len(class_rows) <= 4
        assert {row["random_split"] for row in class_rows} == {"train", "test"}
        assert {row["sensor_split"] for row in class_rows} == {"train", "test"}
