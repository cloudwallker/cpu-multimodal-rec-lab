"""Tiny synthetic fixtures verify timing machinery, never research timings."""
import csv

import numpy as np
import pytest

from test_data import make_raw
from mmrec_lab import benchmark
from mmrec_lab.data import prepare_dataset
from mmrec_lab.io import canonical_hash, file_hash, read_json, write_json
from mmrec_lab.study import source_manifest


@pytest.fixture
def benchmark_fixture(tmp_path):
    raw, dataset, output = tmp_path / "raw", tmp_path / "dataset", tmp_path / "benchmark"
    make_raw(raw)
    manifest = prepare_dataset(raw, dataset, target_items=8, seed=20260929)
    protocol = dict(dataset_identity=manifest["identity"], config=dict(knn_k=2, num_threads=1),
                    alpha=.5, tau=4, source_files=source_manifest())
    protocol["identity"] = canonical_hash(protocol)
    path = tmp_path / "protocol.json"
    write_json(path, protocol)
    return dataset, path, output, protocol


def test_actual_cpu_components_warmup_scope_and_148_recorded_rows(benchmark_fixture, monkeypatch):
    dataset, protocol_path, output, protocol = benchmark_fixture
    original_neighbors = benchmark.build_neighbors
    original_imputation = benchmark.impute_features
    original_knn = benchmark.FreedomModel._knn_graph
    calls = dict(neighbors=0, imputation=0, knn=0)

    def neighbors(*args, **kwargs):
        calls["neighbors"] += 1
        return original_neighbors(*args, **kwargs)

    def imputation(*args, **kwargs):
        calls["imputation"] += 1
        return original_imputation(*args, **kwargs)

    def knn(context, weight):
        calls["knn"] += 1
        # Exercise the real routine with no device/model/embedding attributes.
        assert set(vars(context)) == {"n_items", "knn_k"}
        result = original_knn(context, weight)
        assert result.device.type == "cpu" and not result.requires_grad
        return result

    monkeypatch.setattr(benchmark, "build_neighbors", neighbors)
    monkeypatch.setattr(benchmark, "impute_features", imputation)
    monkeypatch.setattr(benchmark.FreedomModel, "_knn_graph", knn)
    manifest = benchmark.benchmark_components(dataset, protocol_path, output, repeats=2)
    with (output / "component_timings.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 148 and manifest["rows"] == 148
    seconds = np.array([float(row["seconds"]) for row in rows])
    assert np.isfinite(seconds).all() and np.all(seconds >= 0)
    assert calls == dict(neighbors=3, imputation=108, knn=111)
    assert {row["repetition"] for row in rows} == {"1", "2"}
    assert sum(row["component"] == "train_cointeraction_graph" for row in rows) == 2
    assert sum(row["component"] == "text_knn_graph" for row in rows) == 2
    assert sum(row["component"] == "imputation_function" for row in rows) == 72
    assert sum(row["component"] == "image_knn_graph" for row in rows) == 72
    assert manifest["device"] == "cpu" and manifest["threads"] == 1
    assert manifest["protocol_identity"] == protocol["identity"]
    assert manifest["dataset_identity"] == protocol["dataset_identity"]
    assert manifest["csv_sha256"] == file_hash(output / "component_timings.csv")
    assert manifest["benchmark_source_sha256"] == file_hash(benchmark.__file__)
    assert manifest["source_files"] == protocol["source_files"]
    assert manifest["model_source_sha256"] == protocol["source_files"]["model.py"]
    assert manifest["impute_source_sha256"] == protocol["source_files"]["impute.py"]
    assert read_json(output / "component_benchmark.json") == manifest


@pytest.mark.parametrize("corruption", ["protocol_checksum", "dataset_identity", "frozen_model", "frozen_impute"])
def test_benchmark_rejects_wrong_protocol_or_changed_algorithms(benchmark_fixture, corruption):
    dataset, protocol_path, output, protocol = benchmark_fixture
    if corruption == "protocol_checksum":
        protocol["tau"] = 16
    elif corruption == "dataset_identity":
        protocol["dataset_identity"] = "0" * 64
        protocol["identity"] = canonical_hash({k: v for k, v in protocol.items() if k != "identity"})
    else:
        key = "model.py" if corruption == "frozen_model" else "impute.py"
        protocol["source_files"][key] = "0" * 64
        protocol["identity"] = canonical_hash({k: v for k, v in protocol.items() if k != "identity"})
    write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="checksum|frozen|source"):
        benchmark.benchmark_components(dataset, protocol_path, output, repeats=2)
    assert not (output / "component_benchmark.json").exists()
