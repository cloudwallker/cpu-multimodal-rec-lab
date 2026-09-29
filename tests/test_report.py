"""Saved-ranking fixtures test verification; they are not research results."""
import csv
import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import save_npz

from mmrec_lab.impute import build_neighbors, make_missing, impute_features
from mmrec_lab.io import array_hash, canonical_hash, file_hash, read_json, write_json
from mmrec_lab.metrics import evaluate_rankings
from mmrec_lab.study import experiment_matrix, BASE_CONFIG, source_manifest


@pytest.fixture
def saved_study(tmp_path):
    data_dir, study_dir = tmp_path / "dataset", tmp_path / "study"
    data_dir.mkdir()
    study_dir.mkdir()
    n = 25
    arrays = dict(train=np.array([[0, 0], [1, 1]], dtype=np.int64),
                  valid=np.array([[0, 1], [1, 2]], dtype=np.int64),
                  test=np.array([[0, i] for i in range(13, 25)] + [[1, 0]], dtype=np.int64),
                  vision=np.ones((n, 2), np.float32), text=np.ones((n, 2), np.float32),
                  tail=np.arange(n) >= 12, item_ids=np.arange(n), user_ids=np.array([100, 101]))
    np.savez(data_dir / "dataset.npz", **arrays)
    manifest = dict(dataset="saved-ranking-unit-fixture", n_users=2, n_items=n,
                    dataset_sha256=file_hash(data_dir / "dataset.npz"),
                    array_hashes={key: array_hash(value) for key, value in arrays.items()})
    manifest["identity"] = canonical_hash(manifest)
    write_json(data_dir / "manifest.json", manifest)
    protocol = dict(dataset_identity=manifest["identity"], runs=experiment_matrix(),
                    config={**BASE_CONFIG, "epochs": 2}, alpha=0.5, tau=4, source_files=source_manifest())
    protocol["identity"] = canonical_hash(protocol)
    write_json(study_dir / "frozen_protocol.json", protocol)
    neighbors = build_neighbors(arrays["train"], 2, n, 20)
    save_npz(study_dir / "neighbors.npz", neighbors)
    write_json(study_dir / "neighbors.json", dict(sha256=file_hash(study_dir / "neighbors.npz"),
                                                 train_hash=array_hash(arrays["train"])))
    truth = {0: set(range(13, 25)), 1: {0}}
    completed = []
    for condition in protocol["runs"]:
        path = study_dir / condition["run_id"]
        path.mkdir()
        missing = make_missing(n, arrays["tail"], condition["scenario"], condition["mask_seed"] or 0)
        support = np.asarray(neighbors @ (~missing).astype(np.int64)).ravel().astype(np.int64)
        if condition["scenario"] == "clean":
            support[:] = 0
        np.save(path / "missing.npy", missing)
        np.save(path / "support.npy", support)
        vision = arrays["vision"].copy()
        if condition["scenario"] != "clean":
            vision[missing] = 0
            vision, _ = impute_features(vision, missing, neighbors, condition["method"], alpha=.5, tau=4)
        meta = {**condition, "missing_hash": array_hash(missing), "imputation_seconds": 0.01,
                "vision_hash": array_hash(vision) if condition["modality"] != "text" else None,
                "text_hash": array_hash(arrays["text"]) if condition["modality"] != "image" else None,
                "alpha": .5 if condition["method"] == "fixed_mix" else None,
                "tau": 4 if condition["method"] == "support_mix" else None}
        write_json(path / "condition.json", meta)
        items = list(range(24, 4, -1)) if condition["method"] == "support_mix" else list(range(2, 22))
        rankings = [{"user": 0, "items": items}, {"user": 1, "items": [0] + list(range(3, 22))}]
        groups = dict(all=np.ones(n, bool), tail=arrays["tail"], missing=missing,
                      tail_missing=arrays["tail"] & missing)
        metrics = evaluate_rankings(rankings, truth, groups, k=20)
        write_json(path / "rankings.json", rankings)
        validation = evaluate_rankings([{"user": 0, "items": [1] + list(range(2, 21))},
                                        {"user": 1, "items": [2, 0] + list(range(3, 21))}],
                                       {0: {1}, 1: {2}}, groups)
        result = dict(seed=condition["seed"], test=metrics, validation=validation, best_validation_recall=1.,
                      config=protocol["config"], environment={"python": "unit-fixture"},
                      source_files={key: protocol["source_files"][key] for key in ["train.py", "model.py", "metrics.py", "io.py"]},
                      inputs={key: array_hash(arrays[key]) for key in ["train", "valid", "test", "tail"]},
                      artifacts={"rankings.json": file_hash(path / "rankings.json")},
                      best_epoch=2, epochs_completed=2, is_converged=False, stop_reason="epoch_limit",
                      resources=dict(init_seconds=.1, train_seconds=.2, eval_seconds=.1,
                                     checkpoint_seconds=.1, total_seconds=.5, peak_rss_bytes=1024**2,
                                     threads=1, device="cpu"))
        result["inputs"].update(missing=array_hash(missing), vision=meta["vision_hash"], text=meta["text_hash"])
        with (path / "per_user_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["user", "group", "recall", "ndcg", "positives"])
            writer.writeheader()
            writer.writerows(metrics["rows"])
        result["artifacts"]["per_user_metrics.csv"] = file_hash(path / "per_user_metrics.csv")
        identity_payload = dict(schema_version=1, config=result["config"], seed=condition["seed"], test_enabled=True,
                                n_users=2, n_items=n, inputs=result["inputs"], dataset_manifest=manifest,
                                source_files=result["source_files"], environment=result["environment"])
        result["identity"] = canonical_hash(identity_payload)
        write_json(path / "run_metadata.json", {**identity_payload, "identity": result["identity"], "status": "completed"})
        write_json(path / "result.json", result)
        completed.append(dict(run_id=condition["run_id"], result_identity=result["identity"]))
    write_json(study_dir / "study_status.json", dict(status="complete", completed=45, planned=45, runs=completed,
                                                    protocol_identity=protocol["identity"]))
    return data_dir, study_dir, tmp_path / "report"


def report_function():
    assert importlib.util.find_spec("mmrec_lab.report") is not None, "Saved-ranking report implementation is missing"
    return importlib.import_module("mmrec_lab.report").build_report


def test_report_recomputes_all_45_runs_and_exports_paired_summary(saved_study):
    build_report = report_function()
    result = build_report(*saved_study)
    assert result["recomputed_runs"] == 45
    primary = result["primary_comparison"]
    assert primary["differences"] == [0.25, 0.25, 0.25]
    assert primary["mean"] == 0.25 and primary["std"] == 0
    report_dir = saved_study[2]
    for name in ["per_seed.csv", "summary.csv", "paired_differences.csv", "resource_usage.csv",
                 "cases.csv", "cases.md", "effects.png", "support.png", "resources.png", "report_manifest.json"]:
        assert (report_dir / name).is_file()
    with (report_dir / "cases.csv").open(encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))
    assert cases and all(row["user_index"] == "0" for row in cases)
    assert "original_user" not in (report_dir / "cases.csv").read_text(encoding="utf-8")
    summary = [r for r in result["summary"] if r["scenario"] == "tail30" and
               r["method"] == "support_mix" and r["group"] == "tail" and r["metric"] == "recall"]
    assert summary[0]["mean"] == 1 and summary[0]["std"] == 0 and summary[0]["n"] == 3


@pytest.mark.parametrize("corruption", ["metric", "blocked", "users", "missing_run", "mask", "support",
                                        "identity", "configuration", "features", "validation"])
def test_report_rejects_corrupt_or_incomplete_studies(saved_study, corruption):
    build_report = report_function()
    data_dir, study_dir, output = saved_study
    output.mkdir()
    write_json(output / "report_manifest.json", {"fixture": "previous success marker"})
    run = study_dir / "tail30_support_mix_s203"
    result = read_json(run / "result.json")
    if corruption == "metric":
        result["test"]["summary"]["all"]["recall"] += 0.1
        write_json(run / "result.json", result)
    elif corruption in ["blocked", "users"]:
        rankings = read_json(run / "rankings.json")
        if corruption == "blocked":
            rankings[0]["items"][0] = 0
        else:
            rankings.pop()
        write_json(run / "rankings.json", rankings)
        result["test"]["rankings"] = rankings
        result["artifacts"]["rankings.json"] = file_hash(run / "rankings.json")
        write_json(run / "result.json", result)
    elif corruption == "missing_run":
        (run / "result.json").unlink()
    elif corruption == "mask":
        mask = np.load(run / "missing.npy")
        mask[:] = False
        mask[12:19] = True
        np.save(run / "missing.npy", mask)
        condition = read_json(run / "condition.json")
        condition["missing_hash"] = array_hash(mask)
        result["inputs"]["missing"] = array_hash(mask)
        write_json(run / "condition.json", condition)
        write_json(run / "result.json", result)
        metadata = read_json(run / "run_metadata.json")
        metadata["inputs"] = result["inputs"]
        metadata["identity"] = canonical_hash({k: v for k, v in metadata.items() if k not in ["identity", "status"]})
        result["identity"] = metadata["identity"]
        status = read_json(study_dir / "study_status.json")
        for record in status["runs"]:
            if record["run_id"] == run.name:
                record["result_identity"] = result["identity"]
        write_json(study_dir / "study_status.json", status)
        write_json(run / "run_metadata.json", metadata)
        write_json(run / "result.json", result)
    elif corruption == "support":
        support = np.load(run / "support.npy")
        support[12] = 1
        np.save(run / "support.npy", support)
    elif corruption == "validation":
        result["validation"] = None
        write_json(run / "result.json", result)
    else:
        metadata = read_json(run / "run_metadata.json")
        if corruption == "identity":
            metadata["status"] = "running"
        elif corruption == "configuration":
            metadata["config"]["dropout"] = .12
            result["config"] = metadata["config"]
        else:
            result["inputs"]["vision"] = "0" * 64
            metadata["inputs"] = result["inputs"]
            condition = read_json(run / "condition.json")
            condition["vision_hash"] = "0" * 64
            write_json(run / "condition.json", condition)
        metadata["identity"] = canonical_hash({k: v for k, v in metadata.items() if k not in ["identity", "status"]})
        result["identity"] = metadata["identity"]
        status = read_json(study_dir / "study_status.json")
        for record in status["runs"]:
            if record["run_id"] == run.name:
                record["result_identity"] = result["identity"]
        write_json(study_dir / "study_status.json", status)
        write_json(run / "run_metadata.json", metadata)
        write_json(run / "result.json", result)
    expected_error = {"metric": "summary mismatch", "blocked": "blocked", "users": "ranked users",
                      "missing_run": "Missing completed run", "mask": "mask", "support": "support differs",
                      "identity": "run metadata", "configuration": "configuration", "features": "feature hash",
                      "validation": "validation"}
    with pytest.raises((ValueError, FileNotFoundError), match=expected_error[corruption]):
        build_report(data_dir, study_dir, output)
    assert not (output / "report_manifest.json").exists()
