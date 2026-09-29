"""Verify every saved study ranking before exporting reproducible summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from .data import load_dataset
from .impute import build_neighbors, make_missing, impute_features
from .io import array_hash, canonical_hash, file_hash, read_json, write_json
from .metrics import evaluate_rankings
from .study import METHODS, experiment_matrix


GROUPS = ("all", "tail", "missing", "tail_missing")
METRICS = ("recall", "ndcg")


def _sets(pairs):
    result = {}
    for user, item in pairs:
        result.setdefault(int(user), set()).add(int(item))
    return result


def _csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values):
    valid = [float(value) for value in values if value is not None]
    return {"mean": float(np.mean(valid)) if valid else None,
            "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else None, "n": len(valid)}


def _aggregate(rows, keys, value="value"):
    buckets = {}
    for row in rows:
        buckets.setdefault(tuple(row[key] for key in keys), []).append(row[value])
    return [{**dict(zip(keys, key)), **_stats(values)} for key, values in sorted(buckets.items())]


def extractdiff(per_seed_rows: list[dict], reference: str = "fixed_mix") -> list[dict]:
    """Paired support-minus-reference differences, preserving all four groups.

    Values are fractions. Percentage-point differences are exported separately;
    a group with no eligible positives remains None, not an invented zero.
    """
    lookup = {(r["scenario"], r["seed"], r["method"], r["group"], r["metric"]): r
              for r in per_seed_rows if r["scenario"] != "clean"}
    result = []
    for scenario in ["uniform30", "tail30"]:
        for seed in [201, 202, 203]:
            for group in GROUPS:
                for metric in METRICS:
                    key = (scenario, seed, "support_mix", group, metric)
                    ref = (scenario, seed, reference, group, metric)
                    if key not in lookup or ref not in lookup:
                        raise ValueError("Incomplete paired three-seed comparison")
                    left, right = lookup[key]["value"], lookup[ref]["value"]
                    delta = None if left is None or right is None else left - right
                    result.append(dict(scenario=scenario, seed=seed, group=group, metric=metric,
                                       comparison="support_mix_minus_" + reference,
                                       support_value=left, reference_value=right,
                                       delta=delta, delta_percentage_points=None if delta is None else 100 * delta))
    return result


def _check_summary(actual, stored, run_id):
    if set(stored) != set(GROUPS):
        raise ValueError(f"{run_id}: stored metric groups differ")
    for group in GROUPS:
        if set(stored[group]) != {"recall", "ndcg", "users", "positives"}:
            raise ValueError(f"{run_id}: unexpected metric summary fields")
        for key, value in actual[group].items():
            saved = stored[group][key]
            if value is None:
                match = saved is None
            elif key in ("users", "positives"):
                match = value == saved
            else:
                match = saved is not None and np.isfinite(saved) and np.isclose(value, saved, rtol=1e-10, atol=1e-12)
            if not match:
                raise ValueError(f"{run_id}: recomputed summary mismatch for {group}/{key}")


def _check_rows(actual, stored, run_id):
    if len(actual) != len(stored):
        raise ValueError(f"{run_id}: per-user metric row count differs")
    keyed = {(r["user"], r["group"]): r for r in stored}
    if len(keyed) != len(stored) or set(keyed) != {(r["user"], r["group"]) for r in actual}:
        raise ValueError(f"{run_id}: per-user metric keys differ")
    for row in actual:
        saved = keyed[row["user"], row["group"]]
        if row["positives"] != saved["positives"] or any(
            not np.isclose(row[key], saved[key], rtol=1e-10, atol=1e-12) for key in METRICS):
            raise ValueError(f"{run_id}: per-user metrics mismatch")


def _figures(output, summary, supports, resources):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, filename in [("recall", "effects.png"), ("ndcg", "ndcg.png")]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for row_index, group in enumerate(["all", "tail"]):
            for col_index, scenario in enumerate(["clean", "uniform30", "tail30"]):
                ax = axes[row_index, col_index]
                selected = [r for r in summary if r["group"] == group and
                            r["scenario"] == scenario and r["metric"] == metric]
                order = ["both", "image", "text"] if scenario == "clean" else METHODS
                selected.sort(key=lambda r: order.index(r["modality"] if scenario == "clean" else r["method"]))
                labels = [r["modality"] if scenario == "clean" else r["method"] for r in selected]
                means = [r["mean"] for r in selected]
                errors = [r["std"] or 0 for r in selected]
                ax.bar(range(len(means)), means, yerr=errors, capsize=3, color="#347ea8")
                ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
                ax.set_title(f"{scenario} / {group}")
                ax.set_ylabel(metric.upper() + "@20")
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=.2)
        fig.suptitle("Mean and sample SD across three paired seeds")
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, scenario in zip(axes, ["uniform30", "tail30"]):
        for (sc, seed), record in sorted(supports.items()):
            if sc != scenario:
                continue
            values = record["support"][record["missing"]]
            hist = np.bincount(values, minlength=21)[:21]
            ax.plot(np.arange(21), hist / max(len(values), 1), marker=".", label=f"seed {seed}")
        ax.set_title(scenario)
        ax.set_xlabel("Observed direct neighbors of masked items")
        ax.set_ylabel("Fraction of masked items")
        ax.legend()
        ax.set_xticks([0, 5, 10, 15, 20])
    fig.savefig(output / "support.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    order = sorted(resources, key=lambda r: r["run_id"])
    x = np.arange(len(order))
    axes[0].bar(x, [r["total_seconds"] for r in order], color="#43877f")
    axes[0].set_ylabel("Training process total seconds")
    axes[1].bar(x, [r["peak_rss_bytes"] / 1024**2 for r in order], color="#bb7b49")
    axes[1].set_ylabel("Process peak RSS (MiB)")
    for ax in axes:
        ax.set_xlabel("45 runs in run_id order (see resource_usage.csv)")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "resources.png", dpi=170)
    plt.close(fig)


def _cases(details, truth, tail):
    candidates = []
    for seed in [201, 202, 203]:
        support = details[(seed, "support_mix")]
        fixed = details[(seed, "fixed_mix")]
        support_rows = {r["user"]: r for r in support["metrics"]["rows"] if r["group"] == "tail"}
        fixed_rows = {r["user"]: r for r in fixed["metrics"]["rows"] if r["group"] == "tail"}
        if set(support_rows) != set(fixed_rows):
            raise ValueError("Paired tail case users differ")
        support_rank = {r["user"]: r["items"] for r in support["metrics"]["rankings"]}
        fixed_rank = {r["user"]: r["items"] for r in fixed["metrics"]["rankings"]}
        for user in sorted(support_rows):
            left, right = support_rows[user], fixed_rows[user]
            positives = sorted(item for item in truth[user] if tail[item])
            masked = [item for item in positives if support["missing"][item]]
            candidates.append(dict(seed=seed, user_index=user,
                delta_recall=left["recall"] - right["recall"], support_recall=left["recall"],
                fixed_recall=right["recall"], delta_ndcg=left["ndcg"] - right["ndcg"],
                tail_positives=len(positives), masked_tail_positives=len(masked),
                masked_positive_support=json.dumps([int(support["support"][i]) for i in masked]),
                tail_positive_items=json.dumps(positives), masked_tail_items=json.dumps(masked),
                support_top20=json.dumps(support_rank[user]), fixed_top20=json.dumps(fixed_rank[user])))
    improving = sorted((r for r in candidates if r["delta_recall"] > 0),
                       key=lambda r: (-r["delta_recall"], r["seed"], r["user_index"]))[:5]
    worsening = sorted((r for r in candidates if r["delta_recall"] < 0),
                       key=lambda r: (r["delta_recall"], r["seed"], r["user_index"]))[:5]
    selected = [{"kind": "improving", **r} for r in improving] + [{"kind": "worsening", **r} for r in worsening]
    return selected, len(improving), len(worsening)


def build_report(dataset_dir, study_dir, output_dir) -> dict:
    """Fail on missing/corrupt runs, then export rankings-derived study artifacts."""
    study, output = Path(study_dir), Path(output_dir)
    previous_marker = output / "report_manifest.json"
    if previous_marker.exists():
        previous_marker.replace(output / "report_manifest.previous.json")
    dataset = load_dataset(dataset_dir)
    protocol = read_json(study / "frozen_protocol.json")
    if protocol["identity"] != canonical_hash({k: v for k, v in protocol.items() if k != "identity"}):
        raise ValueError("Frozen protocol checksum mismatch")
    if protocol["dataset_identity"] != dataset["manifest"]["identity"]:
        raise ValueError("Report dataset differs from the frozen study")
    planned = experiment_matrix()
    if protocol["runs"] != planned:
        raise ValueError("Report requires the predeclared complete 45-run matrix")
    status = read_json(study / "study_status.json")
    identities = {r["run_id"]: r["result_identity"] for r in status["runs"]}
    if (status.get("status") != "complete" or status["completed"] != 45 or status["planned"] != 45 or len(status["runs"]) != 45 or
        set(identities) != {r["run_id"] for r in planned} or status["protocol_identity"] != protocol["identity"]):
        raise ValueError("Incomplete three-seed study: all 45 completed runs are required")
    train_sets, valid_sets, truth = [_sets(dataset[name]) for name in ["train", "valid", "test"]]
    blocked = {u: train_sets.get(u, set()) | valid_sets.get(u, set()) for u in truth}
    if any(truth[u] & blocked[u] for u in truth):
        raise ValueError("Test positives overlap blocked train/validation items")
    n_items = dataset["n_items"]
    neighbors = load_npz(study / "neighbors.npz").tocsr()
    neighbor_meta = read_json(study / "neighbors.json")
    expected_neighbors = build_neighbors(dataset["train"], dataset["n_users"], n_items, 20)
    if (neighbor_meta["sha256"] != file_hash(study / "neighbors.npz") or
        neighbor_meta["train_hash"] != array_hash(dataset["train"]) or
        neighbors.shape != expected_neighbors.shape or (neighbors != expected_neighbors).nnz):
        raise ValueError("Saved neighbors/hash differ from the train-only graph")
    rows, resources, supports, case_details, input_files, support_summary = [], [], {}, {}, {}, []
    feature_hashes = {"vision": array_hash(dataset["vision"]), "text": array_hash(dataset["text"])}
    for expected in planned:
        run_id = expected["run_id"]
        path = study / run_id
        for name in ["condition.json", "result.json", "run_metadata.json", "rankings.json", "per_user_metrics.csv", "missing.npy", "support.npy"]:
            if not (path / name).is_file():
                raise FileNotFoundError(f"Missing completed run artifact: {run_id}/{name}")
            input_files[f"{run_id}/{name}"] = file_hash(path / name)
        condition, result = read_json(path / "condition.json"), read_json(path / "result.json")
        metadata = read_json(path / "run_metadata.json")
        metadata_identity = canonical_hash({k: v for k, v in metadata.items() if k not in ["identity", "status"]})
        if metadata.get("status") != "completed" or metadata["identity"] != metadata_identity or metadata["identity"] != result["identity"]:
            raise ValueError(f"{run_id}: run metadata identity/status mismatch")
        if metadata.get("test_enabled") is not True or metadata["dataset_manifest"] != dataset["manifest"]:
            raise ValueError(f"{run_id}: run metadata dataset/test setting differs")
        if metadata["n_users"] != dataset["n_users"] or metadata["n_items"] != n_items or metadata["seed"] != expected["seed"]:
            raise ValueError(f"{run_id}: run metadata vocabulary/seed mismatch")
        if result["config"] != protocol["config"] or metadata["config"] != result["config"]:
            raise ValueError(f"{run_id}: frozen configuration mismatch")
        for field in ["inputs", "source_files", "environment"]:
            if metadata[field] != result[field]:
                raise ValueError(f"{run_id}: run metadata {field} mismatch")
        if any(protocol["source_files"].get(name) != digest for name, digest in metadata["source_files"].items()):
            raise ValueError(f"{run_id}: research source differs from frozen protocol")
        if (condition.get("alpha") != (protocol["alpha"] if expected["method"] == "fixed_mix" else None) or
            condition.get("tau") != (protocol["tau"] if expected["method"] == "support_mix" else None)):
            raise ValueError(f"{run_id}: frozen imputation configuration mismatch")
        if any(condition.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{run_id}: condition differs from frozen run matrix")
        if result["identity"] != identities[run_id] or result["seed"] != expected["seed"] or result["test"] is None:
            raise ValueError(f"{run_id}: wrong result identity/seed or missing test results")
        if not isinstance(result.get("validation"), dict):
            raise ValueError(f"{run_id}: completed training must include validation results")
        for key in ["train", "valid", "test", "tail"]:
            if result["inputs"][key] != array_hash(dataset[key]):
                raise ValueError(f"{run_id}: dataset input hash mismatch for {key}")
        for key in ["vision", "text"]:
            if result["inputs"][key] != condition[key + "_hash"]:
                raise ValueError(f"{run_id}: modality feature ledger hash mismatch")
        if "rankings.json" not in result["artifacts"]:
            raise ValueError(f"{run_id}: ranking hash absent from result")
        for name, digest in result["artifacts"].items():
            if Path(name).name != name or not (path / name).is_file() or file_hash(path / name) != digest:
                raise ValueError(f"{run_id}: saved artifact hash mismatch: {name}")
        missing = np.load(path / "missing.npy", allow_pickle=False)
        support = np.load(path / "support.npy", allow_pickle=False)
        if missing.dtype != bool or missing.shape != (n_items,):
            raise ValueError(f"{run_id}: malformed missing mask")
        if support.shape != (n_items,) or support.dtype.kind not in "iu" or np.any(support < 0):
            raise ValueError(f"{run_id}: malformed support counts")
        if array_hash(missing) != condition["missing_hash"] or array_hash(missing) != result["inputs"]["missing"]:
            raise ValueError(f"{run_id}: saved mask hash mismatch")
        expected_missing = make_missing(n_items, dataset["tail"], expected["scenario"], expected["mask_seed"] or 0)
        if not np.array_equal(missing, expected_missing):
            raise ValueError(f"{run_id}: saved mask differs from its declared scenario/seed")
        expected_support = np.zeros(n_items, dtype=np.int64) if expected["scenario"] == "clean" else np.asarray(neighbors @ (~missing).astype(np.int64)).ravel()
        if not np.array_equal(support, expected_support):
            raise ValueError(f"{run_id}: saved support differs from visible train neighbors")
        vision_hash = feature_hashes["vision"]
        if expected["scenario"] != "clean":
            visible = dataset["vision"].copy()
            visible[missing] = 0
            imputed, _ = impute_features(visible, missing, neighbors, expected["method"],
                                         alpha=protocol["alpha"], tau=protocol["tau"])
            vision_hash = array_hash(imputed)
            del visible, imputed
        if expected["modality"] == "text":
            vision_hash = None
        text_hash = None if expected["modality"] == "image" else feature_hashes["text"]
        if vision_hash != condition["vision_hash"] or text_hash != condition["text_hash"]:
            raise ValueError(f"{run_id}: actual reconstructed feature hash mismatch")
        if expected["scenario"] != "clean":
            pair = expected["scenario"], expected["seed"]
            if pair in supports and (not np.array_equal(missing, supports[pair]["missing"]) or
                                     not np.array_equal(support, supports[pair]["support"])):
                raise ValueError(f"{run_id}: methods do not share paired masks/support")
            supports[pair] = {"missing": missing, "support": support}
        rankings = read_json(path / "rankings.json")
        users = [r["user"] for r in rankings]
        if len(users) != len(set(users)) or set(users) != set(truth):
            raise ValueError(f"{run_id}: ranked users differ from exactly the eligible test users")
        for record in rankings:
            user, items = record["user"], record["items"]
            if blocked[user] & set(items):
                raise ValueError(f"{run_id}: blocked train/validation item leaked into ranking")
            if len(items) != min(20, n_items - len(blocked[user])):
                raise ValueError(f"{run_id}: saved Top20 is truncated or oversized")
        if rankings != result["test"]["rankings"]:
            raise ValueError(f"{run_id}: standalone/result rankings differ")
        groups = dict(all=np.ones(n_items, bool), tail=dataset["tail"], missing=missing,
                      tail_missing=dataset["tail"] & missing)
        metrics = evaluate_rankings(rankings, truth, groups, k=20)
        _check_summary(metrics["summary"], result["test"]["summary"], run_id)
        _check_rows(metrics["rows"], result["test"]["rows"], run_id)
        with (path / "per_user_metrics.csv").open(encoding="utf-8") as stream:
            saved_rows = [{"user": int(r["user"]), "group": r["group"], "recall": float(r["recall"]),
                           "ndcg": float(r["ndcg"]), "positives": int(r["positives"])} for r in csv.DictReader(stream)]
        _check_rows(metrics["rows"], saved_rows, run_id)
        validation = result["validation"]
        validation_users = [r["user"] for r in validation["rankings"]]
        if len(validation_users) != len(set(validation_users)) or set(validation_users) != set(valid_sets):
            raise ValueError(f"{run_id}: validation ranking users differ")
        for record in validation["rankings"]:
            excluded = train_sets.get(record["user"], set())
            if excluded & set(record["items"]):
                raise ValueError(f"{run_id}: training item leaked into validation rankings")
            if len(record["items"]) != min(20, n_items - len(excluded)):
                raise ValueError(f"{run_id}: validation Top20 is truncated or oversized")
        recomputed_validation = evaluate_rankings(validation["rankings"], valid_sets, groups, k=20)
        _check_summary(recomputed_validation["summary"], validation["summary"], run_id + "/validation")
        _check_rows(recomputed_validation["rows"], validation["rows"], run_id + "/validation")
        if not np.isclose(recomputed_validation["summary"]["all"]["recall"], result["best_validation_recall"], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{run_id}: best validation score mismatch")
        for group in GROUPS:
            metric_summary = metrics["summary"][group]
            for metric in METRICS:
                rows.append(dict(run_id=run_id, scenario=expected["scenario"], modality=expected["modality"],
                                 method=expected["method"], seed=expected["seed"], group=group, metric=metric,
                                 value=metric_summary[metric], users=metric_summary["users"], positives=metric_summary["positives"]))
        resource = result["resources"]
        required = ["init_seconds", "train_seconds", "eval_seconds", "checkpoint_seconds", "total_seconds", "peak_rss_bytes", "threads"]
        if resource.get("device") != "cpu" or any(not np.isfinite(resource[k]) or resource[k] < 0 for k in required):
            raise ValueError(f"{run_id}: invalid CPU resources")
        resources.append(dict(run_id=run_id, scenario=expected["scenario"], method=expected["method"],
                              modality=expected["modality"], seed=expected["seed"],
                              **{k: resource[k] for k in required},
                              imputation_seconds=condition.get("imputation_seconds", 0),
                              epochs_completed=result["epochs_completed"], best_epoch=result["best_epoch"],
                              is_converged=result["is_converged"], stop_reason=result["stop_reason"]))
        if expected["scenario"] == "tail30" and expected["method"] in ["fixed_mix", "support_mix"]:
            case_details[expected["seed"], expected["method"]] = dict(metrics=metrics, missing=missing, support=support)
    summary = _aggregate(rows, ["scenario", "modality", "method", "group", "metric"])
    differences = extractdiff(rows, "fixed_mix") + extractdiff(rows, "observed_mean")
    comparisons = _aggregate(differences, ["scenario", "comparison", "group", "metric"], "delta")
    primary_values = [r["delta"] for r in differences if r["scenario"] == "tail30" and r["group"] == "tail" and
                      r["metric"] == "recall" and r["comparison"] == "support_mix_minus_fixed_mix"]
    if len(primary_values) != 3 or any(value is None for value in primary_values):
        raise ValueError("Primary tail comparison lacks all three evaluable seeds")
    primary = dict(scenario="tail30", comparison="support_mix_minus_fixed_mix", group="tail", metric="recall",
                   differences=primary_values, **_stats(primary_values))
    for (scenario, seed), record in sorted(supports.items()):
        values = record["support"][record["missing"]]
        counts = np.bincount(values, minlength=21)
        support_summary.append(dict(scenario=scenario, seed=seed, masked_items=len(values),
                                    mean=float(values.mean()) if len(values) else None,
                                    zero_fraction=float(np.mean(values == 0)) if len(values) else None,
                                    at_20_fraction=float(np.mean(values == 20)) if len(values) else None,
                                    histogram={str(i): int(v) for i, v in enumerate(counts)}))
    cases, n_improving, n_worsening = _cases(case_details, truth, dataset["tail"])
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "per_seed.csv", rows, list(rows[0]))
    _csv(output / "summary.csv", summary, list(summary[0]))
    _csv(output / "paired_differences.csv", differences, list(differences[0]))
    _csv(output / "paired_summary.csv", comparisons, list(comparisons[0]))
    _csv(output / "resource_usage.csv", resources, list(resources[0]))
    fields = list(cases[0]) if cases else ["kind", "seed", "user_index", "delta_recall", "support_recall", "fixed_recall"]
    _csv(output / "cases.csv", cases, fields)
    case_text = ["# 预定比较的事后案例", "",
                 "比较 tail30 的 support_mix 与 fixed_mix，按每个种子、每个内部用户的长尾 Recall@20 差值排序，整体最多取5个严格改善与5个严格下降案例。并列按种子和内部用户编号。", "",
                 f"实际选出改善 {n_improving}/5 个、下降 {n_worsening}/5 个；不足时保留不足，不把持平样本写成成功或失败。", "",
                 "这些极端案例用于解释保存的排名，不能代表总体因果效果，不用于重新调参。仅使用子集内部用户编号，未写原始用户字符串。", ""]
    for index, case in enumerate(cases, 1):
        case_text += [f"## {index}. {case['kind']}：种子 {case['seed']}，内部用户 {case['user_index']}", "",
                      f"长尾 Recall@20：固定混合 {case['fixed_recall']:.6f}，支持量混合 {case['support_recall']:.6f}；差值 {100*case['delta_recall']:.3f} 个百分点。", "",
                      f"长尾正样本 {case['tail_positives']} 个，其中缺图 {case['masked_tail_positives']} 个；缺图正样本的可见邻居数为 `{case['masked_positive_support']}`。", "",
                      f"长尾正样本内部商品 ID：`{case['tail_positive_items']}`。", "",
                      f"固定混合 Top20：`{case['fixed_top20']}`。", "",
                      f"支持量混合 Top20：`{case['support_top20']}`。", ""]
    (output / "cases.md").write_text("\n".join(case_text), encoding="utf-8")
    _figures(output, summary, supports, resources)
    artifacts = {p.name: file_hash(p) for p in output.iterdir() if p.is_file() and
                 p.name not in ["report_manifest.json", "report_manifest.previous.json"]}
    report = dict(schema_version=1, recomputed_runs=45, dataset_identity=dataset["manifest"]["identity"],
                  protocol_identity=protocol["identity"], summary=summary, comparisons=comparisons,
                  primary_comparison=primary, support_summary=support_summary,
                  cases={"improving": n_improving, "worsening": n_worsening},
                  input_files=input_files, artifacts=artifacts,
                  verification={"metric_absolute_tolerance": 1e-12, "metric_relative_tolerance": 1e-10,
                                "blocked_items_checked": True, "exact_test_users_checked": True,
                                "paired_masks_checked": True, "support_recomputed_from_train_neighbors": True,
                                "frozen_metadata_identity_checked": True, "features_reconstructed_and_hashed": True,
                                "per_user_json_and_csv_checked": True, "validation_rankings_recomputed": True,
                                "saved_npy_file_hashes_recorded": True, "sample_sd_ddof": 1})
    write_json(output / "report_manifest.json", report)
    return report
