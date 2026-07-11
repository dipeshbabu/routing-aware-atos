from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from routing_aware_atos.provenance import (
    fingerprint_matches,
    live_causal_run_provenance,
    predictive_run_provenance,
    sha256_file,
    sha256_payload,
)
from routing_aware_atos.utils.io import load_json, load_yaml, save_json


def _expected_predictive_results(cfg: dict) -> list[tuple[dict, dict, Path]]:
    experiment_cfg = cfg["experiments"]
    root = Path(experiment_cfg.get("output_dir", "outputs/real")) / "runs"
    return [
        (
            pair,
            policy,
            root
            / f"L{int(pair['source_layer'])}_to_L{int(pair['target_layer'])}"
            / str(policy["name"])
            / "result.json",
        )
        for pair in experiment_cfg["layer_pairs"]
        for policy in experiment_cfg["policies"]
    ]


def _expected_causal_results(cfg: dict) -> list[tuple[dict, Path]]:
    root = Path(cfg["experiments"].get("output_dir", "outputs/real")) / "live_causal"
    return [
        (
            run,
            root
            / f"L{int(run['source_layer'])}_to_L{int(run['target_layer'])}"
            / f"{run['policy']}.json",
        )
        for run in cfg.get("live_causal", {}).get("runs", [])
    ]


def _validate_predictive_payload(
    cfg: dict,
    pair: dict,
    policy: dict,
    path: Path,
    payload: dict,
) -> None:
    expected_provenance = predictive_run_provenance(cfg, pair, policy)
    if not fingerprint_matches(payload, expected_provenance=expected_provenance):
        raise ValueError("run fingerprint or provenance does not match the current configuration")
    source = int(pair["source_layer"])
    target = int(pair["target_layer"])
    expected_identity = (
        source,
        target,
        int(pair.get("leap", target - source)),
        str(policy["name"]),
    )
    actual_identity = (
        int(payload.get("source_layer", -1)),
        int(payload.get("target_layer", -1)),
        int(payload.get("leap", -1)),
        str(payload.get("policy_name")),
    )
    if actual_identity != expected_identity:
        raise ValueError(f"result identity {actual_identity!r} does not match {expected_identity!r}")
    operator_path = path.with_name("operator.npz")
    if not operator_path.exists():
        raise ValueError(f"operator artifact is missing: {operator_path}")
    if payload.get("operator_sha256") != sha256_file(operator_path):
        raise ValueError("result was not produced by the current operator artifact")
    float(payload["selected_ridge_lambda"])
    float(payload["residual_metrics"]["r2"])
    float(payload["feature_summary"]["mean_r2"])
    float(payload["feature_summary"]["median_r2"])
    int(payload["feature_summary"]["num_features"])


def _validate_causal_payload(
    cfg: dict,
    run: dict,
    payload: dict,
) -> None:
    source = int(run["source_layer"])
    target = int(run["target_layer"])
    policy = str(run["policy"])
    operator_path = (
        Path(cfg["experiments"].get("output_dir", "outputs/real"))
        / "runs"
        / f"L{source}_to_L{target}"
        / policy
        / "operator.npz"
    )
    expected_provenance = live_causal_run_provenance(
        cfg,
        operator_path=operator_path,
        source_layer=source,
        target_layer=target,
        policy=policy,
    )
    if not fingerprint_matches(payload, expected_provenance=expected_provenance):
        raise ValueError("run fingerprint or provenance does not match the current configuration")
    actual_identity = (
        int(payload.get("source_layer", -1)),
        int(payload.get("target_layer", -1)),
        str(payload.get("routing_policy")),
    )
    if actual_identity != (source, target, policy):
        raise ValueError("causal result has the wrong run identity")
    expected_modes = {
        str(value)
        for value in cfg.get("live_causal", {}).get(
            "position_counts",
            [1, 5, "all"],
        )
    }
    if set(payload.get("results", {})) != expected_modes:
        raise ValueError("causal result does not contain the configured position modes")
    for mode in expected_modes:
        summary = payload["results"][mode]["summary"]
        for key in (
            "clean_cross_entropy",
            "ablated_cross_entropy",
            "restored_cross_entropy",
            "kl_restoration",
            "logit_mse_restoration",
        ):
            float(summary[key])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_predictive(rows: list[dict[str, Any]], path: Path) -> None:
    targets = sorted({int(row["target_layer"]) for row in rows})
    figure, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5), squeeze=False)
    for axis, target in zip(axes[0], targets):
        target_rows = [row for row in rows if int(row["target_layer"]) == target]
        for policy in sorted({str(row["policy_name"]) for row in target_rows}):
            policy_rows = sorted(
                (row for row in target_rows if row["policy_name"] == policy),
                key=lambda row: int(row["leap"]),
            )
            axis.plot(
                [row["leap"] for row in policy_rows],
                [row["feature_mean_r2"] for row in policy_rows],
                marker="o",
                linewidth=1.5,
                label=policy,
            )
        axis.set_title(f"Target layer {target}")
        axis.set_xlabel("Layer leap k")
        axis.set_ylabel("Mean feature R2")
        axis.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_causal(rows: list[dict[str, Any]], path: Path) -> None:
    modes = [mode for mode in ("1", "5", "all") if any(row["position_mode"] == mode for row in rows)]
    figure, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.5), squeeze=False)
    for axis, mode in zip(axes[0], modes):
        mode_rows = [row for row in rows if row["position_mode"] == mode]
        for policy in sorted({str(row["policy_name"]) for row in mode_rows}):
            policy_rows = sorted(
                (row for row in mode_rows if row["policy_name"] == policy),
                key=lambda row: int(row["leap"]),
            )
            axis.plot(
                [row["leap"] for row in policy_rows],
                [row["restored_log_perplexity"] for row in policy_rows],
                marker="o",
                linewidth=1.5,
                label=policy,
            )
        axis.set_title(f"Modified positions: {mode}")
        axis.set_xlabel("Layer leap k")
        axis.set_ylabel("Restored log perplexity")
        axis.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_efficiency(payloads: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for payload in payloads:
        efficiency = payload.get("transport_efficiency")
        if not efficiency:
            continue
        rows = efficiency["ranks"]
        axis.plot(
            [row["rank"] for row in rows],
            [row["efficiency"] for row in rows],
            linewidth=1.5,
            label=(
                f"{payload['policy_name']} target={payload['target_layer']} "
                f"k={payload['leap']}"
            ),
        )
    axis.set_xlabel("Operator rank")
    axis.set_ylabel("Transport efficiency")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_real_report(cfg: dict, *, allow_partial: bool = False) -> dict[str, Any]:
    report_dir = Path(cfg.get("report", {}).get("output_dir", "outputs/real/report"))
    report_dir.mkdir(parents=True, exist_ok=True)

    predictive_payloads: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    for pair, policy, path in _expected_predictive_results(cfg):
        if path.exists():
            try:
                payload = load_json(path)
                if not isinstance(payload, dict):
                    raise ValueError("result must contain a JSON object")
                _validate_predictive_payload(cfg, pair, policy, path, payload)
                predictive_payloads.append(payload)
            except Exception as exc:
                invalid.append({"path": str(path), "reason": str(exc)})
        else:
            missing.append(str(path))

    causal_payloads: list[dict[str, Any]] = []
    for run, path in _expected_causal_results(cfg):
        if path.exists():
            try:
                payload = load_json(path)
                if not isinstance(payload, dict):
                    raise ValueError("result must contain a JSON object")
                _validate_causal_payload(cfg, run, payload)
                causal_payloads.append(payload)
            except Exception as exc:
                invalid.append({"path": str(path), "reason": str(exc)})
        else:
            missing.append(str(path))
    if (missing or invalid) and not allow_partial:
        first_problem = missing[0] if missing else invalid[0]["path"]
        raise ValueError(
            f"Paper-scale report has {len(missing)} missing and {len(invalid)} invalid result "
            f"files; first problem: {first_problem}"
        )
    if not predictive_payloads:
        raise ValueError("No predictive real-run results were found")

    predictive_rows = [
        {
            "source_layer": int(payload["source_layer"]),
            "target_layer": int(payload["target_layer"]),
            "leap": int(payload["leap"]),
            "policy_name": str(payload["policy_name"]),
            "ridge_lambda": float(payload["selected_ridge_lambda"]),
            "residual_r2": float(payload["residual_metrics"]["r2"]),
            "feature_mean_r2": float(payload["feature_summary"]["mean_r2"]),
            "feature_median_r2": float(payload["feature_summary"]["median_r2"]),
            "num_features": int(payload["feature_summary"]["num_features"]),
        }
        for payload in predictive_payloads
    ]
    causal_rows: list[dict[str, Any]] = []
    for payload in causal_payloads:
        for mode, mode_payload in payload["results"].items():
            summary = mode_payload["summary"]
            causal_rows.append(
                {
                    "source_layer": int(payload["source_layer"]),
                    "target_layer": int(payload["target_layer"]),
                    "leap": int(payload["target_layer"] - payload["source_layer"]),
                    "policy_name": str(payload["routing_policy"]),
                    "position_mode": str(mode),
                    "clean_log_perplexity": float(summary["clean_cross_entropy"]),
                    "ablated_log_perplexity": float(summary["ablated_cross_entropy"]),
                    "restored_log_perplexity": float(summary["restored_cross_entropy"]),
                    "null_log_perplexity": (
                        None
                        if summary.get("null_cross_entropy") is None
                        else float(summary["null_cross_entropy"])
                    ),
                    "kl_restoration": float(summary["kl_restoration"]),
                    "logit_mse_restoration": float(summary["logit_mse_restoration"]),
                }
            )

    predictive_csv = report_dir / "predictive_results.csv"
    causal_csv = report_dir / "causal_results.csv"
    predictive_plot = report_dir / "feature_r2_by_leap.png"
    efficiency_plot = report_dir / "transport_efficiency.png"
    _write_csv(predictive_csv, predictive_rows)
    _plot_predictive(predictive_rows, predictive_plot)
    if any(payload.get("transport_efficiency") for payload in predictive_payloads):
        _plot_efficiency(predictive_payloads, efficiency_plot)
    if causal_rows:
        _write_csv(causal_csv, causal_rows)
        _plot_causal(causal_rows, report_dir / "causal_log_perplexity_by_leap.png")

    report = {
        "protocol": cfg.get("protocol"),
        "config_sha256": sha256_payload(cfg),
        "complete": not missing and not invalid,
        "missing_results": missing,
        "invalid_results": invalid,
        "predictive_run_count": len(predictive_payloads),
        "causal_run_count": len(causal_payloads),
        "predictive_rows": predictive_rows,
        "causal_rows": causal_rows,
        "source_result_sha256": {
            str(path): sha256_file(path)
            for _, _, path in _expected_predictive_results(cfg)
            if path.exists() and str(path) not in {item["path"] for item in invalid}
        }
        | {
            str(path): sha256_file(path)
            for _, path in _expected_causal_results(cfg)
            if path.exists() and str(path) not in {item["path"] for item in invalid}
        },
        "artifacts": {
            "predictive_csv": str(predictive_csv),
            "causal_csv": str(causal_csv) if causal_rows else None,
            "predictive_plot": str(predictive_plot),
            "efficiency_plot": (
                str(efficiency_plot)
                if any(payload.get("transport_efficiency") for payload in predictive_payloads)
                else None
            ),
        },
    }
    save_json(report_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tables and plots from real experiment outputs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = build_real_report(load_yaml(args.config), allow_partial=args.allow_partial)
    print(
        f"Saved real report with {report['predictive_run_count']} predictive and "
        f"{report['causal_run_count']} causal runs"
    )


if __name__ == "__main__":
    main()
