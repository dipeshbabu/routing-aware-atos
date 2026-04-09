from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


def summarize_ablation_sweep(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    metric_keys: Sequence[str],
) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for group, items in sorted(grouped.items()):
        out: Dict[str, Any] = {group_key: group, "count": len(items)}
        for metric in metric_keys:
            vals = [float(item[metric]) for item in items if metric in item]
            if vals:
                out[f"{metric}_mean"] = float(sum(vals) / len(vals))
                out[f"{metric}_min"] = float(min(vals))
                out[f"{metric}_max"] = float(max(vals))
        summary_rows.append(out)
    return {"group_key": group_key, "summary_rows": summary_rows}


def compare_against_baseline(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_policy: str,
    policy_key: str = "policy_name",
    metric_keys: Sequence[str] = ("mean_r2",),
) -> Dict[str, Any]:
    baseline_rows = [row for row in rows if str(row.get(policy_key)) == baseline_policy]
    if not baseline_rows:
        raise ValueError(f"No baseline rows found for {baseline_policy}")
    baseline = baseline_rows[0]

    deltas: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {policy_key: row.get(policy_key), "rank": row.get("rank")}
        for metric in metric_keys:
            if metric in row and metric in baseline:
                item[f"{metric}_delta"] = float(row[metric]) - float(baseline[metric])
        deltas.append(item)
    return {"baseline_policy": baseline_policy, "delta_rows": deltas}
