from __future__ import annotations

import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.utils.io import load_yaml


def _resolve_policy_config(
    sweep_cfg: dict,
    key: str,
    policy: str,
) -> str | None:
    config_map = sweep_cfg.get(f"{key}s")
    if isinstance(config_map, dict):
        path = config_map.get(policy)
        if path:
            return str(path)

    path = sweep_cfg.get(key)
    if path is None:
        return None

    path = str(path)
    return path.replace("{policy}", policy)


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full routing-aware train/eval/causal sweep")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    policies = list(cfg["sweep"]["routing_policies"])

    for policy in policies:
        train_config = _resolve_policy_config(cfg["sweep"], "train_config", policy)
        eval_config_path = _resolve_policy_config(cfg["sweep"], "eval_config", policy)
        causal_config_path = _resolve_policy_config(cfg["sweep"], "causal_config", policy)

        if not train_config or not Path(train_config).exists():
            raise FileNotFoundError(
                f"Missing train config for policy={policy}: {train_config}"
            )
        if not eval_config_path or not Path(eval_config_path).exists():
            raise FileNotFoundError(
                f"Missing eval config for policy={policy}: {eval_config_path}"
            )
        if causal_config_path and not Path(causal_config_path).exists():
            raise FileNotFoundError(
                f"Missing causal config for policy={policy}: {causal_config_path}"
            )

        _run(
            [
                sys.executable,
                "scripts/train_baseline_ato.py" if policy == "same_token" else "scripts/train_ra_atos.py",
                "--config",
                train_config,
            ]
        )

        _run(
            [
                sys.executable,
                "scripts/eval_feature_space.py",
                "--config",
                eval_config_path,
            ]
        )

        if causal_config_path:
            _run(
                [
                    sys.executable,
                    "scripts/run_causal_restore.py",
                    "--config",
                    causal_config_path,
                ]
            )

    if cfg["sweep"].get("taxonomy_config"):
        _run([sys.executable, "scripts/build_transport_taxonomy.py", "--config", cfg["sweep"]["taxonomy_config"]])
    if cfg["sweep"].get("case_studies_config"):
        _run([sys.executable, "scripts/export_feature_case_studies.py", "--config", cfg["sweep"]["case_studies_config"]])


if __name__ == "__main__":
    main()
