from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config import default_config_path
from .data import get_project_root

VALIDATION_CONFIG_NAME = "paper_validation_baseline.yaml"
FINAL_TEST_CONFIG_NAME = "paper_final_test.yaml"


def _set_evaluation_split(base_config: dict[str, Any], evaluation_split: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    simulation = config.get("simulation")
    if simulation is None:
        simulation = {}
        config["simulation"] = simulation
    if not isinstance(simulation, dict):
        raise ValueError("The `simulation` section must be a mapping if provided.")
    simulation["evaluation_split"] = evaluation_split
    return config


def build_paper_run_configs(base_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(base_config, dict):
        raise TypeError("base_config must be a mapping.")

    return {
        VALIDATION_CONFIG_NAME: _set_evaluation_split(base_config, "validation"),
        FINAL_TEST_CONFIG_NAME: _set_evaluation_split(base_config, "final_test"),
    }


def _rebase_relative_path_fields(
    payload: dict[str, Any],
    *,
    source_dir: Path,
    target_dir: Path,
) -> dict[str, Any]:
    rebased = deepcopy(payload)
    for key in ["dataset_path", "paper_path", "output_root"]:
        value = rebased.get(key)
        if value in (None, ""):
            continue
        original_path = Path(value)
        if original_path.is_absolute():
            continue
        resolved_path = (source_dir / original_path).resolve()
        rebased[key] = os.path.relpath(resolved_path, start=target_dir.resolve())
    return rebased


def write_paper_run_configs(
    base_config_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    config_path = Path(base_config_path) if base_config_path is not None else default_config_path()
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else get_project_root() / "configs" / "runs"
    )

    base_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base_payload, dict):
        raise ValueError("Base config YAML must decode to a mapping.")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    config_variants = build_paper_run_configs(base_payload)

    written_paths: dict[str, Path] = {}
    for filename, payload in config_variants.items():
        target_path = resolved_output_dir / filename
        rebased_payload = _rebase_relative_path_fields(
            payload,
            source_dir=config_path.parent,
            target_dir=resolved_output_dir,
        )
        target_path.write_text(
            yaml.safe_dump(rebased_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written_paths[filename] = target_path

    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="create-paper-configs",
        description="Create reproducible validation/final-test config files for the paper workflow.",
    )
    parser.add_argument(
        "--base-config",
        default=str(default_config_path()),
        help="Base YAML config to clone. Defaults to configs/default.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(get_project_root() / "configs" / "runs"),
        help="Directory where paper config files will be written.",
    )
    args = parser.parse_args()

    written_paths = write_paper_run_configs(
        base_config_path=args.base_config,
        output_dir=args.output_dir,
    )
    for filename in [VALIDATION_CONFIG_NAME, FINAL_TEST_CONFIG_NAME]:
        print(f"Created {written_paths[filename]}")


if __name__ == "__main__":
    main()
