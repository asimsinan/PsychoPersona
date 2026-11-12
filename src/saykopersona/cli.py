from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import build_audit_report, write_audit_outputs
from .config import ResearchConfig, load_config
from .data import load_raw_survey, score_survey_dataframe
from .lmstudio import InferencePreset, LMStudioClient
from .pipeline import run_full_pipeline
from .runs import create_fresh_run_dir
from .simulation import build_survey_response_schema


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Optional path to a YAML config file.")


def run_audit_command(config: ResearchConfig) -> Path:
    run_dir = create_fresh_run_dir(config.output_root, "audit")
    raw = load_raw_survey(config.dataset_path, dataset=config.dataset)
    scored = score_survey_dataframe(raw, dataset=config.dataset)
    report = build_audit_report(scored, dataset=config.dataset)
    output_dir = run_dir / "audit"
    write_audit_outputs(report, output_dir)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "command": "audit",
                "run_dir": str(run_dir),
                "dataset_path": str(config.dataset_path),
                "paper_path": str(config.paper_path) if config.paper_path is not None else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def run_lmstudio_smoke_command(config: ResearchConfig) -> Path:
    run_dir = create_fresh_run_dir(config.output_root, "lmstudio-smoke")
    archive_dir = run_dir / "llm_archive"
    client = LMStudioClient(
        base_url=config.lmstudio.base_url,
        archive_dir=archive_dir,
        headers=config.lmstudio.build_headers(),
        default_model_id=config.lmstudio.model_id,
        service_tier=config.lmstudio.service_tier,
        reasoning=config.lmstudio.reasoning,
        provider=config.lmstudio.provider,
    )
    try:
        model = client.discover_loaded_model()
        smoke_schema = build_survey_response_schema(
            ("smoke_item_01",),
            response_min=config.dataset.response_min,
            response_max=config.dataset.response_max,
        )
        parsed, metadata = client.generate_structured(
            system_prompt="Return the requested schema only.",
            user_prompt="Return a minimal survey response payload with one item.",
            schema=smoke_schema,
            preset=InferencePreset(name="smoke", temperature=0.0, max_tokens=128),
        )
    finally:
        client.close()

    result_payload = {
        "command": "lmstudio-smoke",
        "run_dir": str(run_dir),
        "llm_archive_dir": str(archive_dir),
        "model_id": model.id,
        "metadata": metadata.to_dict(),
        "parsed": parsed.model_dump(mode="json"),
    }
    (run_dir / "smoke_result.json").write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result_payload, indent=2, ensure_ascii=False))
    return run_dir


def _run_audit_command(config_path: str | None) -> None:
    config = load_config(config_path)
    run_dir = run_audit_command(config)
    print(f"Wrote audit outputs to {run_dir}")


def _run_lmstudio_smoke(config_path: str | None) -> None:
    config = load_config(config_path)
    run_lmstudio_smoke_command(config)


def main() -> None:
    parser = argparse.ArgumentParser(prog="saykopersona")
    _add_config_argument(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)
    run_all_parser = subparsers.add_parser("run-all", help="Run the full standalone study pipeline.")
    audit_parser = subparsers.add_parser("audit", help="Run only the baseline reproducibility audit.")
    smoke_parser = subparsers.add_parser(
        "lmstudio-smoke",
        help="Verify model discovery and structured output.",
    )
    for subparser in [run_all_parser, audit_parser, smoke_parser]:
        _add_config_argument(subparser)

    args = parser.parse_args()

    if args.command == "audit":
        _run_audit_command(args.config)
        return
    if args.command == "lmstudio-smoke":
        _run_lmstudio_smoke(args.config)
        return

    config = load_config(args.config)
    artifacts = run_full_pipeline(config)
    print(f"Pipeline completed. Artifacts written to {artifacts.run_dir}")
