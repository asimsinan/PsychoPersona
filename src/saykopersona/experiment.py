from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import build_audit_report, write_audit_outputs
from .config import ResearchConfig, load_config
from .data import build_stratified_three_way_split, load_raw_survey, score_survey_dataframe
from .evaluation import evaluate_simulation
from .lmstudio import LMStudioClient, PersonaCard
from .pipeline import (
    _hash_file,
    _run_stability_sweeps,
    _write_calibration_outputs,
    _write_data_outputs,
    _write_evaluation_outputs,
    _write_profile_outputs,
    _write_simulation_outputs,
)
from .progress import make_progress
from .coldstart import (
    fit_coldstart_predictor,
    predict_coldstart_membership,
    predict_coldstart_profiles,
)
from .profiles import build_profile_narratives, fit_empirical_profiles, predict_profiles
from .runs import create_fresh_run_dir
from .simulation import generate_persona_cards, simulate_survey_responses


CHECKPOINT_FILENAME = "experiment_checkpoint.json"
STEP_SEQUENCE = [
    "data",
    "audit",
    "profiles",
    "personas",
    "simulation",
    "evaluation",
    "stability",
]


@dataclass(slots=True)
class ExperimentArtifacts:
    run_dir: Path
    checkpoint: dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / CHECKPOINT_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checkpoint(run_dir: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = _now_iso()
    (run_dir / CHECKPOINT_FILENAME).write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _create_checkpoint(run_dir: Path, config: ResearchConfig) -> dict[str, Any]:
    checkpoint = {
        "run_dir": str(run_dir),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "completed_steps": [],
        "current_step": None,
        "failed_step": None,
        "model_id": None,
        "config": config.model_dump(mode="json"),
    }
    _write_checkpoint(run_dir, checkpoint)
    return checkpoint


def _resolve_run_dir(
    config: ResearchConfig,
    *,
    run_dir: Path | str | None,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    if run_dir is None:
        resolved_run_dir = create_fresh_run_dir(config.output_root, "experiment")
        return resolved_run_dir, _create_checkpoint(resolved_run_dir, config)

    resolved_run_dir = Path(run_dir)
    checkpoint = _load_checkpoint(resolved_run_dir)
    if checkpoint is None:
        if resume:
            raise FileNotFoundError(
                f"No checkpoint found in {resolved_run_dir}. Cannot resume experiment."
            )
        return resolved_run_dir, _create_checkpoint(resolved_run_dir, config)
    return resolved_run_dir, checkpoint


def _target_steps(step: str) -> list[str]:
    if step == "all":
        return STEP_SEQUENCE
    if step == "status":
        return []
    if step not in STEP_SEQUENCE:
        raise ValueError(f"Unsupported experiment step: {step}")
    target_index = STEP_SEQUENCE.index(step)
    return STEP_SEQUENCE[: target_index + 1]


def _read_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_persona_cards(path: Path) -> list[PersonaCard]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [PersonaCard.model_validate(item) for item in payload]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_manifest(
    run_dir: Path,
    config: ResearchConfig,
    checkpoint: dict[str, Any],
) -> None:
    stability_summary_path = run_dir / "stability" / "stability_summary.csv"
    stability_rows = (
        pd.read_csv(stability_summary_path).to_dict(orient="records")
        if stability_summary_path.exists()
        else []
    )
    manifest = {
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": _hash_file(config.dataset_path),
        "paper_path": str(config.paper_path) if config.paper_path is not None else None,
        "paper_sha256": _hash_file(config.paper_path),
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / CHECKPOINT_FILENAME),
        "llm_archive_dir": str(run_dir / "llm_archive"),
        "completed_steps": checkpoint["completed_steps"],
        "current_step": checkpoint.get("current_step"),
        "failed_step": checkpoint.get("failed_step"),
        "model_id": checkpoint.get("model_id"),
        "evaluation_split": config.simulation.evaluation_split,
        "stability_runs_requested": config.simulation.stability_runs,
        "stability_runs_completed": len(stability_rows),
        "stability_runs": stability_rows,
        "calibration_artifacts_path": str(run_dir / "calibration" / "calibration_artifacts.json")
        if config.calibration.enabled
        else None,
        "calibration_donor_pool_path": str(run_dir / "calibration" / "donor_pool.csv")
        if config.calibration.enabled
        else None,
        "config": config.model_dump(mode="json"),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_data_step(config: ResearchConfig, run_dir: Path) -> None:
    raw = load_raw_survey(config.dataset_path, dataset=config.dataset)
    scored = score_survey_dataframe(raw, dataset=config.dataset)
    train, validation, final_test = build_stratified_three_way_split(
        scored,
        validation_size=config.split.validation_size,
        final_test_size=config.split.final_test_size,
        random_state=config.split.random_state,
        dataset=config.dataset,
    )
    _write_data_outputs(
        run_dir,
        raw,
        scored,
        train,
        validation,
        final_test,
        config=config,
    )


def _run_audit_step(config: ResearchConfig, run_dir: Path) -> None:
    scored = _read_dataframe(run_dir / "data" / "scored_responses.csv")
    audit = build_audit_report(scored, dataset=config.dataset)
    write_audit_outputs(audit, run_dir / "audit")


def _run_profiles_step(config: ResearchConfig, run_dir: Path) -> None:
    train = _read_dataframe(run_dir / "data" / "train_split.csv")
    validation = _read_dataframe(run_dir / "data" / "validation_split.csv")
    final_test = _read_dataframe(run_dir / "data" / "final_test_split.csv")
    profile_result = fit_empirical_profiles(
        train,
        feature_columns=config.profiles.feature_columns,
        cluster_range=range(config.profiles.cluster_min, config.profiles.cluster_max + 1),
        random_state=config.split.random_state,
        anchor_columns=config.dataset.profile_anchor_columns,
        method=config.profiles.method,
    )
    validation_with_profiles = validation.copy()
    final_test_with_profiles = final_test.copy()
    # The oracle label is always computed, because the cold-start setting is only
    # interpretable against the assignment it replaces, and the agreement between the
    # two is what bounds a cold-start arm's profile recovery.
    validation_with_profiles["profile_id_oracle"] = predict_profiles(
        profile_result,
        validation_with_profiles,
    )
    final_test_with_profiles["profile_id_oracle"] = predict_profiles(
        profile_result,
        final_test_with_profiles,
    )
    coldstart_predictor = None
    if config.profiles.assignment == "coldstart":
        coldstart_predictor = fit_coldstart_predictor(
            profile_result.assignments,
            dataset=config.dataset,
            random_state=config.split.random_state,
        )
        for frame in (validation_with_profiles, final_test_with_profiles):
            frame["profile_id"] = predict_coldstart_profiles(coldstart_predictor, frame)
        if config.profiles.membership == "soft":
            # The hard label stays the classifier's prediction, so a soft run is scored in
            # the same label space as the hard cold-start run it is compared with.
            validation_with_profiles = validation_with_profiles.join(
                predict_coldstart_membership(
                    coldstart_predictor, validation_with_profiles, profile_result.assignments
                )
            )
            final_test_with_profiles = final_test_with_profiles.join(
                predict_coldstart_membership(
                    coldstart_predictor, final_test_with_profiles, profile_result.assignments
                )
            )
    else:
        for frame in (validation_with_profiles, final_test_with_profiles):
            frame["profile_id"] = frame["profile_id_oracle"]
    _write_profile_outputs(
        run_dir,
        profile_result.assignments,
        validation_with_profiles,
        final_test_with_profiles,
        profile_result,
        assignment=config.profiles.assignment,
        coldstart_predictor=coldstart_predictor,
        method=config.profiles.method,
        membership=config.profiles.membership,
    )
    _write_calibration_outputs(run_dir, config, profile_result.assignments)
    narratives = build_profile_narratives(profile_result, config.dataset)
    _write_json(run_dir / "profiles" / "profile_narratives.json", narratives)


def _evaluation_profile_assignment_path(config: ResearchConfig, run_dir: Path) -> Path:
    return (
        run_dir
        / "profiles"
        / f"{config.simulation.evaluation_split}_profile_assignments.csv"
    )


def _run_personas_step(
    config: ResearchConfig,
    run_dir: Path,
    checkpoint: dict[str, Any],
    *,
    transport,
) -> None:
    narratives = json.loads((run_dir / "profiles" / "profile_narratives.json").read_text(encoding="utf-8"))
    archive_dir = run_dir / "llm_archive"
    client = LMStudioClient(
        base_url=config.lmstudio.base_url,
        transport=transport,
        archive_dir=archive_dir,
        headers=config.lmstudio.build_headers(),
        default_model_id=config.lmstudio.model_id,
        service_tier=config.lmstudio.service_tier,
        reasoning=config.lmstudio.reasoning,
        provider=config.lmstudio.provider,
    )
    try:
        model = client.discover_loaded_model()
        persona_cards, metadata = generate_persona_cards(
            client,
            narratives,
            config.lmstudio.persona.to_runtime(),
            dataset=config.dataset,
        )
    finally:
        client.close()

    checkpoint["model_id"] = model.id
    simulation_dir = run_dir / "simulation"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    (simulation_dir / "persona_cards.json").write_text(
        json.dumps([card.model_dump() for card in persona_cards], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_json(
        simulation_dir / "persona_generation_metadata.json",
        [item.to_dict() for item in metadata],
    )


def _run_simulation_step(
    config: ResearchConfig,
    run_dir: Path,
    checkpoint: dict[str, Any],
    *,
    transport,
) -> None:
    evaluation_profiles = _read_dataframe(
        _evaluation_profile_assignment_path(config, run_dir)
    )
    profile_narratives = json.loads(
        (run_dir / "profiles" / "profile_narratives.json").read_text(encoding="utf-8")
    )
    persona_cards = _read_persona_cards(run_dir / "simulation" / "persona_cards.json")
    archive_dir = run_dir / "llm_archive"
    calibration_artifacts = None
    donor_pool = None
    if config.calibration.enabled:
        from .calibration import load_calibration_artifacts

        calibration_artifacts, donor_pool = load_calibration_artifacts(run_dir)
    client = LMStudioClient(
        base_url=config.lmstudio.base_url,
        transport=transport,
        archive_dir=archive_dir,
        headers=config.lmstudio.build_headers(),
        default_model_id=config.lmstudio.model_id,
        service_tier=config.lmstudio.service_tier,
        reasoning=config.lmstudio.reasoning,
        provider=config.lmstudio.provider,
    )
    try:
        model = client.discover_loaded_model()
        simulated, metadata = simulate_survey_responses(
            client,
            evaluation_profiles,
            persona_cards,
            config.lmstudio.primary.to_runtime(),
            max_rows=config.simulation.max_holdout_rows,
            prompt_version=config.simulation.prompt_version,
            parallel_workers=config.simulation.parallel_workers,
            self_consistency_samples=config.simulation.self_consistency_samples,
            response_mode=config.simulation.response_mode,
            item_scoring_mode=config.simulation.item_scoring_mode,
            trait_plan_samples=config.simulation.trait_plan_samples,
            ordinal_preset=config.lmstudio.ordinal.to_runtime(),
            profile_narratives=profile_narratives,
            calibration_artifacts=calibration_artifacts,
            donor_pool=donor_pool,
            calibration_strategy=config.calibration.strategy,
            random_state=config.split.random_state,
            save_raw_outputs=config.calibration.save_raw_outputs,
            dataset=config.dataset,
        )
    finally:
        client.close()

    checkpoint["model_id"] = model.id
    _write_simulation_outputs(run_dir, persona_cards, simulated)
    _write_json(
        run_dir / "simulation" / "simulation_generation_metadata.json",
        [item.to_dict() for item in metadata],
    )


def _run_evaluation_step(config: ResearchConfig, run_dir: Path) -> None:
    evaluation_profiles = _read_dataframe(
        _evaluation_profile_assignment_path(config, run_dir)
    )
    simulated = _read_dataframe(run_dir / "simulation" / "simulated_responses.csv")
    evaluation_human = evaluation_profiles[
        evaluation_profiles["respondent_id"].isin(simulated["respondent_id"])
    ].reset_index(drop=True)
    simulated = simulated.reset_index(drop=True)
    evaluation_report = evaluate_simulation(
        evaluation_human,
        simulated,
        scale_columns=config.dataset.scale_columns,
        item_columns=config.dataset.item_columns,
        profile_column="profile_id",
        regression_pairs=config.dataset.regression_pairs,
        subgroup_columns=config.dataset.subgroup_columns,
    )
    _write_evaluation_outputs(
        run_dir,
        evaluation_human,
        simulated,
        evaluation_report,
        config=config,
    )


def _run_stability_step(
    config: ResearchConfig,
    run_dir: Path,
    checkpoint: dict[str, Any],
    *,
    transport,
) -> None:
    evaluation_profiles = _read_dataframe(
        _evaluation_profile_assignment_path(config, run_dir)
    )
    profile_narratives = json.loads(
        (run_dir / "profiles" / "profile_narratives.json").read_text(encoding="utf-8")
    )
    persona_cards = _read_persona_cards(run_dir / "simulation" / "persona_cards.json")
    archive_dir = run_dir / "llm_archive"
    client = LMStudioClient(
        base_url=config.lmstudio.base_url,
        transport=transport,
        archive_dir=archive_dir,
        headers=config.lmstudio.build_headers(),
        default_model_id=config.lmstudio.model_id,
        service_tier=config.lmstudio.service_tier,
        reasoning=config.lmstudio.reasoning,
        provider=config.lmstudio.provider,
    )
    try:
        model = client.discover_loaded_model()
        _run_stability_sweeps(
            client,
            config,
            evaluation_profiles,
            persona_cards,
            profile_narratives,
            run_dir,
        )
    finally:
        client.close()

    checkpoint["model_id"] = model.id


def run_experiment(
    config: ResearchConfig,
    *,
    step: str = "all",
    run_dir: Path | str | None = None,
    resume: bool = False,
    transport=None,
) -> ExperimentArtifacts:
    if step == "status" and run_dir is None:
        raise ValueError("--run-dir is required when step=status")

    resolved_run_dir, checkpoint = _resolve_run_dir(config, run_dir=run_dir, resume=resume)
    if step == "status":
        return ExperimentArtifacts(run_dir=resolved_run_dir, checkpoint=checkpoint)

    target_steps = _target_steps(step)
    completed_target_steps = [
        completed_step
        for completed_step in checkpoint["completed_steps"]
        if completed_step in target_steps
    ]
    with make_progress(
        total=len(target_steps),
        desc="Experiment steps",
        unit="step",
        initial=len(completed_target_steps),
    ) as progress:
        for current_step in target_steps:
            if current_step in checkpoint["completed_steps"]:
                continue

            checkpoint["current_step"] = current_step
            checkpoint["failed_step"] = None
            _write_checkpoint(resolved_run_dir, checkpoint)

            try:
                if current_step == "data":
                    _run_data_step(config, resolved_run_dir)
                elif current_step == "audit":
                    _run_audit_step(config, resolved_run_dir)
                elif current_step == "profiles":
                    _run_profiles_step(config, resolved_run_dir)
                elif current_step == "personas":
                    _run_personas_step(
                        config,
                        resolved_run_dir,
                        checkpoint,
                        transport=transport,
                    )
                elif current_step == "simulation":
                    _run_simulation_step(
                        config,
                        resolved_run_dir,
                        checkpoint,
                        transport=transport,
                    )
                elif current_step == "evaluation":
                    _run_evaluation_step(config, resolved_run_dir)
                elif current_step == "stability":
                    _run_stability_step(
                        config,
                        resolved_run_dir,
                        checkpoint,
                        transport=transport,
                    )
            except Exception:
                checkpoint["failed_step"] = current_step
                _write_checkpoint(resolved_run_dir, checkpoint)
                raise

            checkpoint["completed_steps"].append(current_step)
            checkpoint["current_step"] = None
            checkpoint["failed_step"] = None
            _write_checkpoint(resolved_run_dir, checkpoint)
            _write_manifest(resolved_run_dir, config, checkpoint)
            progress.update(1)

    return ExperimentArtifacts(run_dir=resolved_run_dir, checkpoint=checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(prog="saykopersona-experiment")
    parser.add_argument("--config", default=None, help="Optional path to a YAML config file.")
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "status", *STEP_SEQUENCE],
        help="Run the full experiment or stop after a single named step.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Existing experiment run directory. Required to resume a previous run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint in --run-dir instead of starting fresh.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = run_experiment(
        config,
        step=args.step,
        run_dir=args.run_dir,
        resume=args.resume,
    )
    if args.step == "status":
        print(json.dumps(artifacts.checkpoint, indent=2, ensure_ascii=False))
        return

    print(f"Experiment state available at {artifacts.run_dir / CHECKPOINT_FILENAME}")
