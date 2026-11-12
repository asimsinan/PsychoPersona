from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .audit import AuditReport, build_audit_report, write_audit_outputs
from .calibration import (
    fit_calibration_artifacts,
    load_calibration_artifacts,
    write_calibration_artifacts,
)
from .config import ResearchConfig
from .data import (
    DEFAULT_DATASET,
    DIGITAL_SCORE_COLUMNS,
    SOCIAL_SCORE_COLUMNS,
    WELLBEING_SCORE_COLUMNS,
    build_stratified_three_way_split,
    load_raw_survey,
    score_survey_dataframe,
)
from .evaluation import EvaluationReport, evaluate_simulation
from .lmstudio import LMStudioClient, PersonaCard
from .coldstart import (
    fit_coldstart_predictor,
    predict_coldstart_membership,
    predict_coldstart_profiles,
)
from .profiles import build_profile_narratives, fit_empirical_profiles, predict_profiles
from .progress import make_progress
from .runs import create_fresh_run_dir
from .simulation import generate_persona_cards, simulate_survey_responses


@dataclass(slots=True)
class PipelineArtifacts:
    run_dir: Path
    manifest: dict[str, object]


SCALE_COLUMNS = [
    *DEFAULT_DATASET.scale_columns,
]
ITEM_COLUMNS = DIGITAL_SCORE_COLUMNS + SOCIAL_SCORE_COLUMNS + WELLBEING_SCORE_COLUMNS


def _hash_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_profile_outputs(
    run_dir: Path,
    train_profiles: pd.DataFrame,
    validation_profiles: pd.DataFrame,
    final_test_profiles: pd.DataFrame,
    result,
    assignment: str = "oracle",
    coldstart_predictor=None,
    method: str = "gmm",
    membership: str = "hard",
) -> None:
    profile_dir = run_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    train_profiles.to_csv(profile_dir / "train_profile_assignments.csv", index=False)
    validation_profiles.to_csv(
        profile_dir / "validation_profile_assignments.csv",
        index=False,
    )
    final_test_profiles.to_csv(
        profile_dir / "final_test_profile_assignments.csv",
        index=False,
    )
    result.summary.to_csv(profile_dir / "profile_summary.csv", index=False)
    result.model_scores.to_csv(profile_dir / "profile_model_scores.csv", index=False)

    # Record which assignment the run used, and how far the cold-start labels fall from
    # the oracle ones. Without this an archived run cannot be placed in either setting
    # after the fact, and the oracle-to-cold-start gap is the headline quantity.
    summary: dict[str, object] = {
        "assignment": assignment,
        "method": method,
        "membership": membership,
    }
    for name, frame in (("validation", validation_profiles), ("final_test", final_test_profiles)):
        if "profile_id_oracle" not in frame.columns:
            continue
        agreement = float((frame["profile_id"] == frame["profile_id_oracle"]).mean())
        summary[f"{name}_n"] = int(len(frame))
        summary[f"{name}_oracle_agreement"] = round(agreement, 4)
        summary[f"{name}_distinct_assigned"] = int(frame["profile_id"].nunique())
    if coldstart_predictor is not None:
        summary["coldstart_selected"] = coldstart_predictor.selected
        summary["coldstart_cv_scores"] = coldstart_predictor.cv_scores
        summary["coldstart_cv_standard_errors"] = coldstart_predictor.cv_se
        summary["coldstart_multi_select_expanded"] = coldstart_predictor.multi_select
        summary["coldstart_features"] = coldstart_predictor.feature_columns
        summary["coldstart_train_accuracy"] = round(coldstart_predictor.train_accuracy, 4)
        summary["coldstart_n_train"] = coldstart_predictor.n_train
    (profile_dir / "assignment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_calibration_outputs(
    run_dir: Path,
    config: ResearchConfig,
    train_profiles: pd.DataFrame,
) -> None:
    if not config.calibration.enabled:
        return
    artifacts, donor_pool = fit_calibration_artifacts(
        train_profiles,
        scale_columns=config.dataset.scale_columns,
        item_columns=config.dataset.item_columns,
        scale_to_items=config.dataset.scale_to_items,
        scale_shift_aliases={
            scale.mean_column: scale.effective_shift_key for scale in config.dataset.scales
        },
        profile_column="profile_id",
        subgroup_columns=config.dataset.subgroup_columns,
        donor_pool_min=config.calibration.donor_pool_min,
        profile_shrinkage=config.calibration.profile_shrinkage,
        signal_scale=config.calibration.signal_scale,
        neighbor_count=config.calibration.neighbor_count,
        response_min=config.dataset.response_min,
        response_max=config.dataset.response_max,
    )
    write_calibration_artifacts(run_dir, artifacts, donor_pool)


def _normalized_distribution(series: pd.Series) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in series.value_counts(normalize=True, dropna=False).sort_index().items()
    }


def _write_data_outputs(
    run_dir: Path,
    raw: pd.DataFrame,
    scored: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    final_test: pd.DataFrame,
    *,
    config: ResearchConfig,
) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(data_dir / "scored_responses.csv", index=False)
    train.to_csv(data_dir / "train_split.csv", index=False)
    validation.to_csv(data_dir / "validation_split.csv", index=False)
    final_test.to_csv(data_dir / "final_test_split.csv", index=False)

    data_quality = {
        "raw_rows": int(len(raw)),
        "usable_rows": int(len(scored)),
        "excluded_rows": int(len(raw) - len(scored)),
        "raw_missing_cells": int(raw.isna().sum().sum()),
        "scored_missing_cells": int(scored.isna().sum().sum()),
        "raw_missing_by_column": {
            column: int(value)
            for column, value in raw.isna().sum().sort_values(ascending=False).items()
        },
        "scored_missing_by_column": {
            column: int(value)
            for column, value in scored.isna().sum().sort_values(ascending=False).items()
        },
    }
    (data_dir / "data_quality.json").write_text(
        json.dumps(data_quality, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    split_summary = {
        "train_n": int(len(train)),
        "validation_n": int(len(validation)),
        "final_test_n": int(len(final_test)),
        "categorical_balance": {
            column: {
                "train": _normalized_distribution(train[column]),
                "validation": _normalized_distribution(validation[column]),
                "final_test": _normalized_distribution(final_test[column]),
            }
            for column in config.dataset.subgroup_columns
            if column in train.columns and column in validation.columns and column in final_test.columns
        },
        "scale_means": {
            column: {
                "train": float(train[column].mean()),
                "validation": float(validation[column].mean()),
                "final_test": float(final_test[column].mean()),
            }
            for column in config.dataset.scale_columns
        },
    }
    (data_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_simulation_outputs(
    run_dir: Path,
    persona_cards: list[PersonaCard],
    simulated: pd.DataFrame,
) -> None:
    simulation_dir = run_dir / "simulation"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    (simulation_dir / "persona_cards.json").write_text(
        json.dumps([card.model_dump() for card in persona_cards], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    simulated.to_csv(simulation_dir / "simulated_responses.csv", index=False)
    calibration_signals = simulated.attrs.get("calibration_signals")
    if calibration_signals:
        (simulation_dir / "calibration_signals.json").write_text(
            json.dumps(calibration_signals, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _write_evaluation_outputs(
    run_dir: Path,
    human: pd.DataFrame,
    simulated: pd.DataFrame,
    report: EvaluationReport,
    *,
    config: ResearchConfig,
) -> None:
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    scale_columns = config.dataset.scale_columns

    scale_means = pd.DataFrame(
        {
            "human": human[scale_columns].mean(),
            "simulated": simulated[scale_columns].mean(),
        }
    )
    scale_means["absolute_delta"] = (scale_means["human"] - scale_means["simulated"]).abs()
    scale_means.to_csv(evaluation_dir / "scale_mean_comparison.csv", index_label="scale")

    scale_stds = pd.DataFrame(
        {
            "human": human[scale_columns].std(ddof=1),
            "simulated": simulated[scale_columns].std(ddof=1),
        }
    )
    scale_stds["absolute_delta"] = (scale_stds["human"] - scale_stds["simulated"]).abs()
    scale_stds.to_csv(evaluation_dir / "scale_std_comparison.csv", index_label="scale")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    scale_means.plot(kind="bar", ax=axis)
    axis.set_title("Human vs simulated scale means")
    axis.set_ylabel("Mean score")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(evaluation_dir / "scale_mean_comparison.png", dpi=200)
    plt.close(figure)

    human_corr = human[scale_columns].corr()
    simulated_corr = simulated[scale_columns].corr()
    human_corr.to_csv(evaluation_dir / "human_correlations.csv")
    simulated_corr.to_csv(evaluation_dir / "simulated_correlations.csv")

    item_mean_deltas = pd.DataFrame(
        [
            {"item_id": item_id, "absolute_delta": delta}
            for item_id, delta in report.item_mean_deltas.items()
        ]
    ).sort_values("item_id")
    item_mean_deltas.to_csv(evaluation_dir / "item_mean_deltas.csv", index=False)

    nonlinear_comparison = pd.DataFrame(
        [
            {"metric": metric, "absolute_delta": delta}
            for metric, delta in report.nonlinear_deltas.items()
        ]
    ).sort_values("metric")
    nonlinear_comparison.to_csv(
        evaluation_dir / "nonlinear_regression_comparison.csv",
        index=False,
    )
    standardized_regression = pd.DataFrame(
        [
            {"metric": metric, "absolute_delta": delta}
            for metric, delta in report.standardized_regression_deltas.items()
        ]
    ).sort_values("metric")
    standardized_regression.to_csv(
        evaluation_dir / "standardized_regression_comparison.csv",
        index=False,
    )

    (evaluation_dir / "response_diversity.json").write_text(
        json.dumps(report.response_pattern_diversity, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"subgroup": subgroup, "rmse": value}
            for subgroup, value in report.subgroup_rmse.items()
        ]
    ).to_csv(evaluation_dir / "subgroup_rmse.csv", index=False)

    heatmap_figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(human_corr, annot=True, cmap="Blues", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Human correlations")
    sns.heatmap(simulated_corr, annot=True, cmap="Oranges", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Simulated correlations")
    heatmap_figure.tight_layout()
    heatmap_figure.savefig(evaluation_dir / "correlation_heatmaps.png", dpi=200)
    plt.close(heatmap_figure)

    calibration_base_columns = [
        f"base_{column}" for column in scale_columns if f"base_{column}" in simulated.columns
    ]
    calibration_target_columns = [
        f"target_{column}" for column in scale_columns if f"target_{column}" in simulated.columns
    ]
    if calibration_base_columns and calibration_target_columns:
        movement_deltas = []
        target_gaps = []
        for scale_name in scale_columns:
            base_column = f"base_{scale_name}"
            target_column = f"target_{scale_name}"
            if base_column not in simulated.columns or target_column not in simulated.columns:
                continue
            movement_deltas.extend(
                (simulated[target_column] - simulated[base_column]).abs().tolist()
            )
            target_gaps.extend(
                (simulated[target_column] - simulated[scale_name]).abs().tolist()
            )
        calibration_diagnostics = {
            "strategy": simulated.attrs.get("calibration_strategy")
            or (
                simulated["calibration_strategy"].iloc[0]
                if "calibration_strategy" in simulated.columns and not simulated.empty
                else None
            ),
            "within_profile_scale_rmse": report.within_profile_scale_rmse,
            "mean_absolute_target_movement": float(np.mean(movement_deltas))
            if movement_deltas
            else float("nan"),
            "mean_absolute_target_gap": float(np.mean(target_gaps))
            if target_gaps
            else float("nan"),
        }
        (evaluation_dir / "calibration_diagnostics.json").write_text(
            json.dumps(calibration_diagnostics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (evaluation_dir / "evaluation_report.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_stability_sweeps(
    client: LMStudioClient,
    config: ResearchConfig,
    evaluation_profiles: pd.DataFrame,
    persona_cards: list[PersonaCard],
    profile_narratives: list[dict[str, object]] | None,
    run_dir: Path,
) -> list[dict[str, object]]:
    if config.simulation.stability_runs <= 0:
        return []

    stability_dir = run_dir / "stability"
    stability_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    max_rows = config.simulation.max_holdout_rows or min(25, len(evaluation_profiles))
    evaluation_human = evaluation_profiles.head(max_rows)
    calibration_bundle = (
        load_calibration_artifacts(run_dir) if config.calibration.enabled else (None, None)
    )
    with make_progress(
        total=config.simulation.stability_runs,
        desc="Running stability sweeps",
        unit="run",
        leave=False,
    ) as progress:
        for run_index in range(1, config.simulation.stability_runs + 1):
            simulated, _ = simulate_survey_responses(
                client,
                evaluation_human,
                persona_cards,
                config.lmstudio.stability.to_runtime(),
                max_rows=max_rows,
                run_label=f"stability_{run_index}",
                prompt_version=config.simulation.prompt_version,
                parallel_workers=config.simulation.parallel_workers,
                self_consistency_samples=config.simulation.self_consistency_samples,
                response_mode=config.simulation.response_mode,
                item_scoring_mode=config.simulation.item_scoring_mode,
                trait_plan_samples=config.simulation.trait_plan_samples,
                ordinal_preset=config.lmstudio.ordinal.to_runtime(),
                profile_narratives=profile_narratives,
                calibration_artifacts=calibration_bundle[0],
                donor_pool=calibration_bundle[1],
                calibration_strategy=config.calibration.strategy,
                random_state=config.split.random_state,
                save_raw_outputs=config.calibration.save_raw_outputs,
                dataset=config.dataset,
            )
            report = evaluate_simulation(
                evaluation_human,
                simulated,
                scale_columns=config.dataset.scale_columns,
                item_columns=config.dataset.item_columns,
                profile_column="profile_id",
                regression_pairs=config.dataset.regression_pairs,
                subgroup_columns=config.dataset.subgroup_columns,
            )
            simulated.to_csv(stability_dir / f"simulated_responses_run_{run_index}.csv", index=False)
            rows.append(
                {
                    "run_label": f"stability_{run_index}",
                    "correlation_rmse": report.correlation_rmse,
                    "profile_recovery_accuracy": report.profile_recovery_accuracy,
                    **{
                        f"{key}_delta": value for key, value in report.scale_mean_deltas.items()
                    },
                }
            )
            progress.update(1)
    if rows:
        pd.DataFrame(rows).to_csv(stability_dir / "stability_summary.csv", index=False)
    return rows


def _select_evaluation_split(
    config: ResearchConfig,
    validation_profiles: pd.DataFrame,
    final_test_profiles: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    if config.simulation.evaluation_split == "validation":
        return "validation", validation_profiles
    return "final_test", final_test_profiles


def run_full_pipeline(
    config: ResearchConfig,
    *,
    transport=None,
) -> PipelineArtifacts:
    run_dir = create_fresh_run_dir(config.output_root, "run-all")
    pipeline_step_total = 6 + (1 if config.simulation.stability_runs > 0 else 0)

    with make_progress(
        total=pipeline_step_total,
        desc="Pipeline steps",
        unit="step",
    ) as progress:
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
        progress.update(1)

        audit = build_audit_report(scored, dataset=config.dataset)
        write_audit_outputs(audit, run_dir / "audit")
        progress.update(1)

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
        # interpretable against the assignment it replaces, and the agreement between
        # the two is what bounds a cold-start arm's profile recovery.
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
        progress.update(1)

        llm_archive_dir = run_dir / "llm_archive"
        client = LMStudioClient(
            base_url=config.lmstudio.base_url,
            transport=transport,
            archive_dir=llm_archive_dir,
            headers=config.lmstudio.build_headers(),
            default_model_id=config.lmstudio.model_id,
            service_tier=config.lmstudio.service_tier,
            reasoning=config.lmstudio.reasoning,
            provider=config.lmstudio.provider,
        )
        try:
            loaded_model = client.discover_loaded_model()
            narratives = build_profile_narratives(profile_result, config.dataset)
            persona_cards, persona_metadata = generate_persona_cards(
                client,
                narratives,
                config.lmstudio.persona.to_runtime(),
                dataset=config.dataset,
            )
            progress.update(1)

            evaluation_split, evaluation_profiles = _select_evaluation_split(
                config,
                validation_with_profiles,
                final_test_with_profiles,
            )
            calibration_artifacts = None
            donor_pool = None
            if config.calibration.enabled:
                calibration_artifacts, donor_pool = load_calibration_artifacts(run_dir)
            simulated, response_metadata = simulate_survey_responses(
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
                profile_narratives=narratives,
                calibration_artifacts=calibration_artifacts,
                donor_pool=donor_pool,
                calibration_strategy=config.calibration.strategy,
                random_state=config.split.random_state,
                save_raw_outputs=config.calibration.save_raw_outputs,
                dataset=config.dataset,
            )
            _write_simulation_outputs(run_dir, persona_cards, simulated)
            progress.update(1)

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
            progress.update(1)
            stability_rows = _run_stability_sweeps(
                client,
                config,
                evaluation_profiles,
                persona_cards,
                narratives,
                run_dir,
            )
            if config.simulation.stability_runs > 0:
                progress.update(1)
        finally:
            client.close()

    manifest = {
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": _hash_file(config.dataset_path),
        "paper_path": str(config.paper_path) if config.paper_path is not None else None,
        "paper_sha256": _hash_file(config.paper_path),
        "run_dir": str(run_dir),
        "llm_archive_dir": str(llm_archive_dir),
        "model_id": loaded_model.id,
        "prompt_version": config.simulation.prompt_version,
        "evaluation_split": evaluation_split,
        "config": config.model_dump(mode="json"),
        "calibration_artifacts_path": str(run_dir / "calibration" / "calibration_artifacts.json")
        if config.calibration.enabled
        else None,
        "calibration_donor_pool_path": str(run_dir / "calibration" / "donor_pool.csv")
        if config.calibration.enabled
        else None,
        "persona_generations": [metadata.to_dict() for metadata in persona_metadata],
        "simulation_generations": [metadata.to_dict() for metadata in response_metadata],
        "stability_runs": stability_rows,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return PipelineArtifacts(run_dir=run_dir, manifest=manifest)
