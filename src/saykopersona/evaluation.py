from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import DEFAULT_DATASET
from .dataset_spec import RegressionPairSettings


@dataclass(slots=True)
class EvaluationReport:
    scale_mean_deltas: dict[str, float]
    scale_std_deltas: dict[str, float]
    correlation_rmse: float
    regression_deltas: dict[str, float]
    standardized_regression_deltas: dict[str, float]
    profile_recovery_accuracy: float
    within_profile_scale_rmse: float
    item_mean_deltas: dict[str, float] = field(default_factory=dict)
    nonlinear_deltas: dict[str, float] = field(default_factory=dict)
    response_pattern_diversity: dict[str, float] = field(default_factory=dict)
    subgroup_rmse: dict[str, float] = field(default_factory=dict)
    individual_level_metrics: dict[str, float] = field(default_factory=dict)


def _safe_abs_delta(left: float, right: float) -> float:
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    return float(abs(left - right))


def _correlation_rmse(
    human: pd.DataFrame,
    simulated: pd.DataFrame,
    scale_columns: list[str],
) -> float:
    human_corr = human.loc[:, scale_columns].astype(float).corr().to_numpy(dtype=float)
    simulated_corr = simulated.loc[:, scale_columns].astype(float).corr().to_numpy(dtype=float)
    upper = np.triu_indices_from(human_corr, k=1)
    deltas = human_corr[upper] - simulated_corr[upper]
    return float(np.sqrt(np.mean(deltas**2)))


def _regression_beta(frame: pd.DataFrame, y_col: str, x_col: str) -> float:
    clean = frame[[y_col, x_col]].dropna().astype(float)
    if clean.shape[0] < 3 or np.isclose(clean[x_col].var(ddof=1), 0.0):
        return float("nan")
    model = sm.OLS(clean[y_col], sm.add_constant(clean[x_col])).fit()
    return float(model.params.get(x_col, float("nan")))


def _standardized_regression_beta(frame: pd.DataFrame, y_col: str, x_col: str) -> float:
    clean = frame[[y_col, x_col]].dropna().astype(float)
    if clean.shape[0] < 3:
        return float("nan")
    x_std = clean[x_col].std(ddof=1)
    y_std = clean[y_col].std(ddof=1)
    if np.isclose(x_std, 0.0) or np.isclose(y_std, 0.0):
        return float("nan")
    standardized = clean.copy()
    standardized[x_col] = (standardized[x_col] - standardized[x_col].mean()) / x_std
    standardized[y_col] = (standardized[y_col] - standardized[y_col].mean()) / y_std
    model = sm.OLS(standardized[y_col], sm.add_constant(standardized[x_col])).fit()
    return float(model.params.get(x_col, float("nan")))


def _quadratic_coefficients(
    frame: pd.DataFrame,
    y_col: str,
    x_col: str,
) -> dict[str, float]:
    clean = frame[[y_col, x_col]].dropna().astype(float).copy()
    if clean.shape[0] < 4 or np.isclose(clean[x_col].var(ddof=1), 0.0):
        return {
            "intercept": float("nan"),
            "linear_beta": float("nan"),
            "quadratic_beta": float("nan"),
            "r_squared": float("nan"),
        }
    clean["x_sq"] = clean[x_col] ** 2
    model = sm.OLS(clean[y_col], sm.add_constant(clean[[x_col, "x_sq"]])).fit()
    return {
        "intercept": float(model.params.get("const", float("nan"))),
        "linear_beta": float(model.params.get(x_col, float("nan"))),
        "quadratic_beta": float(model.params.get("x_sq", float("nan"))),
        "r_squared": float(model.rsquared),
    }


def _profile_recovery_accuracy(
    human: pd.DataFrame,
    simulated: pd.DataFrame,
    profile_column: str,
    scale_columns: list[str],
) -> float:
    centroid_source = human[[profile_column, *scale_columns]].dropna().copy()
    if centroid_source.empty:
        return float("nan")

    centroids = centroid_source.groupby(profile_column)[scale_columns].mean().astype(float)
    if len(centroids) < 2:
        # With a single centroid every respondent is trivially nearest to it, so human
        # and simulated labels agree by construction and the metric reads 1.000 whatever
        # the model produced. That is not recovery, so the metric is undefined here.
        # This arises under cold-start assignment when the membership predictor collapses
        # onto one profile; it cannot arise under oracle assignment, where the mixture
        # supplies every fitted profile.
        return float("nan")

    scale_spreads = pd.Series(
        centroid_source.loc[:, scale_columns].astype(float).std(ddof=1),
        index=scale_columns,
        dtype="float64",
    ).replace(0.0, 1.0).fillna(1.0)
    standardized_centroids = centroids.div(scale_spreads, axis=1)
    centroid_labels = standardized_centroids.index.to_numpy()
    centroid_vectors = standardized_centroids.to_numpy(dtype=float)

    def nearest_labels(frame: Any) -> pd.Series:
        clean = pd.DataFrame(frame).loc[:, scale_columns].dropna().astype(float)
        if clean.empty:
            return pd.Series(dtype=object)
        standardized = clean.div(scale_spreads, axis=1)
        distances = (
            standardized.to_numpy(dtype=float)[:, None, :] - centroid_vectors[None, :, :]
        ) ** 2
        nearest = distances.sum(axis=2).argmin(axis=1)
        return pd.Series(centroid_labels[nearest], index=clean.index)

    if "respondent_id" in human.columns and "respondent_id" in simulated.columns:
        joined = human[["respondent_id", *scale_columns]].merge(
            simulated[["respondent_id", *scale_columns]],
            on="respondent_id",
            suffixes=("_human", "_simulated"),
        )
        if joined.empty:
            return float("nan")

        human_vectors = joined[[f"{column}_human" for column in scale_columns]].copy()
        human_vectors.columns = scale_columns
        simulated_vectors = joined[
            [f"{column}_simulated" for column in scale_columns]
        ].copy()
        simulated_vectors.columns = scale_columns
        human_labels = nearest_labels(human_vectors)
        simulated_labels = nearest_labels(simulated_vectors)
        aligned = human_labels.to_frame("human").join(
            simulated_labels.to_frame("simulated"),
            how="inner",
        )
        if aligned.empty:
            return float("nan")
        return float((aligned["human"] == aligned["simulated"]).mean())

    human_labels = nearest_labels(human.loc[:, scale_columns].copy())
    simulated_labels = nearest_labels(simulated.loc[:, scale_columns].copy())
    if human_labels.empty or simulated_labels.empty:
        return float("nan")

    human_dist = human_labels.value_counts(normalize=True).sort_index()
    simulated_dist = simulated_labels.value_counts(normalize=True).sort_index()
    aligned = human_dist.to_frame("human").join(
        simulated_dist.to_frame("simulated"),
        how="outer",
    ).fillna(0.0)
    return float(1.0 - 0.5 * np.abs(aligned["human"] - aligned["simulated"]).sum())


def _unique_pattern_ratio(frame: pd.DataFrame, item_columns: list[str]) -> float:
    if not item_columns:
        return float("nan")
    clean = frame[item_columns].dropna()
    if clean.empty:
        return float("nan")
    return float(clean.drop_duplicates().shape[0] / clean.shape[0])


def _subgroup_rmse(
    human: pd.DataFrame,
    simulated: pd.DataFrame,
    *,
    scale_columns: list[str],
    subgroup_column: str,
) -> float:
    if subgroup_column not in human.columns or subgroup_column not in simulated.columns:
        return float("nan")

    human_grouped = human.groupby(subgroup_column)[scale_columns].mean()
    simulated_grouped = simulated.groupby(subgroup_column)[scale_columns].mean()
    aligned = human_grouped.join(simulated_grouped, how="outer", lsuffix="_human", rsuffix="_simulated")
    if aligned.empty:
        return float("nan")

    deltas: list[float] = []
    for scale in scale_columns:
        human_col = f"{scale}_human"
        simulated_col = f"{scale}_simulated"
        scale_delta = (aligned[human_col].fillna(aligned[human_col].mean()) - aligned[simulated_col].fillna(aligned[simulated_col].mean())).to_numpy(dtype=float)
        deltas.extend(scale_delta.tolist())
    return float(np.sqrt(np.mean(np.square(deltas)))) if deltas else float("nan")


def _within_profile_scale_rmse(
    human: pd.DataFrame,
    simulated: pd.DataFrame,
    *,
    scale_columns: list[str],
    profile_column: str,
) -> float:
    if profile_column not in human.columns or profile_column not in simulated.columns:
        return float("nan")

    human_grouped = human.groupby(profile_column)[scale_columns].mean()
    simulated_grouped = simulated.groupby(profile_column)[scale_columns].mean()
    aligned = human_grouped.join(
        simulated_grouped,
        how="outer",
        lsuffix="_human",
        rsuffix="_simulated",
    )
    if aligned.empty:
        return float("nan")

    deltas: list[float] = []
    for scale in scale_columns:
        human_values = aligned[f"{scale}_human"].fillna(human_grouped[scale].mean())
        simulated_values = aligned[f"{scale}_simulated"].fillna(
            simulated_grouped[scale].mean()
        )
        deltas.extend((human_values - simulated_values).to_numpy(dtype=float).tolist())
    return float(np.sqrt(np.mean(np.square(deltas)))) if deltas else float("nan")


def _individual_level_metrics(
    human: pd.DataFrame,
    simulated: pd.DataFrame,
    *,
    scale_columns: list[str],
    item_columns: list[str],
) -> dict[str, float]:
    metrics = {
        "aligned_respondent_count": 0.0,
        "respondent_scale_rmse_mean": float("nan"),
        "respondent_item_mae_mean": float("nan"),
        "respondent_item_exact_match_mean": float("nan"),
        "respondent_item_all_match_ratio": float("nan"),
    }
    if "respondent_id" not in human.columns or "respondent_id" not in simulated.columns:
        return metrics

    joined = human[["respondent_id", *scale_columns, *item_columns]].merge(
        simulated[["respondent_id", *scale_columns, *item_columns]],
        on="respondent_id",
        suffixes=("_human", "_simulated"),
    )
    if joined.empty:
        return metrics

    metrics["aligned_respondent_count"] = float(joined.shape[0])

    if scale_columns:
        human_scale = joined[[f"{column}_human" for column in scale_columns]].astype(float)
        simulated_scale = joined[[f"{column}_simulated" for column in scale_columns]].astype(float)
        scale_delta = human_scale.to_numpy(dtype=float) - simulated_scale.to_numpy(dtype=float)
        respondent_scale_rmse = np.sqrt(np.mean(np.square(scale_delta), axis=1))
        metrics["respondent_scale_rmse_mean"] = float(np.mean(respondent_scale_rmse))

    if item_columns:
        human_items = joined[[f"{column}_human" for column in item_columns]].astype(float)
        simulated_items = joined[[f"{column}_simulated" for column in item_columns]].astype(float)
        human_array = human_items.to_numpy(dtype=float)
        simulated_array = simulated_items.to_numpy(dtype=float)
        # Item responses are occasionally missing in the human data (0.20% of Turkish
        # cells). Those items carry no information about accuracy, so they are excluded
        # from both metrics rather than propagated: a plain mean would return NaN for the
        # whole run, and an equality test would silently score a missing item as a
        # mismatch. Splits whose held-out set happens to contain no missing item are
        # unaffected, which includes every split reported in the paper.
        observed = ~np.isnan(human_array) & ~np.isnan(simulated_array)
        item_delta = human_array - simulated_array
        observed_per_respondent = observed.sum(axis=1)
        with np.errstate(invalid="ignore"):
            respondent_item_mae = np.nanmean(
                np.where(observed, np.abs(item_delta), np.nan), axis=1
            )
            respondent_item_exact_match = np.divide(
                ((human_array == simulated_array) & observed).sum(axis=1),
                observed_per_respondent,
                out=np.full(observed_per_respondent.shape, np.nan, dtype=float),
                where=observed_per_respondent > 0,
            )
        metrics["respondent_item_mae_mean"] = float(np.nanmean(respondent_item_mae))
        metrics["respondent_item_exact_match_mean"] = float(
            np.nanmean(respondent_item_exact_match)
        )
        metrics["respondent_item_all_match_ratio"] = float(
            np.nanmean((respondent_item_exact_match == 1.0).astype(float))
        )

    return metrics


def evaluate_simulation(
    human: pd.DataFrame,
    simulated: pd.DataFrame,
    *,
    scale_columns: list[str],
    item_columns: list[str] | None = None,
    profile_column: str,
    regression_pairs: list[RegressionPairSettings] | None = None,
    subgroup_columns: list[str] | None = None,
) -> EvaluationReport:
    scale_mean_deltas = {
        column: float(abs(human[column].mean() - simulated[column].mean()))
        for column in scale_columns
    }
    scale_std_deltas = {
        column: float(abs(human[column].std(ddof=1) - simulated[column].std(ddof=1)))
        for column in scale_columns
    }
    item_mean_deltas = {
        column: float(abs(human[column].mean() - simulated[column].mean()))
        for column in (item_columns or [])
    }

    regression_deltas: dict[str, float] = {}
    standardized_regression_deltas: dict[str, float] = {}
    nonlinear_deltas: dict[str, float] = {}
    for pair in regression_pairs or list(DEFAULT_DATASET.regression_pairs):
        if {pair.x, pair.y}.issubset(scale_columns):
            regression_deltas[pair.name] = float(
                _safe_abs_delta(
                    _regression_beta(human, pair.y, pair.x),
                    _regression_beta(simulated, pair.y, pair.x),
                )
            )
            standardized_regression_deltas[pair.name] = float(
                _safe_abs_delta(
                    _standardized_regression_beta(human, pair.y, pair.x),
                    _standardized_regression_beta(simulated, pair.y, pair.x),
                )
            )
            human_quadratic = _quadratic_coefficients(human, pair.y, pair.x)
            simulated_quadratic = _quadratic_coefficients(simulated, pair.y, pair.x)
            nonlinear_deltas[f"{pair.name}_linear"] = float(
                _safe_abs_delta(
                    human_quadratic["linear_beta"],
                    simulated_quadratic["linear_beta"],
                )
            )
            nonlinear_deltas[f"{pair.name}_quadratic"] = float(
                _safe_abs_delta(
                    human_quadratic["quadratic_beta"],
                    simulated_quadratic["quadratic_beta"],
                )
            )

    human_unique_ratio = _unique_pattern_ratio(human, item_columns or [])
    simulated_unique_ratio = _unique_pattern_ratio(simulated, item_columns or [])
    response_pattern_diversity = {
        "human_unique_ratio": human_unique_ratio,
        "simulated_unique_ratio": simulated_unique_ratio,
        "absolute_delta": _safe_abs_delta(human_unique_ratio, simulated_unique_ratio),
    }
    subgroup_rmse = {
        subgroup: _subgroup_rmse(
            human,
            simulated,
            scale_columns=scale_columns,
            subgroup_column=subgroup,
        )
        for subgroup in dict.fromkeys(
            [*(subgroup_columns or list(DEFAULT_DATASET.subgroup_columns)), profile_column]
        )
    }
    individual_level_metrics = _individual_level_metrics(
        human,
        simulated,
        scale_columns=scale_columns,
        item_columns=item_columns or [],
    )

    return EvaluationReport(
        scale_mean_deltas=scale_mean_deltas,
        scale_std_deltas=scale_std_deltas,
        correlation_rmse=_correlation_rmse(human, simulated, scale_columns),
        regression_deltas=regression_deltas,
        standardized_regression_deltas=standardized_regression_deltas,
        profile_recovery_accuracy=_profile_recovery_accuracy(
            human,
            simulated,
            profile_column,
            scale_columns,
        ),
        within_profile_scale_rmse=_within_profile_scale_rmse(
            human,
            simulated,
            scale_columns=scale_columns,
            profile_column=profile_column,
        ),
        item_mean_deltas=item_mean_deltas,
        nonlinear_deltas=nonlinear_deltas,
        response_pattern_diversity=response_pattern_diversity,
        subgroup_rmse=subgroup_rmse,
        individual_level_metrics=individual_level_metrics,
    )
