from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .data import DEFAULT_DATASET
from .dataset_spec import DatasetSettings


@dataclass(slots=True)
class ProfileResult:
    selected_k: int
    feature_columns: list[str]
    assignments: pd.DataFrame
    summary: pd.DataFrame
    model_scores: pd.DataFrame
    scaler: StandardScaler
    model: GaussianMixture | KMeans


def _fit_candidate_models(
    frame: pd.DataFrame,
    feature_columns: list[str],
    cluster_range: range,
    random_state: int,
) -> tuple[pd.DataFrame, StandardScaler, GaussianMixture]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(frame[feature_columns].astype(float))

    fitted_models: dict[int, GaussianMixture] = {}
    rows: list[dict[str, float | int]] = []
    reg_covar = 1e-6
    for n_components in cluster_range:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
            n_init=10,
            reg_covar=reg_covar,
        )
        labels = model.fit_predict(scaled)
        fitted_models[n_components] = model
        # A component whose covariance has collapsed onto the regularization floor has
        # near-zero volume, so its density is unbounded and the likelihood, and with it
        # the information criteria, are not comparable to a proper fit. Such solutions
        # are excluded from selection rather than allowed to win on an inflated
        # likelihood: on the Bangladesh scales alone they produced BIC values of 3823.7
        # at four components and 3887.4 at six against 4538.5 for the three-component
        # fit, which made a degenerate solution look like the best available.
        smallest_eigenvalue = min(
            float(np.linalg.eigvalsh(covariance).min()) for covariance in model.covariances_
        )
        rows.append(
            {
                "k": n_components,
                "bic": float(model.bic(scaled)),
                "aic": float(model.aic(scaled)),
                "silhouette": float(silhouette_score(scaled, labels)),
                "min_component_eigenvalue": smallest_eigenvalue,
                "degenerate": bool(smallest_eigenvalue <= reg_covar * 1.001),
            }
        )

    scores = pd.DataFrame(rows)
    eligible = scores[~scores["degenerate"]]
    if eligible.empty:
        eligible = scores
    scores_all = scores
    scores = eligible.reset_index(drop=True)
    ordered = scores.sort_values("silhouette", ascending=False).reset_index(drop=True)
    best_silhouette = float(ordered.iloc[0]["silhouette"])
    second_best_silhouette = (
        float(ordered.iloc[1]["silhouette"]) if len(ordered) > 1 else float("-inf")
    )

    # Prefer the simplest/highest-separation solution only when silhouette is clearly dominant.
    # Otherwise, among plausible solutions, let information criteria choose the richer profile set.
    if best_silhouette - second_best_silhouette >= 0.2:
        selected_k = int(ordered.iloc[0]["k"])
    else:
        silhouette_floor = max(0.1, best_silhouette - 0.2)
        candidate_pool = scores[scores["silhouette"] >= silhouette_floor].copy()
        candidate_pool = candidate_pool.sort_values(
            ["bic", "aic", "silhouette", "k"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
        selected_k = int(candidate_pool.iloc[0]["k"])

    scores = scores_all.sort_values("k").reset_index(drop=True)
    scores["selected"] = scores["k"] == selected_k
    return scores, scaler, fitted_models[selected_k]


def _fit_kmeans_models(
    frame: pd.DataFrame,
    feature_columns: list[str],
    cluster_range: range,
    random_state: int,
) -> tuple[pd.DataFrame, StandardScaler, KMeans]:
    """k-means on the same standardised features; the count is chosen by silhouette."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(frame[feature_columns].astype(float))
    fitted: dict[int, KMeans] = {}
    rows: list[dict[str, float | int]] = []
    for n_clusters in cluster_range:
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(scaled)
        fitted[n_clusters] = model
        rows.append(
            {
                "k": n_clusters,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled, model.labels_)),
            }
        )
    scores = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    selected_k = int(scores.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    scores["selected"] = scores["k"] == selected_k
    return scores, scaler, fitted[selected_k]


def _summarize_profiles(
    assignments: pd.DataFrame,
    feature_columns: list[str],
    anchor_columns: list[str] | None = None,
) -> pd.DataFrame:
    summary = (
        assignments.groupby("profile_id")[feature_columns]
        .mean()
        .rename_axis("profile_id")
        .reset_index()
    )
    sizes = assignments.groupby("profile_id").size().rename("size").reset_index()
    summary = summary.merge(sizes, on="profile_id", how="left")

    for column in anchor_columns or []:
        if column not in assignments.columns:
            continue
        modes = (
            assignments.groupby("profile_id")[column]
            .agg(lambda series: series.dropna().mode().iloc[0] if not series.dropna().empty else None)
            .rename(f"modal_{column}")
            .reset_index()
        )
        summary = summary.merge(modes, on="profile_id", how="left")

    return summary


def fit_empirical_profiles(
    frame: pd.DataFrame,
    feature_columns: list[str],
    cluster_range: range = range(2, 7),
    random_state: int = 42,
    anchor_columns: list[str] | None = None,
    method: str = "gmm",
) -> ProfileResult:
    if method not in {"gmm", "kmeans"}:
        raise ValueError(f"unknown profile method '{method}'")
    fit = _fit_candidate_models if method == "gmm" else _fit_kmeans_models
    score_table, scaler, model = fit(
        frame,
        feature_columns=feature_columns,
        cluster_range=cluster_range,
        random_state=random_state,
    )
    scaled = scaler.transform(frame[feature_columns].astype(float))
    profile_labels = model.predict(scaled)

    assignments = frame.copy()
    assignments["profile_id"] = profile_labels
    summary = _summarize_profiles(
        assignments,
        feature_columns,
        anchor_columns=anchor_columns,
    )

    return ProfileResult(
        selected_k=int(len(np.unique(profile_labels))),
        feature_columns=feature_columns,
        assignments=assignments,
        summary=summary,
        model_scores=score_table,
        scaler=scaler,
        model=model,
    )


def predict_profiles(result: ProfileResult, frame: pd.DataFrame) -> pd.Series:
    scaled = result.scaler.transform(frame[result.feature_columns].astype(float))
    return pd.Series(result.model.predict(scaled), index=frame.index, name="profile_id")


def build_profile_narratives(
    result: ProfileResult,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> list[dict[str, object]]:
    overall_means = result.assignments[result.feature_columns].mean()
    narratives: list[dict[str, object]] = []
    for row in result.summary.to_dict(orient="records"):
        profile_id = int(row["profile_id"])
        descriptors: list[str] = []
        for feature in result.feature_columns:
            descriptor = dataset.feature_descriptors.get(feature)
            if descriptor is None:
                label = dataset.feature_label(feature)
                descriptor_high = f"higher {label}"
                descriptor_low = f"lower {label}"
            else:
                descriptor_high = descriptor.high_text
                descriptor_low = descriptor.low_text
            if row[feature] >= overall_means.get(feature, 0):
                descriptors.append(descriptor_high)
            else:
                descriptors.append(descriptor_low)

        anchor_parts = [row.get(f"modal_{column}") for column in dataset.profile_anchor_columns]
        demographic_anchor = ", ".join(str(part) for part in anchor_parts if part)
        if not demographic_anchor:
            unit_label = "öğrenci" if dataset.locale == "tr" else "respondents"
            demographic_anchor = f"{int(row['size'])} {unit_label}"

        narratives.append(
            {
                "profile_id": profile_id,
                "label": f"Profil {profile_id}",
                "summary": ", ".join(descriptors),
                "demographic_anchor": demographic_anchor,
                "size": int(row["size"]),
                "feature_means": {
                    feature: float(row[feature]) for feature in result.feature_columns
                },
            }
        )
    return narratives
