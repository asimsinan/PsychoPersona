"""Compute dataset psychometrics for the BRM revision.

For each dataset (Turkish anket, Bangladesh SNS) and each scale we report:
- item count and response range
- reverse-coded items
- missingness (item-level and listwise)
- Cronbach's alpha and McDonald's omega (single-factor)
- mean inter-item correlation
- scale-score descriptives (M, SD, min, max, skewness, kurtosis)

Outputs two CSVs in the results/ root: psychometrics_scales.csv and
psychometrics_items.csv, plus a console summary used to populate the manuscript.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from saykopersona.data import cronbach_alpha, load_raw_survey, score_survey_dataframe  # noqa: E402
from saykopersona.dataset_catalog import (  # noqa: E402
    bangladesh_sns_dataset_settings,
    default_dataset_settings,
)

DATASETS = {
    "anket_tr": (default_dataset_settings(), ROOT / "datasets" / "anket.xlsx"),
    "bangladesh_sns": (
        bangladesh_sns_dataset_settings(),
        ROOT / "datasets" / "bangladesh_sns" / "mmc2.xlsx",
    ),
}


def mcdonald_omega(items: pd.DataFrame) -> float:
    """Single-factor McDonald's omega_total from standardized loadings."""
    clean = items.dropna().astype(float)
    if clean.shape[1] < 2 or clean.shape[0] < clean.shape[1] + 1:
        return float("nan")
    # standardize so loadings/uniquenesses are on a comparable scale
    z = (clean - clean.mean()) / clean.std(ddof=1)
    z = z.dropna(axis=1, how="any")
    if z.shape[1] < 2:
        return float("nan")
    fa = FactorAnalysis(n_components=1, random_state=42)
    fa.fit(z.values)
    loadings = fa.components_.ravel()
    uniqueness = fa.noise_variance_
    sum_load_sq = float(np.sum(loadings) ** 2)
    denom = sum_load_sq + float(np.sum(uniqueness))
    if denom <= 0:
        return float("nan")
    return sum_load_sq / denom


def mean_interitem_corr(items: pd.DataFrame) -> float:
    clean = items.dropna().astype(float)
    if clean.shape[1] < 2:
        return float("nan")
    corr = clean.corr().values
    iu = np.triu_indices_from(corr, k=1)
    return float(np.nanmean(corr[iu]))


def main() -> None:
    scale_rows: list[dict[str, object]] = []
    item_rows: list[dict[str, object]] = []

    for dataset_id, (settings, path) in DATASETS.items():
        if not path.exists():
            print(f"[skip] {dataset_id}: file not found {path}")
            continue
        raw = load_raw_survey(path, dataset=settings)
        scored = score_survey_dataframe(raw, dataset=settings)
        n_total = len(scored)
        print(f"\n=== {dataset_id} (N={n_total}, range {settings.response_min}-{settings.response_max}) ===")

        for scale in settings.scales:
            items = scored[scale.item_columns]
            n_items = len(scale.item_columns)
            # missingness
            item_missing = items.isna().mean().mean() * 100.0
            complete = items.dropna()
            n_complete = len(complete)
            listwise_missing = (1 - n_complete / n_total) * 100.0 if n_total else float("nan")

            alpha = cronbach_alpha(items)
            omega = mcdonald_omega(items)
            mic = mean_interitem_corr(items)

            scale_score = scored[scale.mean_column]
            valid_scores = scale_score.dropna()
            m = float(valid_scores.mean())
            sd = float(valid_scores.std(ddof=1))
            skew = float(stats.skew(valid_scores)) if len(valid_scores) > 2 else float("nan")
            kurt = float(stats.kurtosis(valid_scores)) if len(valid_scores) > 2 else float("nan")

            reverse = ", ".join(scale.reverse_item_columns) if scale.reverse_item_columns else "none"

            scale_rows.append(
                {
                    "dataset": dataset_id,
                    "scale": scale.name,
                    "label": scale.label,
                    "n_items": n_items,
                    "response_min": settings.response_min,
                    "response_max": settings.response_max,
                    "reverse_coded": reverse,
                    "n_respondents": n_total,
                    "n_complete": n_complete,
                    "item_missing_pct": round(item_missing, 2),
                    "listwise_missing_pct": round(listwise_missing, 2),
                    "alpha": round(alpha, 3) if not np.isnan(alpha) else None,
                    "omega": round(omega, 3) if not np.isnan(omega) else None,
                    "mean_interitem_r": round(mic, 3) if not np.isnan(mic) else None,
                    "mean": round(m, 4),
                    "sd": round(sd, 4),
                    "min": round(float(valid_scores.min()), 4),
                    "max": round(float(valid_scores.max()), 4),
                    "skew": round(skew, 4) if not np.isnan(skew) else None,
                    "kurtosis": round(kurt, 4) if not np.isnan(kurt) else None,
                }
            )

            print(
                f"  {scale.name:18s} k={n_items:2d} alpha={alpha:.3f} omega={omega:.3f} "
                f"mic={mic:.3f} M={m:.2f} SD={sd:.2f} miss(item)={item_missing:.2f}% "
                f"miss(listwise)={listwise_missing:.2f}% reverse=[{reverse}]"
            )

            for item_col in scale.item_columns:
                col = scored[item_col].dropna()
                item_rows.append(
                    {
                        "dataset": dataset_id,
                        "scale": scale.name,
                        "item": item_col,
                        "n": len(col),
                        "missing_pct": round(scored[item_col].isna().mean() * 100.0, 2),
                        "mean": round(float(col.mean()), 3),
                        "sd": round(float(col.std(ddof=1)), 3),
                    }
                )

    scales_df = pd.DataFrame(scale_rows)
    items_df = pd.DataFrame(item_rows)
    scales_path = ROOT / "results" / "psychometrics_scales.csv"
    items_path = ROOT / "results" / "psychometrics_items.csv"
    scales_df.to_csv(scales_path, index=False)
    items_df.to_csv(items_path, index=False)
    print(f"\nWrote {scales_path}")
    print(f"Wrote {items_path}")


if __name__ == "__main__":
    main()
