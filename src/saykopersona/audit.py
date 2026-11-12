from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, pearsonr, skew

from .data import DEFAULT_DATASET, compute_reliability
from .dataset_spec import DatasetSettings


@dataclass(slots=True)
class DescriptiveStat:
    n: int
    mean: float
    std: float
    skewness: float
    kurtosis: float


@dataclass(slots=True)
class RegressionStat:
    intercept: float
    beta: float
    r_squared: float
    p_value: float


@dataclass(slots=True)
class AuditReport:
    dataset_n: int
    paper_n: int | None
    sample_mismatch: int | None
    reliability: dict[str, float]
    descriptives: dict[str, DescriptiveStat]
    correlations: dict[str, float]
    regressions: dict[str, RegressionStat]
    anomalies: list[str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["descriptives"] = {
            key: asdict(value) for key, value in self.descriptives.items()
        }
        payload["regressions"] = {
            key: asdict(value) for key, value in self.regressions.items()
        }
        return payload


def _describe(series: pd.Series) -> DescriptiveStat:
    clean = series.dropna().astype(float)
    return DescriptiveStat(
        n=int(clean.shape[0]),
        mean=float(clean.mean()),
        std=float(clean.std(ddof=1)),
        skewness=float(skew(clean, bias=False)),
        kurtosis=float(kurtosis(clean, fisher=True, bias=False)),
    )


def _run_regression(y: pd.Series, x: pd.Series) -> RegressionStat:
    data = pd.DataFrame({"y": y, "x": x}).dropna()
    model = sm.OLS(data["y"], sm.add_constant(data["x"])).fit()
    return RegressionStat(
        intercept=float(model.params["const"]),
        beta=float(model.params["x"]),
        r_squared=float(model.rsquared),
        p_value=float(model.pvalues["x"]),
    )


def _detect_anomalies(
    scored: pd.DataFrame,
    *,
    dataset: DatasetSettings,
) -> list[str]:
    anomalies: list[str] = []
    audit_reference = dataset.audit_reference
    if audit_reference is None:
        return ["No external paper audit reference configured for this dataset."]

    if audit_reference.paper_sample_size is not None:
        sample_mismatch = len(scored) - audit_reference.paper_sample_size
        if sample_mismatch != 0:
            anomalies.append(
                f"Dataset contains {len(scored)} usable rows while the reference reports "
                f"{audit_reference.paper_sample_size} (difference: {sample_mismatch})."
            )

    for column, expected_counts in audit_reference.expected_category_counts.items():
        if column not in scored.columns:
            anomalies.append(f"Expected audit column '{column}' is missing from the scored dataset.")
            continue
        observed_counts = scored[column].value_counts(dropna=False).to_dict()
        for label, expected in expected_counts.items():
            observed = int(observed_counts.get(label, 0))
            if observed != expected:
                anomalies.append(
                    f"Column '{column}' count for '{label}' is {observed} in the dataset versus "
                    f"{expected} in the reference."
                )

    if not anomalies:
        anomalies.append("No workbook-to-reference discrepancies were detected.")
    return anomalies


def build_audit_report(
    scored: pd.DataFrame,
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> AuditReport:
    descriptives = {
        scale.name: _describe(scored[scale.mean_column])
        for scale in dataset.scales
    }
    correlations = {}
    for left_scale, right_scale in combinations(dataset.scales, 2):
        correlations[f"{left_scale.name}_vs_{right_scale.name}"] = float(
            pearsonr(
                scored[left_scale.mean_column].astype(float),
                scored[right_scale.mean_column].astype(float),
            ).statistic
        )
    if dataset.dataset_id == DEFAULT_DATASET.dataset_id:
        legacy_aliases = {
            "digital_vs_social": "digital_addiction_vs_social_anxiety",
            "digital_vs_wellbeing": "digital_addiction_vs_wellbeing",
            "social_vs_wellbeing": "social_anxiety_vs_wellbeing",
        }
        for legacy_key, dynamic_key in legacy_aliases.items():
            if dynamic_key in correlations:
                correlations.setdefault(legacy_key, correlations[dynamic_key])
    regressions = {
        pair.name: _run_regression(scored[pair.y], scored[pair.x])
        for pair in dataset.regression_pairs
    }
    paper_n = dataset.audit_reference.paper_sample_size if dataset.audit_reference else None
    return AuditReport(
        dataset_n=int(len(scored)),
        paper_n=paper_n,
        sample_mismatch=(int(len(scored) - paper_n) if paper_n is not None else None),
        reliability=compute_reliability(scored, dataset=dataset),
        descriptives=descriptives,
        correlations=correlations,
        regressions=regressions,
        anomalies=_detect_anomalies(scored, dataset=dataset),
    )


def write_audit_outputs(report: AuditReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            key: {
                "n": value.n,
                "mean": value.mean,
                "std": value.std,
                "skewness": value.skewness,
                "kurtosis": value.kurtosis,
            }
            for key, value in report.descriptives.items()
        }
    ).T.to_csv(output_dir / "baseline_descriptives.csv", index_label="scale")

    pd.DataFrame(
        {
            key: {
                "intercept": value.intercept,
                "beta": value.beta,
                "r_squared": value.r_squared,
                "p_value": value.p_value,
            }
            for key, value in report.regressions.items()
        }
    ).T.to_csv(output_dir / "baseline_regressions.csv", index_label="model")

    (output_dir / "audit_report.json").write_text(
        pd.Series(report.to_dict()).to_json(indent=2, force_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "audit_report.md").write_text(
        "\n".join(
            [
                "# Baseline Audit Report",
                "",
                f"- Workbook usable rows: {report.dataset_n}",
                f"- Paper reported rows: {report.paper_n if report.paper_n is not None else 'N/A'}",
                f"- Sample mismatch: {report.sample_mismatch if report.sample_mismatch is not None else 'N/A'}",
                "",
                "## Anomalies",
                *[f"- {item}" for item in report.anomalies],
            ]
        ),
        encoding="utf-8",
    )
