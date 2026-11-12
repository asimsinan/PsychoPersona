from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset_catalog import default_dataset_settings
from .dataset_spec import DatasetSettings, MetadataFieldSettings, ScaleSettings


DEFAULT_DATASET = default_dataset_settings()

TIMESTAMP_COLUMN = DEFAULT_DATASET.metadata_field("timestamp").source_column
GENDER_COLUMN = DEFAULT_DATASET.metadata_field("gender").source_column
AGE_COLUMN = DEFAULT_DATASET.metadata_field("age_band").source_column
PROGRAM_COLUMN = DEFAULT_DATASET.metadata_field("program").source_column
GPA_COLUMN = DEFAULT_DATASET.metadata_field("gpa_band").source_column
RESIDENCE_COLUMN = DEFAULT_DATASET.metadata_field("residence").source_column
PLATFORMS_COLUMN = DEFAULT_DATASET.metadata_field("platforms").source_column
USAGE_COLUMN = DEFAULT_DATASET.metadata_field("usage_band").source_column

DIGITAL_SCALE = DEFAULT_DATASET.scale_for_mean_column("digital_addiction_mean")
SOCIAL_SCALE = DEFAULT_DATASET.scale_for_mean_column("social_anxiety_mean")
WELLBEING_SCALE = DEFAULT_DATASET.scale_for_mean_column("wellbeing_mean")

DIGITAL_RAW_COLUMNS = DIGITAL_SCALE.raw_columns
SOCIAL_RAW_COLUMNS = SOCIAL_SCALE.raw_columns
WELLBEING_RAW_COLUMNS = WELLBEING_SCALE.raw_columns

DIGITAL_SCORE_COLUMNS = DIGITAL_SCALE.item_columns
SOCIAL_SCORE_COLUMNS = SOCIAL_SCALE.item_columns
WELLBEING_SCORE_COLUMNS = WELLBEING_SCALE.item_columns

LIKERT_MAP = DIGITAL_SCALE.response_map
USAGE_HOURS_MAP = DEFAULT_DATASET.metadata_field("usage_hours_midpoint").value_map
SCALE_COLUMN_MAP = {
    scale.name: list(scale.item_columns) for scale in DEFAULT_DATASET.scales
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_default_dataset_path() -> Path:
    return locate_dataset(get_project_root() / "anket.xlsx")


def locate_dataset(path: Path) -> Path:
    """Return ``path``, or the file of that name in ``datasets/`` when ``path`` is absent.

    Run configurations written before the survey files moved into ``datasets/`` still
    name the old location. The file itself is the same one.
    """
    path = Path(path)
    if not path.exists():
        moved = get_project_root() / "datasets" / path.name
        if moved.exists():
            return moved
    return path


def load_raw_survey(
    path: Path | str,
    dataset: DatasetSettings | None = None,
) -> pd.DataFrame:
    dataset_spec = dataset or default_dataset_settings()
    resolved_path = Path(path)
    file_format = dataset_spec.file_format
    if file_format == "auto":
        suffix = resolved_path.suffix.casefold()
        file_format = "csv" if suffix == ".csv" else "xlsx"
    if file_format == "csv":
        frame = pd.read_csv(resolved_path)
    else:
        read_kwargs: dict[str, object] = {}
        if dataset_spec.sheet_name is not None:
            read_kwargs["sheet_name"] = dataset_spec.sheet_name
        frame = pd.read_excel(resolved_path, **read_kwargs)
    frame = frame.dropna(how="all").reset_index(drop=True)
    return frame


def _normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize_age_band(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if "-" in stripped or "+" in stripped:
            return stripped
        if stripped.isdigit():
            value = int(stripped)
        else:
            return stripped or None
    if isinstance(value, (int, float, np.integer, np.floating)):
        age = int(value)
        if age <= 18:
            return "16-18"
        if age <= 20:
            return "19-20"
        if age <= 22:
            return "21-22"
        return "23+"
    return str(value)


def _normalize_program(value: object) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    lowered = text.casefold().replace(" ", "")
    if lowered == "önlisans":
        return "Ön lisans"
    if lowered == "lisans":
        return "Lisans"
    return text


def _lookup_mapped_value(
    value: object,
    mapping: dict[str, str | float],
) -> str | float | None:
    normalized = _normalize_text(value)
    if normalized is None:
        return None
    return mapping.get(normalized.casefold(), normalized)


def _transform_metadata_series(
    series: pd.Series,
    field: MetadataFieldSettings,
) -> pd.Series:
    if field.transform == "text":
        return series.map(_normalize_text)
    if field.transform == "mapped_text":
        mapping = field.normalized_value_map()
        return series.map(lambda value: _lookup_mapped_value(value, mapping))
    if field.transform == "mapped_number":
        mapping = field.normalized_value_map()
        return series.map(lambda value: _lookup_mapped_value(value, mapping)).astype("float64")
    if field.transform == "age_band_default":
        return series.map(_normalize_age_band)
    if field.transform == "program_tr":
        return series.map(_normalize_program)
    if field.transform == "float":
        return pd.to_numeric(series, errors="coerce")
    raise ValueError(f"Unsupported metadata transform: {field.transform}")


def _transform_item_value(transform_name: str, value: str | None) -> float | None:
    if value is None:
        return None
    if transform_name == "psqi_sleep_latency":
        lowered = value.casefold()
        if lowered.startswith("within 15"):
            return 0.0
        if lowered.startswith("16-30"):
            return 1.0
        if lowered.startswith("31-60"):
            return 2.0
        if lowered.startswith("more than"):
            return 3.0
        return None
    raise ValueError(f"Unsupported item transform: {transform_name}")


def _score_item_series(
    series: pd.Series,
    *,
    scale: ScaleSettings,
    item_column: str,
    dataset: DatasetSettings,
) -> pd.Series:
    mapping = scale.response_map_for_item(item_column)
    transform_name = scale.transform_for_item(item_column)

    def score_value(value: object) -> float | None:
        normalized = _normalize_text(value)
        transformed: float | None = None
        if transform_name is not None:
            transformed = _transform_item_value(transform_name, normalized)
        if transformed is not None:
            numeric = transformed
        elif normalized is not None and normalized.casefold() in mapping:
            numeric = float(mapping[normalized.casefold()])
        else:
            try:
                numeric = float(value) if value is not None else float("nan")
            except (TypeError, ValueError):
                return None
        if item_column in scale.reverse_item_columns:
            numeric = float(dataset.response_min + dataset.response_max - numeric)
        return float(
            np.clip(numeric, dataset.response_min, dataset.response_max)
        )

    return series.map(score_value).astype("float64")


def score_survey_dataframe(
    raw: pd.DataFrame,
    dataset: DatasetSettings | None = None,
) -> pd.DataFrame:
    dataset_spec = dataset or default_dataset_settings()
    scored = pd.DataFrame(index=raw.index)
    scored["respondent_id"] = [f"R{idx + 1:04d}" for idx in range(len(raw))]

    for field in dataset_spec.metadata_fields:
        if field.source_column not in raw.columns:
            if not field.include_in_context:
                scored[field.name] = None
                continue
            raise KeyError(
                f"Expected source column '{field.source_column}' for '{field.name}' "
                f"in dataset '{dataset_spec.dataset_id}'."
            )
        scored[field.name] = _transform_metadata_series(raw[field.source_column], field)

    for scale in dataset_spec.scales:
        for raw_column, item_column in zip(
            scale.raw_columns,
            scale.item_columns,
            strict=True,
        ):
            if raw_column not in raw.columns:
                raise KeyError(
                    f"Expected raw item column '{raw_column}' for scale '{scale.name}'."
                )
            scored[item_column] = _score_item_series(
                raw[raw_column],
                scale=scale,
                item_column=item_column,
                dataset=dataset_spec,
            )
        scored[scale.mean_column] = scored[scale.item_columns].mean(axis=1)

    return scored


def cronbach_alpha(item_frame: pd.DataFrame) -> float:
    clean = item_frame.dropna().astype(float)
    n_items = clean.shape[1]
    if n_items < 2 or clean.empty:
        return float("nan")
    item_variances = clean.var(axis=0, ddof=1)
    total_scores = clean.sum(axis=1)
    total_variance = total_scores.var(ddof=1)
    if np.isclose(total_variance, 0.0):
        return float("nan")
    return float(
        (n_items / (n_items - 1))
        * (1 - item_variances.sum() / total_variance)
    )


def compute_reliability(
    scored: pd.DataFrame,
    dataset: DatasetSettings | None = None,
) -> dict[str, float]:
    dataset_spec = dataset or default_dataset_settings()
    return {
        scale.name: cronbach_alpha(scored[scale.item_columns])
        for scale in dataset_spec.scales
    }


def _score_band_labels(series: pd.Series, q: int, label: str) -> pd.Series:
    try:
        bucketed = pd.qcut(series, q=q, duplicates="drop")
        return bucketed.astype(str).radd(f"{label}=")
    except ValueError:
        return pd.Series([f"{label}=all"] * len(series), index=series.index)


def _compose_labels(parts: list[pd.Series]) -> pd.Series:
    labels = parts[0].astype(str)
    for part in parts[1:]:
        labels = labels + "|" + part.astype(str)
    return labels


def _candidate_stratification_parts(
    scored: pd.DataFrame,
    dataset: DatasetSettings,
) -> list[pd.Series]:
    parts: list[pd.Series] = []
    for column in dataset.stratify_categorical_columns:
        if column in scored.columns:
            parts.append(scored[column].fillna("missing").astype(str))
    for column in dataset.stratify_score_columns:
        if column in scored.columns:
            label = column.removesuffix("_mean")
            parts.append(_score_band_labels(scored[column], q=4, label=label))
    return parts


def _build_stratification_labels(
    scored: pd.DataFrame,
    *,
    test_size: float,
    dataset: DatasetSettings,
) -> pd.Series:
    parts = _candidate_stratification_parts(scored, dataset)
    if not parts:
        return pd.Series(["all"] * len(scored), index=scored.index, dtype="object")

    test_count = int(round(len(scored) * test_size))
    train_count = len(scored) - test_count
    candidate_parts = [parts[: length] for length in range(len(parts), 0, -1)]
    for part_group in candidate_parts:
        labels = _compose_labels(part_group)
        counts = labels.value_counts()
        if counts.min() < 2:
            continue
        n_classes = int(counts.shape[0])
        if n_classes <= test_count and n_classes <= train_count:
            return labels
    return parts[0].astype(str)


def build_stratified_holdout_split(
    scored: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    dataset: DatasetSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_spec = dataset or default_dataset_settings()
    stratify_labels = _build_stratification_labels(
        scored,
        test_size=test_size,
        dataset=dataset_spec,
    )
    train_idx, holdout_idx = train_test_split(
        scored.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    train = scored.loc[train_idx].reset_index(drop=True)
    holdout = scored.loc[holdout_idx].reset_index(drop=True)
    return train, holdout


def build_stratified_three_way_split(
    scored: pd.DataFrame,
    validation_size: float = 0.1,
    final_test_size: float = 0.1,
    random_state: int = 42,
    dataset: DatasetSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_spec = dataset or default_dataset_settings()
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if not 0 < final_test_size < 1:
        raise ValueError("final_test_size must be between 0 and 1.")
    combined_eval_size = validation_size + final_test_size
    if combined_eval_size >= 1:
        raise ValueError("validation_size + final_test_size must be less than 1.")

    stratify_labels = _build_stratification_labels(
        scored,
        test_size=combined_eval_size,
        dataset=dataset_spec,
    )
    train_idx, eval_idx = train_test_split(
        scored.index,
        test_size=combined_eval_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    evaluation_pool = scored.loc[eval_idx]
    final_test_relative = final_test_size / combined_eval_size
    eval_labels = _build_stratification_labels(
        evaluation_pool,
        test_size=final_test_relative,
        dataset=dataset_spec,
    )
    validation_idx, final_test_idx = train_test_split(
        evaluation_pool.index,
        test_size=final_test_relative,
        random_state=random_state,
        stratify=eval_labels,
    )

    train = scored.loc[train_idx].reset_index(drop=True)
    validation = scored.loc[validation_idx].reset_index(drop=True)
    final_test = scored.loc[final_test_idx].reset_index(drop=True)
    return train, validation, final_test
