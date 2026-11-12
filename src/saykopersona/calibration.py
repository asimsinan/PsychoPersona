from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, create_model, model_validator

from .data import DEFAULT_DATASET
from .dataset_spec import DatasetSettings


DEFAULT_SCALE_COLUMNS = list(DEFAULT_DATASET.scale_columns)
DEFAULT_SUBGROUP_COLUMNS = list(DEFAULT_DATASET.subgroup_columns)
DEFAULT_SCALE_TO_ITEMS = DEFAULT_DATASET.scale_to_items
DEFAULT_ITEM_COLUMNS = list(DEFAULT_DATASET.item_columns)
DEFAULT_SCALE_SHIFT_ALIASES = {
    scale.mean_column: scale.effective_shift_key for scale in DEFAULT_DATASET.scales
}


class CalibrationSignal(BaseModel):
    scale_shifts: dict[str, float] = Field(default_factory=dict)
    digital_shift: float | None = Field(default=None, ge=-1.5, le=1.5)
    social_shift: float | None = Field(default=None, ge=-1.5, le=1.5)
    wellbeing_shift: float | None = Field(default=None, ge=-1.5, le=1.5)
    extremity_preference: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _merge_legacy_shift_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        scale_shifts = dict(payload.get("scale_shifts") or {})
        for legacy_key in ["digital_shift", "social_shift", "wellbeing_shift"]:
            if legacy_key in payload and payload[legacy_key] is not None:
                scale_shifts.setdefault(legacy_key.removesuffix("_shift"), payload[legacy_key])
        payload["scale_shifts"] = scale_shifts
        return payload

    @model_validator(mode="after")
    def _sync_legacy_shift_fields(self) -> CalibrationSignal:
        if self.digital_shift is None and "digital" in self.scale_shifts:
            self.digital_shift = float(self.scale_shifts["digital"])
        if self.social_shift is None and "social" in self.scale_shifts:
            self.social_shift = float(self.scale_shifts["social"])
        if self.wellbeing_shift is None and "wellbeing" in self.scale_shifts:
            self.wellbeing_shift = float(self.scale_shifts["wellbeing"])
        return self

    def shift_for_scale(self, scale_name: str, alias: str | None = None) -> float:
        if scale_name in self.scale_shifts:
            return float(self.scale_shifts[scale_name])
        if alias is not None and alias in self.scale_shifts:
            return float(self.scale_shifts[alias])
        simplified = scale_name.removesuffix("_mean")
        if simplified in self.scale_shifts:
            return float(self.scale_shifts[simplified])
        if alias == "digital" and self.digital_shift is not None:
            return float(self.digital_shift)
        if alias == "social" and self.social_shift is not None:
            return float(self.social_shift)
        if alias == "wellbeing" and self.wellbeing_shift is not None:
            return float(self.wellbeing_shift)
        return 0.0


def build_calibration_signal_schema(
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> type[BaseModel]:
    shift_model_name = f"{dataset.dataset_id.title().replace('_', '')}ScaleShifts"
    shift_fields = {
        scale.mean_column: (float, Field(..., ge=-1.5, le=1.5))
        for scale in dataset.scales
    }
    shift_model = create_model(shift_model_name, **shift_fields)

    schema_name = f"{dataset.dataset_id.title().replace('_', '')}CalibrationSignal"
    schema_model = create_model(
        schema_name,
        scale_shifts=(shift_model, Field(...)),
        extremity_preference=(float, Field(..., ge=0.0, le=1.0)),
        reasoning_note=(str | None, Field(...)),
    )

    def to_calibration_signal(self) -> CalibrationSignal:
        return CalibrationSignal(
            scale_shifts={
                key: float(value)
                for key, value in self.scale_shifts.model_dump(mode="json").items()
            },
            extremity_preference=float(self.extremity_preference),
            reasoning_note=self.reasoning_note,
        )

    setattr(schema_model, "to_calibration_signal", to_calibration_signal)
    return schema_model


@dataclass(slots=True)
class CalibrationArtifacts:
    scale_columns: list[str]
    item_columns: list[str]
    scale_to_items: dict[str, list[str]]
    scale_shift_aliases: dict[str, str]
    profile_column: str
    subgroup_columns: list[str]
    donor_pool_min: int
    profile_shrinkage: float
    signal_scale: float
    neighbor_count: int
    response_min: int
    response_max: int
    global_scale_means: dict[str, float]
    global_scale_stds: dict[str, float]
    global_scale_covariance: dict[str, dict[str, float]]
    profile_scale_means: dict[int, dict[str, float]]
    profile_scale_covariances: dict[int, dict[str, dict[str, float]]]
    profile_counts: dict[int, int]
    global_subgroup_offsets: dict[str, dict[str, dict[str, float]]]
    profile_subgroup_offsets: dict[int, dict[str, dict[str, dict[str, Any]]]]
    global_item_priority: dict[str, list[str]]
    profile_item_priority: dict[int, dict[str, list[str]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationArtifacts:
        profile_scale_means = {
            int(key): value for key, value in payload["profile_scale_means"].items()
        }
        profile_scale_covariances = {
            int(key): value for key, value in payload["profile_scale_covariances"].items()
        }
        profile_counts = {int(key): int(value) for key, value in payload["profile_counts"].items()}
        profile_subgroup_offsets = {
            int(key): value for key, value in payload["profile_subgroup_offsets"].items()
        }
        profile_item_priority = {
            int(key): value for key, value in payload["profile_item_priority"].items()
        }
        return cls(
            scale_columns=list(payload["scale_columns"]),
            item_columns=list(payload["item_columns"]),
            scale_to_items={
                str(key): list(value) for key, value in payload["scale_to_items"].items()
            },
            scale_shift_aliases={
                str(key): str(value)
                for key, value in payload.get("scale_shift_aliases", {}).items()
            },
            profile_column=str(payload["profile_column"]),
            subgroup_columns=list(payload["subgroup_columns"]),
            donor_pool_min=int(payload["donor_pool_min"]),
            profile_shrinkage=float(payload["profile_shrinkage"]),
            signal_scale=float(payload["signal_scale"]),
            neighbor_count=int(payload["neighbor_count"]),
            response_min=int(payload.get("response_min", 1)),
            response_max=int(payload.get("response_max", 5)),
            global_scale_means=dict(payload["global_scale_means"]),
            global_scale_stds=dict(payload["global_scale_stds"]),
            global_scale_covariance=dict(payload["global_scale_covariance"]),
            profile_scale_means=profile_scale_means,
            profile_scale_covariances=profile_scale_covariances,
            profile_counts=profile_counts,
            global_subgroup_offsets=dict(payload["global_subgroup_offsets"]),
            profile_subgroup_offsets=profile_subgroup_offsets,
            global_item_priority=dict(payload["global_item_priority"]),
            profile_item_priority=profile_item_priority,
        )


@dataclass(slots=True)
class CalibratedResponse:
    item_responses: dict[str, int]
    target_scale_values: dict[str, float]
    realized_scale_values: dict[str, float]
    base_scale_values: dict[str, float]
    selected_donor_id: str | None


def _ensure_covariance(frame: pd.DataFrame, scale_columns: list[str]) -> pd.DataFrame:
    covariance = frame[scale_columns].cov(ddof=1).reindex(
        index=scale_columns,
        columns=scale_columns,
    )
    covariance = covariance.fillna(0.0)
    for index, column in enumerate(scale_columns):
        if covariance.iat[index, index] <= 0:
            covariance.iat[index, index] = 0.05
    return covariance


def _series_to_float_dict(series: pd.Series) -> dict[str, float]:
    return {str(key): float(value) for key, value in series.items()}


def _covariance_to_dict(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(index): {str(column): float(value) for column, value in row.items()}
        for index, row in frame.iterrows()
    }


def _dict_to_covariance(
    payload: dict[str, dict[str, float]],
    scale_columns: list[str],
) -> np.ndarray:
    return np.array(
        [
            [float(payload[row][column]) for column in scale_columns]
            for row in scale_columns
        ],
        dtype="float64",
    )


def _build_item_priority(
    frame: pd.DataFrame,
    item_columns: list[str],
    scale_to_items: dict[str, list[str]],
) -> dict[str, list[str]]:
    priorities: dict[str, list[str]] = {}
    for scale_name, scale_items in scale_to_items.items():
        present_items = [item for item in scale_items if item in item_columns]
        if not present_items:
            continue
        item_means = frame[present_items].mean().sort_values(ascending=False)
        priorities[scale_name] = [str(item_id) for item_id in item_means.index]
    return priorities


def fit_calibration_artifacts(
    train_profiles: pd.DataFrame,
    *,
    scale_columns: list[str] | None = None,
    item_columns: list[str] | None = None,
    scale_to_items: dict[str, list[str]] | None = None,
    scale_shift_aliases: dict[str, str] | None = None,
    profile_column: str = "profile_id",
    subgroup_columns: list[str] | None = None,
    donor_pool_min: int = 20,
    profile_shrinkage: float = 0.25,
    signal_scale: float = 0.6,
    neighbor_count: int = 8,
    response_min: int | None = None,
    response_max: int | None = None,
) -> tuple[CalibrationArtifacts, pd.DataFrame]:
    scales = list(scale_columns or DEFAULT_SCALE_COLUMNS)
    items = list(item_columns or DEFAULT_ITEM_COLUMNS)
    resolved_scale_to_items = {
        scale_name: [
            item_id for item_id in (scale_to_items or DEFAULT_SCALE_TO_ITEMS).get(scale_name, [])
            if item_id in items
        ]
        for scale_name in scales
    }
    resolved_scale_shift_aliases = {
        scale_name: (scale_shift_aliases or DEFAULT_SCALE_SHIFT_ALIASES).get(
            scale_name,
            scale_name.removesuffix("_mean"),
        )
        for scale_name in scales
    }
    subgroups = list(subgroup_columns or DEFAULT_SUBGROUP_COLUMNS)

    donor_columns = ["respondent_id", profile_column] + subgroups + scales + items
    donor_columns = [column for column in donor_columns if column in train_profiles.columns]
    donor_pool = train_profiles[donor_columns].copy().reset_index(drop=True)
    if "respondent_id" not in donor_pool.columns:
        donor_pool["respondent_id"] = donor_pool.index.map(lambda idx: f"donor_{idx}")

    global_means = train_profiles[scales].mean()
    global_stds = train_profiles[scales].std(ddof=1).fillna(0.0).replace(0.0, 0.25)
    global_covariance = _ensure_covariance(train_profiles, scales)

    profile_scale_means: dict[int, dict[str, float]] = {}
    profile_scale_covariances: dict[int, dict[str, dict[str, float]]] = {}
    profile_counts: dict[int, int] = {}
    profile_item_priority: dict[int, dict[str, list[str]]] = {}
    profile_subgroup_offsets: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}

    for profile_id, group in train_profiles.groupby(profile_column):
        numeric_profile_id = int(profile_id)
        profile_counts[numeric_profile_id] = int(len(group))
        profile_means = group[scales].mean()
        profile_scale_means[numeric_profile_id] = _series_to_float_dict(profile_means)
        profile_scale_covariances[numeric_profile_id] = _covariance_to_dict(
            _ensure_covariance(group, scales)
        )
        profile_item_priority[numeric_profile_id] = _build_item_priority(
            group,
            items,
            resolved_scale_to_items,
        )

        subgroup_map: dict[str, dict[str, dict[str, Any]]] = {}
        for subgroup_column in subgroups:
            if subgroup_column not in group.columns:
                continue
            offsets_by_value: dict[str, dict[str, Any]] = {}
            for subgroup_value, subgroup_frame in group.groupby(subgroup_column):
                subgroup_means = subgroup_frame[scales].mean()
                offsets = subgroup_means - profile_means
                offsets_by_value[str(subgroup_value)] = {
                    "count": int(len(subgroup_frame)),
                    "offsets": _series_to_float_dict(offsets),
                }
            subgroup_map[subgroup_column] = offsets_by_value
        profile_subgroup_offsets[numeric_profile_id] = subgroup_map

    global_subgroup_offsets: dict[str, dict[str, dict[str, float]]] = {}
    for subgroup_column in subgroups:
        if subgroup_column not in train_profiles.columns:
            continue
        offsets_by_value: dict[str, dict[str, float]] = {}
        for subgroup_value, subgroup_frame in train_profiles.groupby(subgroup_column):
            subgroup_means = subgroup_frame[scales].mean()
            offsets_by_value[str(subgroup_value)] = _series_to_float_dict(
                subgroup_means - global_means
            )
        global_subgroup_offsets[subgroup_column] = offsets_by_value

    artifacts = CalibrationArtifacts(
        scale_columns=scales,
        item_columns=items,
        scale_to_items=resolved_scale_to_items,
        scale_shift_aliases=resolved_scale_shift_aliases,
        profile_column=profile_column,
        subgroup_columns=subgroups,
        donor_pool_min=donor_pool_min,
        profile_shrinkage=profile_shrinkage,
        signal_scale=signal_scale,
        neighbor_count=neighbor_count,
        response_min=int(response_min if response_min is not None else DEFAULT_DATASET.response_min),
        response_max=int(response_max if response_max is not None else DEFAULT_DATASET.response_max),
        global_scale_means=_series_to_float_dict(global_means),
        global_scale_stds=_series_to_float_dict(global_stds),
        global_scale_covariance=_covariance_to_dict(global_covariance),
        profile_scale_means=profile_scale_means,
        profile_scale_covariances=profile_scale_covariances,
        profile_counts=profile_counts,
        global_subgroup_offsets=global_subgroup_offsets,
        profile_subgroup_offsets=profile_subgroup_offsets,
        global_item_priority=_build_item_priority(
            train_profiles,
            items,
            resolved_scale_to_items,
        ),
        profile_item_priority=profile_item_priority,
    )
    return artifacts, donor_pool


def _profile_weight(artifacts: CalibrationArtifacts, profile_id: int) -> float:
    count = artifacts.profile_counts.get(profile_id, 0)
    empirical_weight = count / (count + artifacts.donor_pool_min)
    return max(0.0, min(1.0, empirical_weight * (1.0 - artifacts.profile_shrinkage)))


def _offset_vector(
    artifacts: CalibrationArtifacts,
    row: pd.Series,
    profile_id: int,
) -> np.ndarray:
    offsets: list[np.ndarray] = []
    for subgroup_column in artifacts.subgroup_columns:
        if subgroup_column not in row.index:
            continue
        subgroup_value = row[subgroup_column]
        subgroup_key = str(subgroup_value)
        profile_map = artifacts.profile_subgroup_offsets.get(profile_id, {}).get(
            subgroup_column,
            {},
        )
        profile_payload = profile_map.get(subgroup_key)
        if profile_payload and int(profile_payload["count"]) >= 2:
            offsets.append(
                np.array(
                    [
                        float(profile_payload["offsets"][scale_name])
                        for scale_name in artifacts.scale_columns
                    ],
                    dtype="float64",
                )
            )
            continue
        global_payload = artifacts.global_subgroup_offsets.get(subgroup_column, {}).get(
            subgroup_key
        )
        if global_payload:
            offsets.append(
                np.array(
                    [float(global_payload[scale_name]) for scale_name in artifacts.scale_columns],
                    dtype="float64",
                )
            )
    if not offsets:
        return np.zeros(len(artifacts.scale_columns), dtype="float64")
    return np.mean(offsets, axis=0)


def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
    stabilized = covariance.copy()
    stabilized = (stabilized + stabilized.T) / 2.0
    stabilized += np.eye(stabilized.shape[0]) * 0.05
    return stabilized


def _select_donor(
    donor_pool: pd.DataFrame,
    row: pd.Series,
    target_scales: np.ndarray,
    artifacts: CalibrationArtifacts,
    profile_id: int,
    rng: np.random.Generator,
) -> pd.Series:
    subset = donor_pool[donor_pool[artifacts.profile_column] == profile_id]
    if len(subset) < artifacts.donor_pool_min:
        subset = donor_pool

    global_stds = np.array(
        [artifacts.global_scale_stds[scale_name] for scale_name in artifacts.scale_columns],
        dtype="float64",
    )
    donor_scales = subset[artifacts.scale_columns].to_numpy(dtype="float64")
    distances = (((donor_scales - target_scales) / global_stds) ** 2).sum(axis=1)

    for subgroup_column in artifacts.subgroup_columns:
        if subgroup_column not in subset.columns or subgroup_column not in row.index:
            continue
        distances += 0.15 * (
            subset[subgroup_column].astype(str).to_numpy() != str(row[subgroup_column])
        )

    candidate_count = min(artifacts.neighbor_count, len(subset))
    ranked = np.argsort(distances)[:candidate_count]
    ranked_subset = subset.iloc[ranked].reset_index(drop=True)
    ranked_distances = distances[ranked]
    weights = np.exp(-(ranked_distances - ranked_distances.min()))
    weights = weights / weights.sum()
    selected_index = int(rng.choice(len(ranked_subset), p=weights))
    return ranked_subset.iloc[selected_index]


def _adjust_scale_sum(
    responses: dict[str, int],
    ordered_items: list[str],
    *,
    delta: int,
    direction: int,
    extremity_preference: float,
    rng: np.random.Generator,
    lower_bound: int,
    upper_bound: int,
) -> None:
    remaining = abs(delta)
    while remaining > 0:
        changed = False
        candidate_items = list(ordered_items)
        if extremity_preference < 0.5:
            rng.shuffle(candidate_items)
        for item_id in candidate_items:
            current_value = responses[item_id]
            if direction > 0 and current_value < upper_bound:
                responses[item_id] = current_value + 1
                remaining -= 1
                changed = True
            elif direction < 0 and current_value > lower_bound:
                responses[item_id] = current_value - 1
                remaining -= 1
                changed = True
            if remaining == 0:
                break
            if changed and extremity_preference >= 0.5:
                break
        if not changed:
            break


def _signal_vector(
    signal: CalibrationSignal,
    artifacts: CalibrationArtifacts,
) -> np.ndarray:
    return np.array(
        [
            signal.shift_for_scale(
                scale_name,
                alias=artifacts.scale_shift_aliases.get(scale_name),
            )
            for scale_name in artifacts.scale_columns
        ],
        dtype="float64",
    )


def _fallback_item_value(
    item_id: str,
    *,
    profile_id: int,
    artifacts: CalibrationArtifacts,
) -> int:
    for scale_name, scale_items in artifacts.scale_to_items.items():
        if item_id not in scale_items:
            continue
        mean_map = artifacts.profile_scale_means.get(profile_id, artifacts.global_scale_means)
        fallback_center = (artifacts.response_min + artifacts.response_max) / 2
        scale_value = mean_map.get(
            scale_name,
            artifacts.global_scale_means.get(scale_name, fallback_center),
        )
        return int(
            np.clip(
                round(float(scale_value)),
                artifacts.response_min,
                artifacts.response_max,
            )
        )
    return int(round((artifacts.response_min + artifacts.response_max) / 2))


def neutral_calibration_signal(
    scale_columns: list[str],
    *,
    scale_shift_aliases: dict[str, str] | None = None,
    reasoning_note: str | None = None,
) -> CalibrationSignal:
    aliases = scale_shift_aliases or {
        scale_name: DEFAULT_SCALE_SHIFT_ALIASES.get(scale_name, scale_name.removesuffix("_mean"))
        for scale_name in scale_columns
    }
    return CalibrationSignal(
        scale_shifts={aliases[scale_name]: 0.0 for scale_name in scale_columns},
        extremity_preference=0.5,
        reasoning_note=reasoning_note,
    )


def decode_calibrated_response(
    row: pd.Series,
    signal: CalibrationSignal,
    artifacts: CalibrationArtifacts,
    donor_pool: pd.DataFrame,
    *,
    random_state: int,
) -> CalibratedResponse:
    profile_id = int(row[artifacts.profile_column])
    weight = _profile_weight(artifacts, profile_id)

    global_mean = np.array(
        [artifacts.global_scale_means[scale_name] for scale_name in artifacts.scale_columns],
        dtype="float64",
    )
    global_cov = _dict_to_covariance(artifacts.global_scale_covariance, artifacts.scale_columns)

    profile_mean = np.array(
        [
            artifacts.profile_scale_means.get(profile_id, artifacts.global_scale_means)[scale_name]
            for scale_name in artifacts.scale_columns
        ],
        dtype="float64",
    )
    profile_cov_payload = artifacts.profile_scale_covariances.get(
        profile_id,
        artifacts.global_scale_covariance,
    )
    profile_cov = _dict_to_covariance(profile_cov_payload, artifacts.scale_columns)

    base_mean = weight * profile_mean + (1.0 - weight) * global_mean
    subgroup_offset = _offset_vector(artifacts, row, profile_id)
    base_mean = np.clip(
        base_mean + subgroup_offset,
        artifacts.response_min,
        artifacts.response_max,
    )

    covariance = weight * profile_cov + (1.0 - weight) * global_cov
    global_stds = np.array(
        [artifacts.global_scale_stds[scale_name] for scale_name in artifacts.scale_columns],
        dtype="float64",
    )
    shifted_mean = base_mean + (
        global_stds * artifacts.signal_scale * _signal_vector(signal, artifacts)
    )
    shifted_mean = np.clip(
        shifted_mean,
        artifacts.response_min,
        artifacts.response_max,
    )

    variance_multiplier = 0.85 + (signal.extremity_preference * 0.5)
    covariance = _stabilize_covariance(covariance * variance_multiplier)

    rng = np.random.default_rng(random_state)
    target_scales = rng.multivariate_normal(shifted_mean, covariance)
    target_scales = np.clip(
        target_scales,
        artifacts.response_min,
        artifacts.response_max,
    )

    donor = _select_donor(donor_pool, row, target_scales, artifacts, profile_id, rng)
    responses: dict[str, int] = {}
    for item_id in artifacts.item_columns:
        if item_id not in donor.index:
            responses[item_id] = _fallback_item_value(
                item_id,
                profile_id=profile_id,
                artifacts=artifacts,
            )
            continue
        donor_value = donor[item_id]
        if pd.isna(donor_value):
            responses[item_id] = _fallback_item_value(
                item_id,
                profile_id=profile_id,
                artifacts=artifacts,
            )
            continue
        responses[item_id] = int(donor_value)

    for scale_name, scale_items in artifacts.scale_to_items.items():
        present_items = [item_id for item_id in scale_items if item_id in responses]
        if not present_items or scale_name not in artifacts.scale_columns:
            continue
        scale_index = artifacts.scale_columns.index(scale_name)
        target_sum = int(round(float(target_scales[scale_index]) * len(present_items)))
        current_sum = sum(responses[item_id] for item_id in present_items)
        delta = target_sum - current_sum
        if delta == 0:
            continue
        descending_priority = (
            artifacts.profile_item_priority.get(profile_id, {}).get(scale_name)
            or artifacts.global_item_priority.get(scale_name)
            or present_items
        )
        ordered_items = [
            item_id for item_id in descending_priority if item_id in present_items
        ]
        if delta > 0:
            _adjust_scale_sum(
                responses,
                ordered_items,
                delta=delta,
                direction=1,
                extremity_preference=signal.extremity_preference,
                rng=rng,
                lower_bound=artifacts.response_min,
                upper_bound=artifacts.response_max,
            )
        else:
            _adjust_scale_sum(
                responses,
                list(reversed(ordered_items)),
                delta=delta,
                direction=-1,
                extremity_preference=signal.extremity_preference,
                rng=rng,
                lower_bound=artifacts.response_min,
                upper_bound=artifacts.response_max,
            )

    realized_scale_values = {
        scale_name: float(
            np.mean(
                [responses[item_id] for item_id in artifacts.scale_to_items[scale_name] if item_id in responses]
            )
        )
        for scale_name in artifacts.scale_columns
    }
    target_scale_values = {
        scale_name: float(target_scales[index])
        for index, scale_name in enumerate(artifacts.scale_columns)
    }
    base_scale_values = {
        scale_name: float(base_mean[index])
        for index, scale_name in enumerate(artifacts.scale_columns)
    }

    return CalibratedResponse(
        item_responses=responses,
        target_scale_values=target_scale_values,
        realized_scale_values=realized_scale_values,
        base_scale_values=base_scale_values,
        selected_donor_id=str(donor.get("respondent_id")) if "respondent_id" in donor else None,
    )


def write_calibration_artifacts(
    run_dir: Path,
    artifacts: CalibrationArtifacts,
    donor_pool: pd.DataFrame,
) -> None:
    calibration_dir = run_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "calibration_artifacts.json").write_text(
        json.dumps(artifacts.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    donor_pool.to_csv(calibration_dir / "donor_pool.csv", index=False)


def load_calibration_artifacts(run_dir: Path) -> tuple[CalibrationArtifacts, pd.DataFrame]:
    calibration_dir = run_dir / "calibration"
    artifacts = CalibrationArtifacts.from_dict(
        json.loads((calibration_dir / "calibration_artifacts.json").read_text(encoding="utf-8"))
    )
    donor_pool = pd.read_csv(calibration_dir / "donor_pool.csv")
    return artifacts, donor_pool
