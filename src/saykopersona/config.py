from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .dataset_catalog import default_dataset_settings, dataset_settings_for_preset
from .dataset_spec import DatasetSettings
from .data import get_default_dataset_path, get_project_root, locate_dataset
from .lmstudio import InferencePreset


class SplitSettings(BaseModel):
    validation_size: float = 0.1
    final_test_size: float = 0.1
    random_state: int = 42

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_test_size(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "test_size" not in data:
            return data
        if "validation_size" in data or "final_test_size" in data:
            return data
        upgraded = dict(data)
        test_size = upgraded.pop("test_size")
        if test_size is not None:
            half = float(test_size) / 2
            upgraded["validation_size"] = half
            upgraded["final_test_size"] = half
        return upgraded

    @model_validator(mode="after")
    def _validate_split_sizes(self) -> SplitSettings:
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        if not 0 < self.final_test_size < 1:
            raise ValueError("final_test_size must be between 0 and 1.")
        if self.validation_size + self.final_test_size >= 1:
            raise ValueError(
                "validation_size + final_test_size must be less than 1."
            )
        return self


class ProfileSettings(BaseModel):
    """Empirical-profile fitting settings.

    ``feature_columns`` deliberately excludes ``usage_hours_midpoint``. That variable is
    derived from a five-level usage band and takes only four distinct values, so a
    full-covariance Gaussian mixture collapses its variance toward zero and partitions on
    it alone: with it included the Turkish profile label was a one-to-one relabeling of
    the usage band (eta^2 = 1.000) and the information criteria went negative for every
    candidate profile count from four upward. Fitting on scale means alone keeps the
    criteria well behaved and yields profiles that separate on the psychometric scales.
    Usage remains available to the prompt as respondent context and as a profile anchor;
    it is simply not a mixture feature.
    """

    cluster_min: int = 2
    cluster_max: int = 6
    feature_columns: list[str] = Field(
        default_factory=lambda: [
            "digital_addiction_mean",
            "social_anxiety_mean",
            "wellbeing_mean",
        ]
    )
    method: Literal["gmm", "kmeans"] = "gmm"
    """Clustering method for the profile layer.

    ``gmm`` is the benchmark's full-covariance Gaussian mixture. ``kmeans`` partitions
    the same standardised scale means with k-means and exists to test whether generation
    depends on the clustering method. k-means has no likelihood, so when a range of
    profile counts is given the count is chosen by silhouette alone.
    """
    membership: Literal["hard", "soft"] = "hard"
    """Whether a prompt receives one profile or the membership probabilities.

    ``soft`` passes the cold-start classifier's probabilities for every profile to the
    prompt. It is allowed only with ``assignment: coldstart``. Under oracle assignment
    the probabilities would be a finer function of the held-out respondent's own
    answers than a hard label, which would widen the information path that the oracle
    setting already opens.
    """
    assignment: Literal["oracle", "coldstart"] = "oracle"
    """How held-out respondents are assigned to a training profile.

    ``oracle`` matches a respondent's own scale means against the mixture centroids.
    Those means come from the answers the simulation is asked to produce, so the
    assignment is profile-known and the evaluation measures performance given correct
    profile membership.

    ``coldstart`` predicts membership from the pre-response context fields alone, using
    a classifier fitted on the training split inside the same run. This is the
    deployable setting, since nothing derived from a held-out respondent's answers
    enters the assignment.

    The mixture itself is fitted on scale means under both settings. Only the routing of
    held-out respondents differs, which is what determines the persona card and the
    profile statistics a prompt receives.
    """

    @model_validator(mode="after")
    def _soft_membership_needs_coldstart(self) -> "ProfileSettings":
        if self.membership == "soft" and self.assignment != "coldstart":
            raise ValueError("profiles.membership='soft' requires profiles.assignment='coldstart'")
        return self


class PresetSettings(BaseModel):
    name: str
    temperature: float
    max_tokens: int
    top_p: float = 1.0

    def to_runtime(self) -> InferencePreset:
        return InferencePreset(
            name=self.name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )


def _load_project_dotenv() -> dict[str, str]:
    env_path = get_project_root() / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


class LMStudioSettings(BaseModel):
    base_url: str = "http://127.0.0.1:1234/v1"
    model_id: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    # OpenRouter service tier ("flex" or "priority"). Unset sends no tier and leaves
    # routing at the provider default. Flex serves the same model snapshot at lower
    # priority and lower price; the tier that actually served each call is returned
    # in the response and kept in the per-run request archive.
    service_tier: Literal["flex", "priority", "default"] | None = None
    # OpenRouter `reasoning` object, sent as given, for example {"enabled": false}.
    # Unset keeps the built-in behaviour: minimal reasoning for Gemini 3 and no
    # reasoning field for any other model. Open-weight reasoning models need this set,
    # because hidden reasoning otherwise consumes the whole answer budget.
    reasoning: dict[str, Any] | None = None
    # OpenRouter `provider` routing object, sent as given, for example
    # {"only": ["deepinfra"], "quantizations": ["bf16"], "allow_fallbacks": false}.
    # Open-weight models are served by many hosts at different numeric precision, so a
    # run is only reproducible when the host and precision are pinned.
    provider: dict[str, Any] | None = None
    primary: PresetSettings = Field(
        default_factory=lambda: PresetSettings(
            name="primary",
            temperature=0.15,
            max_tokens=600,
            top_p=1.0,
        )
    )

    def resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            env_value = os.environ.get(self.api_key_env)
            if env_value:
                return env_value
            return _load_project_dotenv().get(self.api_key_env)
        return None

    def build_headers(self) -> dict[str, str]:
        resolved_headers = dict(self.headers)
        api_key = self.resolve_api_key()
        if api_key:
            resolved_headers.setdefault("Authorization", f"Bearer {api_key}")
        return resolved_headers
    persona: PresetSettings = Field(
        default_factory=lambda: PresetSettings(
            name="persona_card",
            temperature=0.2,
            max_tokens=1100,
            top_p=1.0,
        )
    )
    stability: PresetSettings = Field(
        default_factory=lambda: PresetSettings(
            name="stability",
            temperature=0.2,
            max_tokens=600,
            top_p=1.0,
        )
    )
    ordinal: PresetSettings = Field(
        default_factory=lambda: PresetSettings(
            name="ordinal",
            temperature=0.0,
            max_tokens=1,
            top_p=1.0,
        )
    )


class SimulationSettings(BaseModel):
    prompt_version: str = "v2_tr"
    response_mode: Literal["direct_json", "latent_logprob"] = "direct_json"
    item_scoring_mode: Literal["argmax", "posterior_mean_round"] = "argmax"
    trait_plan_samples: int = Field(default=1, ge=1)
    max_holdout_rows: int | None = None
    stability_runs: int = 3
    parallel_workers: int = Field(default=4, ge=1)
    self_consistency_samples: int = Field(default=1, ge=1)
    evaluation_split: Literal["validation", "final_test"] = "validation"


class CalibrationSettings(BaseModel):
    enabled: bool = False
    strategy: Literal["direct", "statistical_only", "hybrid_llm"] = "direct"
    donor_pool_min: int = Field(default=20, ge=1)
    profile_shrinkage: float = Field(default=0.25, ge=0.0, le=1.0)
    signal_scale: float = Field(default=0.6, ge=0.0, le=2.0)
    neighbor_count: int = Field(default=8, ge=1)
    save_raw_outputs: bool = True


class ResearchConfig(BaseModel):
    dataset_path: Path = Field(default_factory=get_default_dataset_path)
    dataset: DatasetSettings = Field(default_factory=default_dataset_settings)
    paper_path: Path | None = Field(
        default_factory=lambda: get_project_root()
        / "prior_study_extract.pdf"
    )
    output_root: Path = Field(default_factory=lambda: get_project_root() / "results")
    split: SplitSettings = Field(default_factory=SplitSettings)
    profiles: ProfileSettings = Field(default_factory=ProfileSettings)
    lmstudio: LMStudioSettings = Field(default_factory=LMStudioSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)


def default_config_path() -> Path:
    return get_project_root() / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_relative_paths(payload: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    resolved = dict(payload)
    for key in ["dataset_path", "paper_path", "output_root"]:
        value = resolved.get(key)
        if value in (None, ""):
            continue
        path = Path(value)
        if not path.is_absolute():
            resolved[key] = str((config_dir / path).resolve())
    return resolved


def _resolve_dataset_payload(payload: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    dataset_value = resolved.get("dataset")
    if dataset_value is None:
        return resolved
    if isinstance(dataset_value, str):
        resolved["dataset"] = dataset_settings_for_preset(dataset_value).model_dump(mode="json")
        return resolved
    if isinstance(dataset_value, dict) and "preset" in dataset_value:
        preset_name = dataset_value["preset"]
        preset_payload = dataset_settings_for_preset(str(preset_name)).model_dump(mode="json")
        overrides = {key: value for key, value in dataset_value.items() if key != "preset"}
        resolved["dataset"] = _deep_merge(preset_payload, overrides)
        return resolved
    return resolved


def load_config(path: Path | str | None = None) -> ResearchConfig:
    config_path = Path(path) if path is not None else default_config_path()
    base_payload = ResearchConfig().model_dump(mode="json")
    if config_path.exists():
        override_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        override_payload = _resolve_dataset_payload(override_payload)
        payload = _deep_merge(base_payload, override_payload)
        payload = _resolve_relative_paths(payload, config_path.parent)
        payload = _resolve_dataset_payload(payload)
        if payload.get("dataset_path"):
            payload["dataset_path"] = str(locate_dataset(Path(payload["dataset_path"])))
    else:
        payload = base_payload
    return ResearchConfig.model_validate(payload)
