from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FeatureDescriptorSettings(BaseModel):
    label: str
    high_text: str
    low_text: str


class MetadataFieldSettings(BaseModel):
    name: str
    source_column: str
    label: str | None = None
    transform: Literal[
        "text",
        "mapped_text",
        "mapped_number",
        "age_band_default",
        "program_tr",
        "float",
    ] = "text"
    value_map: dict[str, str | float] = Field(default_factory=dict)
    include_in_context: bool = True

    def normalized_value_map(self) -> dict[str, str | float]:
        return {str(key).strip().casefold(): value for key, value in self.value_map.items()}


class ScaleSettings(BaseModel):
    name: str
    mean_column: str
    raw_columns: list[str]
    item_columns: list[str]
    response_map: dict[str, float]
    item_response_maps: dict[str, dict[str, float]] = Field(default_factory=dict)
    item_transforms: dict[str, str] = Field(default_factory=dict)
    reverse_item_columns: list[str] = Field(default_factory=list)
    label: str | None = None
    shift_key: str | None = None

    @model_validator(mode="after")
    def _validate_columns(self) -> ScaleSettings:
        if len(self.raw_columns) != len(self.item_columns):
            raise ValueError("raw_columns and item_columns must have the same length.")
        if not self.response_map:
            raise ValueError("response_map cannot be empty.")
        return self

    def normalized_response_map(self) -> dict[str, float]:
        return {
            str(key).strip().casefold(): float(value)
            for key, value in self.response_map.items()
        }

    def normalized_item_response_maps(self) -> dict[str, dict[str, float]]:
        return {
            str(item_column): {
                str(key).strip().casefold(): float(value)
                for key, value in response_map.items()
            }
            for item_column, response_map in self.item_response_maps.items()
        }

    @property
    def effective_shift_key(self) -> str:
        if self.shift_key:
            return self.shift_key
        if self.mean_column.endswith("_mean"):
            return self.mean_column.removesuffix("_mean")
        return self.mean_column

    def response_map_for_item(self, item_column: str) -> dict[str, float]:
        specific = self.normalized_item_response_maps().get(item_column)
        if specific is not None:
            return specific
        return self.normalized_response_map()

    def transform_for_item(self, item_column: str) -> str | None:
        return self.item_transforms.get(item_column)


class RegressionPairSettings(BaseModel):
    name: str
    x: str
    y: str


class AuditReferenceSettings(BaseModel):
    paper_sample_size: int | None = None
    expected_category_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    note: str | None = None


class DatasetSettings(BaseModel):
    dataset_id: str
    locale: Literal["tr", "en"] = "tr"
    file_format: Literal["auto", "xlsx", "csv"] = "auto"
    sheet_name: str | None = None
    metadata_fields: list[MetadataFieldSettings] = Field(default_factory=list)
    scales: list[ScaleSettings] = Field(default_factory=list)
    feature_descriptors: dict[str, FeatureDescriptorSettings] = Field(default_factory=dict)
    profile_anchor_columns: list[str] = Field(default_factory=list)
    subgroup_columns: list[str] = Field(default_factory=list)
    stratify_categorical_columns: list[str] = Field(default_factory=list)
    stratify_score_columns: list[str] = Field(default_factory=list)
    regression_pairs: list[RegressionPairSettings] = Field(default_factory=list)
    response_min: int = 1
    response_max: int = 5
    audit_reference: AuditReferenceSettings | None = None

    @model_validator(mode="after")
    def _validate_response_bounds(self) -> DatasetSettings:
        if self.response_min > self.response_max:
            raise ValueError("response_min cannot exceed response_max.")
        if not self.scales:
            raise ValueError("At least one scale definition is required.")
        for scale in self.scales:
            values = list(scale.response_map.values())
            if min(values) < self.response_min or max(values) > self.response_max:
                raise ValueError(
                    f"Scale '{scale.name}' response_map must stay within "
                    f"[{self.response_min}, {self.response_max}]."
                )
        return self

    @property
    def item_columns(self) -> list[str]:
        return [item for scale in self.scales for item in scale.item_columns]

    @property
    def scale_columns(self) -> list[str]:
        return [scale.mean_column for scale in self.scales]

    @property
    def scale_to_items(self) -> dict[str, list[str]]:
        return {scale.mean_column: list(scale.item_columns) for scale in self.scales}

    @property
    def question_spec(self) -> list[tuple[str, str]]:
        return [
            (item_id, raw_column)
            for scale in self.scales
            for item_id, raw_column in zip(scale.item_columns, scale.raw_columns, strict=True)
        ]

    @property
    def context_columns(self) -> list[str]:
        return [
            field.name for field in self.metadata_fields if field.include_in_context
        ]

    @property
    def context_labels(self) -> dict[str, str]:
        return {
            field.name: field.label or field.name
            for field in self.metadata_fields
            if field.include_in_context
        }

    def scale_for_mean_column(self, mean_column: str) -> ScaleSettings:
        for scale in self.scales:
            if scale.mean_column == mean_column:
                return scale
        raise KeyError(f"Unknown scale mean column: {mean_column}")

    def metadata_field(self, name: str) -> MetadataFieldSettings:
        for field in self.metadata_fields:
            if field.name == name:
                return field
        raise KeyError(f"Unknown metadata field: {name}")

    def shift_field_name(self, mean_column: str) -> str:
        return f"{self.scale_for_mean_column(mean_column).effective_shift_key}_shift"

    def scale_column_for_shift_field(self, shift_field_name: str) -> str:
        for scale in self.scales:
            if self.shift_field_name(scale.mean_column) == shift_field_name:
                return scale.mean_column
        raise KeyError(f"Unknown calibration shift field: {shift_field_name}")

    def feature_label(self, column: str) -> str:
        descriptor = self.feature_descriptors.get(column)
        if descriptor is not None:
            return descriptor.label
        for scale in self.scales:
            if scale.mean_column == column:
                return scale.label or scale.name
        for field in self.metadata_fields:
            if field.name == column:
                return field.label or field.name
        return column

