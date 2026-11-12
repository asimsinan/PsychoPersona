from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)
_NUMERIC_BOUND_KEYS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
}


@dataclass(slots=True)
class InferencePreset:
    name: str
    temperature: float
    max_tokens: int
    top_p: float = 1.0


@dataclass(slots=True)
class ModelInfo:
    id: str
    object: str = "model"


@dataclass(slots=True)
class RunMetadata:
    model_id: str
    preset_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PersonaCard(BaseModel):
    profile_id: int
    label: str
    summary: str
    demographic_anchor: str
    behavior_markers: list[str]
    tone: str


class SurveySimulationResponse(BaseModel):
    responses: dict[str, int | float | str] = Field(default_factory=dict)
    reasoning_note: str | None = None


def _normalize_structured_output_schema(fragment: Any) -> Any:
    if isinstance(fragment, dict):
        normalized = {
            key: _normalize_structured_output_schema(value)
            for key, value in fragment.items()
        }
        if normalized.get("type") == "object" or "properties" in normalized:
            normalized["additionalProperties"] = False
            properties = normalized.get("properties")
            if isinstance(properties, dict):
                normalized["required"] = list(properties.keys())
        return normalized
    if isinstance(fragment, list):
        return [_normalize_structured_output_schema(item) for item in fragment]
    return fragment


def _strip_unsupported_numeric_bounds(fragment: Any) -> Any:
    if isinstance(fragment, dict):
        normalized = {
            key: _strip_unsupported_numeric_bounds(value)
            for key, value in fragment.items()
        }
        if normalized.get("type") in {"integer", "number"}:
            for key in _NUMERIC_BOUND_KEYS:
                normalized.pop(key, None)
        return normalized
    if isinstance(fragment, list):
        return [_strip_unsupported_numeric_bounds(item) for item in fragment]
    return fragment


def _normalize_choice_token(token: str, valid_choices: set[str]) -> str | None:
    normalized = token.strip()
    if normalized in valid_choices:
        return normalized
    return None


def _logsumexp(log_values: list[float]) -> float:
    if not log_values:
        return float("-inf")
    pivot = max(log_values)
    return pivot + math.log(sum(math.exp(value - pivot) for value in log_values))


def _extract_choice_probabilities(
    response_payload: dict[str, Any],
    *,
    choices: list[str],
) -> tuple[dict[str, float], str | None]:
    valid_choices = set(choices)
    choice_payload = response_payload["choices"][0]
    raw_choice = _normalize_choice_token(
        str(choice_payload.get("message", {}).get("content", "")),
        valid_choices,
    )
    logprob_content = choice_payload.get("logprobs", {}).get("content", [])
    if not logprob_content:
        return {}, raw_choice

    token_entry = logprob_content[0]
    top_logprobs = token_entry.get("top_logprobs") or []
    grouped_logprobs: dict[str, list[float]] = {choice: [] for choice in choices}

    for candidate in top_logprobs:
        normalized_choice = _normalize_choice_token(
            str(candidate.get("token", "")),
            valid_choices,
        )
        if normalized_choice is None:
            continue
        logprob = candidate.get("logprob")
        if logprob is None:
            continue
        grouped_logprobs[normalized_choice].append(float(logprob))

    combined_logprobs = {
        choice: _logsumexp(values)
        for choice, values in grouped_logprobs.items()
        if values
    }
    if not combined_logprobs:
        return {}, raw_choice

    normalizer = _logsumexp(list(combined_logprobs.values()))
    probabilities = {
        choice: math.exp(logprob - normalizer)
        for choice, logprob in combined_logprobs.items()
    }
    for choice in choices:
        probabilities.setdefault(choice, 0.0)
    return probabilities, raw_choice


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        archive_dir: Path | None = None,
        headers: dict[str, str] | None = None,
        default_model_id: str | None = None,
        service_tier: str | None = None,
        reasoning: dict[str, Any] | None = None,
        provider: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._service_tier = service_tier
        self._reasoning = reasoning
        self._provider = provider
        request_headers = {"Content-Type": "application/json"}
        if headers is not None:
            request_headers.update(headers)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=request_headers,
            timeout=timeout,
            transport=transport,
        )
        self._cached_model: ModelInfo | None = None
        self._default_model_id = default_model_id
        self._archive_dir = archive_dir
        self._archive_sequence = 0
        self._archive_lock = threading.Lock()
        if self._archive_dir is not None:
            self._archive_dir.mkdir(parents=True, exist_ok=True)
            existing_sequences: list[int] = []
            for path in self._archive_dir.glob("*.json"):
                prefix = path.stem.split("_", 1)[0]
                if prefix.isdigit():
                    existing_sequences.append(int(prefix))
            self._archive_sequence = max(existing_sequences, default=0)

    def close(self) -> None:
        self._client.close()

    def _discrete_choice_max_tokens(self, preset: InferencePreset) -> int:
        max_tokens = max(1, int(preset.max_tokens))
        host = self._host_name
        if host == "openrouter.ai":
            return max(max_tokens, 16)
        return max_tokens

    @property
    def _host_name(self) -> str:
        return (self._client.base_url.host or "").lower()

    def _structured_schema_payload(
        self,
        *,
        schema: type[StructuredModel],
        model_id: str,
    ) -> dict[str, Any]:
        normalized_schema = _normalize_structured_output_schema(
            schema.model_json_schema()
        )
        if self._host_name == "openrouter.ai" and model_id.startswith("anthropic/"):
            return _strip_unsupported_numeric_bounds(normalized_schema)
        return normalized_schema

    def _openrouter_reasoning_payload(self, *, model_id: str) -> dict[str, Any]:
        if self._host_name != "openrouter.ai":
            return {}
        if self._reasoning is not None:
            # An explicit setting from the run config replaces the per-model default.
            return {"reasoning": dict(self._reasoning)}
        if model_id.startswith("google/gemini-3"):
            # Gemini 3 models can emit long reasoning traces by default on OpenRouter.
            # For structured JSON and single-token choice tasks, keep reasoning minimal
            # and exclude it from the response so content budgets are spent on outputs.
            return {"reasoning": {"effort": "minimal", "exclude": True}}
        return {}

    def _openrouter_routing_payload(self) -> dict[str, Any]:
        if self._host_name != "openrouter.ai":
            return {}
        payload: dict[str, Any] = {}
        if self._service_tier:
            payload["service_tier"] = self._service_tier
        if self._provider:
            payload["provider"] = dict(self._provider)
        return payload

    def _structured_retry_max_tokens(self, *, preset: InferencePreset, attempt: int) -> int:
        if attempt == 0:
            return preset.max_tokens
        return preset.max_tokens + (512 * attempt)

    def _should_retry_structured_parse(
        self,
        *,
        model_id: str,
        response_payload: dict[str, Any],
        validation_error: ValidationError | None = None,
    ) -> bool:
        if self._host_name != "openrouter.ai":
            return False
        choice = response_payload.get("choices", [{}])[0]
        finish_reason = str(choice.get("finish_reason", "")).lower()
        native_finish_reason = str(choice.get("native_finish_reason", "")).lower()
        if finish_reason == "length" or native_finish_reason == "max_tokens":
            return True
        if validation_error is None:
            return False
        return any(issue.get("type") == "json_invalid" for issue in validation_error.errors())

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_fixed(1),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def list_models(self) -> list[ModelInfo]:
        payload = self._request_json("GET", "models")
        return [
            ModelInfo(
                id=item["id"],
                object=item.get("object", "model"),
            )
            for item in payload.get("data", [])
            if "id" in item
        ]

    def discover_loaded_model(self) -> ModelInfo:
        if self._cached_model is not None:
            return self._cached_model

        if self._default_model_id is not None:
            self._cached_model = ModelInfo(id=self._default_model_id)
            return self._cached_model

        models = self.list_models()
        if not models:
            raise RuntimeError("LM Studio returned no loaded models.")
        self._cached_model = models[0]
        return self._cached_model

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_fixed(1),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[StructuredModel],
        preset: InferencePreset,
        model_id: str | None = None,
    ) -> tuple[StructuredModel, RunMetadata]:
        selected_model = model_id or self.discover_loaded_model().id
        last_error: ValidationError | None = None

        for attempt in range(3):
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": preset.temperature,
                "top_p": preset.top_p,
                "max_tokens": self._structured_retry_max_tokens(
                    preset=preset,
                    attempt=attempt,
                ),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": self._structured_schema_payload(
                            schema=schema,
                            model_id=selected_model,
                        ),
                    },
                },
            }
            payload.update(self._openrouter_reasoning_payload(model_id=selected_model))
            payload.update(self._openrouter_routing_payload())
            response_payload = self._request_json(
                "POST",
                "chat/completions",
                json_payload=payload,
            )
            content = response_payload["choices"][0]["message"].get("content")
            if not isinstance(content, str):
                content = ""
            try:
                parsed = schema.model_validate_json(content)
            except ValidationError as error:
                last_error = error
                if not self._should_retry_structured_parse(
                    model_id=selected_model,
                    response_payload=response_payload,
                    validation_error=error,
                ):
                    raise
                continue

            usage = response_payload.get("usage", {})
            metadata = RunMetadata(
                model_id=selected_model,
                preset_name=preset.name,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                response_id=response_payload.get("id"),
            )
            return parsed, metadata

        if last_error is not None:
            raise last_error
        raise RuntimeError("Structured generation failed without a parseable response.")

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_fixed(1),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def score_discrete_choices(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        choices: list[str],
        preset: InferencePreset,
        model_id: str | None = None,
        top_logprobs: int = 20,
    ) -> tuple[dict[str, float], RunMetadata, str]:
        selected_model = model_id or self.discover_loaded_model().id
        prompt_variants = [
            user_prompt,
            f"{user_prompt}\n\nReturn exactly one of these choices: {', '.join(choices)}.",
        ]
        response_payload: dict[str, Any] | None = None
        probabilities: dict[str, float] = {}
        raw_choice: str | None = None

        for prompt_variant in prompt_variants:
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_variant},
                ],
                "temperature": preset.temperature,
                "top_p": preset.top_p,
                "max_tokens": self._discrete_choice_max_tokens(preset),
                "logprobs": True,
                "top_logprobs": top_logprobs,
            }
            payload.update(self._openrouter_reasoning_payload(model_id=selected_model))
            payload.update(self._openrouter_routing_payload())
            response_payload = self._request_json("POST", "chat/completions", json_payload=payload)
            probabilities, raw_choice = _extract_choice_probabilities(
                response_payload,
                choices=choices,
            )
            visible_choices = [choice for choice, prob in probabilities.items() if prob > 0.0]
            if len(visible_choices) >= 2:
                break

        if response_payload is None:
            raise RuntimeError("Discrete choice scoring did not produce a response payload.")
        visible_choices = [choice for choice, prob in probabilities.items() if prob > 0.0]
        if not visible_choices and raw_choice in choices:
            probabilities = {
                choice: 1.0 if choice == raw_choice else 0.0 for choice in choices
            }
            visible_choices = [raw_choice]
        if not visible_choices:
            raise RuntimeError(
                "Discrete choice scoring could not recover enough valid ordinal token probabilities."
            )

        usage = response_payload.get("usage", {})
        chosen = raw_choice or max(probabilities, key=probabilities.__getitem__)
        metadata = RunMetadata(
            model_id=selected_model,
            preset_name=preset.name,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            response_id=response_payload.get("id"),
        )
        return probabilities, metadata, chosen

    def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, endpoint, json=json_payload)
        except httpx.HTTPError as exc:
            self._archive_event(
                method=method,
                endpoint=endpoint,
                request_payload=json_payload,
                response_status=None,
                response_payload=None,
                error=str(exc),
            )
            raise

        try:
            response_payload = response.json()
        except ValueError:
            response_payload = {"raw_text": response.text}

        self._archive_event(
            method=method,
            endpoint=endpoint,
            request_payload=json_payload,
            response_status=response.status_code,
            response_payload=response_payload,
            error=None,
        )
        response.raise_for_status()
        return response_payload

    def _archive_event(
        self,
        *,
        method: str,
        endpoint: str,
        request_payload: dict[str, Any] | None,
        response_status: int | None,
        response_payload: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        if self._archive_dir is None:
            return

        with self._archive_lock:
            self._archive_sequence += 1
            slug = endpoint.strip("/").replace("/", "_") or "root"
            path = self._archive_dir / f"{self._archive_sequence:04d}_{slug}.json"
            archive_payload = {
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "method": method,
                "endpoint": f"/v1/{endpoint.lstrip('/')}",
                "request": request_payload,
                "response_status": response_status,
                "response": response_payload,
                "error": error,
            }
            path.write_text(
                json.dumps(archive_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
