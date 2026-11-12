"""Persona-card construction and survey-response simulation.

Manuscript arm labels map to ``prompt_version`` values as follows. The
abbreviations are the ones used in the paper's simulation-arm table and in the
prompt-structure figures; the figure named for each arm depicts exactly the
inputs and instruction set built here.

===========================================  =========================================
Arm                                          Selector
===========================================  =========================================
DOP   Demographic-Only Prompting             ``demographic_only_v1``
BPP   Balanced Profile Prompting             ``v3_tr_gpt_b``
FPNP  First-Person Narrative Prompting       ``v3_tr_gpt_d``
SCP   Self-Consistency Prompting             ``self_consistency_samples > 1`` (config,
                                             not a prompt version)
BBP   Bias-Balanced Prompting                ``v3_tr_gpt_e`` (GPT-5.4, Gemini 3.1 Pro)
                                             ``v3_tr_claude_a`` (Claude Opus 4.6)
SADP  Statistics-Aware Direct Prompting      ``direct_stats_v1``
TPSAP Two-Pass Statistics-Aware Prompting    ``direct_stats_two_pass_v1``
HC    Hybrid Calibration                     ``v4_tr_hybrid_b`` with
                                             ``calibration.enabled: true``
===========================================  =========================================

BBP is split across two selectors because it applies a model-specific response-band
correction. ``v3_tr_claude_a`` carries the same instruction lines as ``v3_tr_gpt_b``.

Remaining versions are development-only and are not reported: ``v3_tr_gpt_c``
(explicit low/mid/high scale anchors), ``v3_tr_gpt_g`` (repetition-wording tweak),
and the legacy ``v1``, ``v1_compact``, ``v2``, ``v2_tr``, ``v3_tr_gpt``,
``v4_tr_hybrid``.

Note on persona-card serialization: every persona-conditioned arm renders the persona
card and respondent context as native-language prose. ``v3_tr_gpt_b`` (BPP) was
originally absent from that branch and received both blocks as JSON dumps instead,
which made BPP differ in prompt format from the arms it is compared against. It now
renders like the others.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from functools import lru_cache
from typing import Any, Iterable, cast

import pandas as pd
from pydantic import BaseModel, Field, create_model

from .calibration import (
    CalibrationArtifacts,
    CalibrationSignal,
    build_calibration_signal_schema,
    decode_calibrated_response,
    neutral_calibration_signal,
)
from .data import (
    DEFAULT_DATASET,
    DIGITAL_RAW_COLUMNS,
    DIGITAL_SCORE_COLUMNS,
    SOCIAL_RAW_COLUMNS,
    SOCIAL_SCORE_COLUMNS,
    WELLBEING_RAW_COLUMNS,
    WELLBEING_SCORE_COLUMNS,
)
from .coldstart import MEMBERSHIP_PREFIX
from .dataset_spec import DatasetSettings
from .lmstudio import (
    InferencePreset,
    LMStudioClient,
    PersonaCard,
    RunMetadata,
)
from .ordinal_scoring import (
    LatentTraitPlan,
    build_tune_f_plan_prompt,
    render_latent_trait_plan,
    score_item_with_latent_plan,
    select_self_consistent_plan_index,
)
from .progress import make_progress


QUESTION_SPEC = list(
    zip(
        DIGITAL_SCORE_COLUMNS + SOCIAL_SCORE_COLUMNS + WELLBEING_SCORE_COLUMNS,
        DIGITAL_RAW_COLUMNS + SOCIAL_RAW_COLUMNS + WELLBEING_RAW_COLUMNS,
        strict=True,
    )
)

RESPONDENT_CONTEXT_COLUMNS = [
    *DEFAULT_DATASET.context_columns,
]

RESPONDENT_CONTEXT_LABELS_TR = dict(DEFAULT_DATASET.context_labels)
FEATURE_LABELS_TR = {
    column: DEFAULT_DATASET.feature_label(column)
    for column in DEFAULT_DATASET.profile_anchor_columns
    + DEFAULT_DATASET.scale_columns
    + ["usage_hours_midpoint"]
}


def _persona_system_prompt(dataset: DatasetSettings) -> str:
    if dataset.locale == "tr":
        return (
            "Sen, istatistiksel öğrenci profillerini kısa ve veri temelli persona kartlarına "
            "dönüştüren dikkatli bir araştırma asistanısın. JSON içindeki tüm doğal dil "
            "alanlarını Türkçe yaz. Yalnızca JSON döndür ve profil özetinde desteklenmeyen "
            "hiçbir bilgiyi uydurma."
        )
    return (
        "You are a careful research assistant who converts statistical respondent profiles into "
        "short, evidence-grounded persona cards. Write all natural-language JSON fields in "
        "English. Return JSON only and do not invent unsupported details."
    )


def _survey_system_prompt(dataset: DatasetSettings, prompt_version: str = "v1") -> str:
    if prompt_version == "demographic_only_v1":
        if dataset.locale == "tr":
            return (
                "Sen, doğrulanmış Likert tipi bir anketi yanıtlayan katılımcıyı simüle "
                "ediyorsun. Yalnızca verilen katılımcı bağlamı ile tutarlı kal. Yalnızca JSON "
                f"döndür. Her yanıt değeri {dataset.response_min} ile {dataset.response_max} "
                "arasında tam sayı olmalıdır."
            )
        return (
            "You are simulating a respondent answering a validated Likert-style questionnaire. "
            "Stay consistent only with the given respondent context. Return JSON only. Every "
            f"response must be an integer between {dataset.response_min} and "
            f"{dataset.response_max}."
        )
    if dataset.locale == "tr":
        return (
            "Sen, doğrulanmış Likert tipi bir anketi yanıtlayan katılımcıyı simüle ediyorsun. "
            "Persona kartı ve katılımcı bağlamı ile tutarlı kal. Yalnızca JSON döndür. Her "
            f"yanıt değeri {dataset.response_min} ile {dataset.response_max} arasında tam sayı "
            "olmalıdır."
        )
    return (
        "You are simulating a respondent answering a validated Likert-style questionnaire. Stay "
        "consistent with the persona card and respondent context. Return JSON only. Every "
        f"response must be an integer between {dataset.response_min} and {dataset.response_max}."
    )


def _calibration_signal_system_prompt(dataset: DatasetSettings) -> str:
    if dataset.locale == "tr":
        return (
            "Sen, persona kartı ile katılımcı bağlamı arasındaki bireysel farkı kompakt "
            "kalibrasyon sinyalleri olarak özetleyen dikkatli bir araştırma asistanısın. "
            "Yalnızca JSON döndür. Shift alanları profil ortalamasından sapmayı temsil eder ve "
            "-1.5 ile 1.5 arasında olmalıdır."
        )
    return (
        "You are a careful research assistant who summarizes respondent-specific deviations from "
        "the profile center as compact calibration signals. Return JSON only. Scale shifts "
        "represent deviations from the profile mean and must stay between -1.5 and 1.5."
    )


def _latent_plan_system_prompt(dataset: DatasetSettings) -> str:
    if dataset.locale == "tr":
        return (
            "Sen, persona kartı ve bağlamdan kompakt latent trait planı çıkaran dikkatli bir "
            "araştırma asistanısın. Yalnızca JSON döndür. Bu aşamada madde yanıtı üretme."
        )
    return (
        "You are a careful research assistant who extracts a compact latent-trait plan from a "
        "persona card and respondent context. Return JSON only. Do not generate item responses "
        "at this stage."
    )


def build_survey_response_schema(
    item_ids: tuple[str, ...] | list[str],
    *,
    response_min: int = DEFAULT_DATASET.response_min,
    response_max: int = DEFAULT_DATASET.response_max,
) -> type[BaseModel]:
    return _build_survey_response_schema(tuple(item_ids), response_min, response_max)


@lru_cache(maxsize=8)
def _build_survey_response_schema(
    normalized_ids: tuple[str, ...],
    response_min: int,
    response_max: int,
) -> type[BaseModel]:
    response_fields = cast(
        dict[str, Any],
        {
            item_id: (int, Field(..., ge=response_min, le=response_max))
            for item_id in normalized_ids
        },
    )
    response_model = create_model("QuestionnaireResponses", **response_fields)
    return create_model(
        "StrictSurveySimulationResponse",
        responses=(response_model, ...),
        reasoning_note=(str | None, None),
    )


class DirectStatsTwoPassPlan(BaseModel):
    target_scale_means: dict[str, float]
    extremity_preference: float = Field(..., ge=0.0, le=1.0)
    reasoning_note: str | None = None


def build_direct_stats_two_pass_plan_schema(
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> type[BaseModel]:
    target_model_name = f"{dataset.dataset_id.title().replace('_', '')}TwoPassTargetScaleMeans"
    target_fields = {
        scale.mean_column: (
            float,
            Field(..., ge=dataset.response_min, le=dataset.response_max),
        )
        for scale in dataset.scales
    }
    target_model = create_model(target_model_name, **target_fields)

    schema_name = f"{dataset.dataset_id.title().replace('_', '')}DirectStatsTwoPassPlan"
    schema_model = create_model(
        schema_name,
        target_scale_means=(target_model, Field(...)),
        extremity_preference=(float, Field(..., ge=0.0, le=1.0)),
        reasoning_note=(str | None, Field(...)),
    )

    def to_direct_stats_two_pass_plan(self) -> DirectStatsTwoPassPlan:
        return DirectStatsTwoPassPlan(
            target_scale_means={
                key: float(value)
                for key, value in self.target_scale_means.model_dump(mode="json").items()
            },
            extremity_preference=float(self.extremity_preference),
            reasoning_note=self.reasoning_note,
        )

    setattr(schema_model, "to_direct_stats_two_pass_plan", to_direct_stats_two_pass_plan)
    return schema_model


def _coerce_likert_value(
    value: object,
    *,
    response_min: int,
    response_max: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        numeric = int(round(float(value)))
    else:
        text = str(value).strip()
        if not text:
            return None
        numeric = int(round(float(text)))
    return max(response_min, min(response_max, numeric))


def _coerce_int_like(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def _is_missing_scalar(value: object) -> bool:
    missing = pd.isna(value)
    if isinstance(missing, bool):
        return missing
    if hasattr(missing, "item"):
        extracted = missing.item()
        if isinstance(extracted, bool):
            return extracted
    return False


def _questionnaire_prompt(
    prompt_version: str = "v1",
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    if dataset.locale != "tr":
        if prompt_version == "demographic_only_v1":
            lines = [
                "Answer the questionnaire as the respondent described by the respondent context alone.",
                "Base every answer only on the demographic and usage information in the respondent context.",
                "Do not repeat the same response mechanically across every item.",
                "Keep the pattern plausible and internally consistent while allowing small item-level differences.",
                f"Return every item id exactly once with an integer from {dataset.response_min} to {dataset.response_max}.",
                "Keep reasoning_note to one sentence describing how the respondent context shaped the answers.",
                "",
                "Questionnaire items:",
            ]
        elif prompt_version == "direct_stats_v1":
            lines = [
                "Use the persona card, respondent context, and empirical profile statistics together.",
                "Treat the empirical profile statistics as scale-level anchors learned from the training split.",
                "Do not copy those anchors mechanically; generate plausible item-level answers that remain internally consistent.",
                "Respondent context may create small but visible deviations from the profile center.",
                "First position the overall level of each subscale around these statistics, then answer the items one by one.",
                "Items within the same subscale need not be identical, but the scale direction must stay consistent with the empirical profile center.",
                f"Return every item id exactly once with an integer from {dataset.response_min} to {dataset.response_max}.",
                "Keep reasoning_note to one sentence describing which scales deviate from the profile center.",
                "",
                "Questionnaire items:",
            ]
        elif prompt_version == "direct_stats_two_pass_v1":
            lines = [
                "Use the persona card, respondent context, empirical profile statistics, and the target scale plan together.",
                "Treat the target scale plan as the main scale-level anchor for item generation.",
                "Do not copy the target means mechanically; generate plausible item-level answers that remain internally consistent.",
                "Use extremity_preference as a style hint: lower values stay closer to the midpoint, higher values allow more extreme options.",
                "Respondent context may create small but visible deviations from the profile center and from the target plan.",
                "Items within the same subscale need not be identical, but the realised scale direction must stay consistent with the target plan.",
                f"Return every item id exactly once with an integer from {dataset.response_min} to {dataset.response_max}.",
                "Keep reasoning_note to one sentence describing which items deviate from the target scale plan.",
                "",
                "Questionnaire items:",
            ]
        elif prompt_version == "v3_tr_gpt_d":
            # English counterpart of the Turkish FPNP block, mirrored line for line.
            interior_low = dataset.response_min + 1
            interior_high = dataset.response_max - 1
            lines = [
                "Use both the persona card and the respondent context.",
                "The persona card gives profile-level tendencies; the respondent context determines individual variation within that same profile.",
                "Use the short first-person backstory below as an internal first-person reference; do not restate the backstory in your output.",
                "The respondent context must have a visible effect on at least a few items, but do not scatter the answers so far that the profile direction breaks.",
                f"Distribute the answers realistically: use {dataset.response_min} and {dataset.response_max} only when the context strongly supports them.",
                "Items within the same subscale need not be identical; they may point in the same direction, but limited and meaningful variation is usually enough.",
                f"Across the symptom scales, do not force extreme options; produce variation from the context, and in most cases staying in the interior {interior_low}-{interior_high} band is more likely.",
                f"Return every item id exactly once with an integer from {dataset.response_min} to {dataset.response_max}.",
                "Keep reasoning_note to one sentence describing the individual deviations you made from the profile mean.",
                "",
                "Questionnaire items:",
            ]
        elif prompt_version == "v3_tr_gpt_e":
            # English counterpart of the Turkish BBP block, mirrored line for line. The
            # Turkish response-band correction names two scales and a 2-3-4 interior band
            # on a 1-5 range; here the band is derived from the response range and no
            # dataset-specific scale names are introduced.
            interior_low = dataset.response_min + 1
            interior_high = dataset.response_max - 1
            lines = [
                "Use both the persona card and the respondent context.",
                "The persona card gives profile-level tendencies; the respondent context determines individual variation within that same profile.",
                "The respondent context must have a visible effect on at least a few items, but do not scatter the answers so far that the profile direction breaks.",
                f"Distribute the answers realistically: use {dataset.response_min} and {dataset.response_max} only when the context strongly supports them.",
                "Items within the same subscale need not be identical; they may point in the same direction, but limited and meaningful variation is usually enough.",
                "Across the symptom scales, do not repeat one single mid-band pattern; produce variation from the context without forcing extreme options.",
                f"Do not treat {interior_high} as a default value; without strong supporting evidence stay more balanced around {interior_low}.",
                f"Return every item id exactly once with an integer from {dataset.response_min} to {dataset.response_max}.",
                "Keep reasoning_note to one sentence describing the individual deviations you made from the profile mean.",
                "",
                "Questionnaire items:",
            ]
        elif prompt_version in {"v3_tr_gpt_b", "v3_tr_claude_a"}:
            # English counterpart of the Turkish BPP block, mirrored line for line.
            # BPP is the *portable* persona arm, so this is a faithful port rather
            # than a Bangladesh-tuned variant: the interior-band hint is rescaled to
            # the response range but no dataset-specific scale names are introduced.
            interior_low = dataset.response_min + 1
            interior_high = dataset.response_max - 1
            lines = [
                "Use both the persona card and the respondent context.",
                "The persona card gives profile-level tendencies; the respondent context determines individual variation within that same profile.",
                "The respondent context must have a visible effect on at least a few items, but do not scatter the answers so far that the profile direction breaks.",
                f"Distribute the answers realistically: use {dataset.response_min} and {dataset.response_max} only when the context strongly supports them.",
                "Items within the same subscale need not be identical; they may point in the same direction, but limited and meaningful variation is usually enough.",
                f"Across the symptom scales, do not force extreme options; produce variation from the context, and in most cases staying in the interior {interior_low}-{interior_high} band is more likely.",
                f"Return every item id exactly once with an integer from {dataset.response_min} to {dataset.response_max}.",
                "Keep reasoning_note to one sentence describing the individual deviations you made from the profile mean.",
                "",
                "Questionnaire items:",
            ]
        else:
            lines = [
                "Use both the persona card and the respondent context.",
                "The persona card defines profile-level tendencies; the respondent context defines the individual variant.",
                "Do not repeat the same response mechanically across every item.",
                "Keep the pattern plausible and internally consistent while allowing small item-level differences.",
                f"Return every item id exactly once with an integer from {dataset.response_min} to {dataset.response_max}.",
                "",
                "Questionnaire items:",
            ]
        for item_id, question_text in dataset.question_spec:
            lines.append(f"- {item_id}: {question_text}")
        return "\n".join(lines)

    if prompt_version == "demographic_only_v1":
        lines = [
            "Anketi yalnızca katılımcı bağlamında tanımlanan katılımcı olarak yanıtla.",
            "Her yanıtı yalnızca katılımcı bağlamındaki demografik ve kullanım bilgilerine dayandır.",
            "Tüm maddelere mekanik biçimde aynı yanıtı verme.",
            "Örüntüyü makul ve kendi içinde tutarlı tut; maddeler arasında küçük farklılıklara izin ver.",
            f"Her madde kimliğini tam olarak bir kez kullan ve {dataset.response_min} ile {dataset.response_max} arasında bir tam sayı ata.",
            "reasoning_note alanında katılımcı bağlamının yanıtları nasıl şekillendirdiğini tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "v1":
        lines = [
            "Return answers for the following normalized item ids using the persona card.",
            "Use the ids exactly as given and map each one to an integer from 1 to 5.",
            "",
            "Questionnaire items:",
        ]
    elif prompt_version == "v1_compact":
        lines = [
            "Fill every item id below with a single integer from 1 to 5.",
            "Base the answers only on the persona card and stay internally consistent.",
            "",
            "Questionnaire items:",
        ]
    elif prompt_version == "v2":
        lines = [
            "Use both the persona card and the respondent context.",
            "The persona card defines profile-level tendencies; respondent context defines the individual variant.",
            "Do not copy the same answer mechanically across all items.",
            "Keep the pattern plausible: related items can differ slightly, but the overall response style must remain internally consistent.",
            "Return every item id exactly once with an integer from 1 to 5.",
            "",
            "Questionnaire items:",
        ]
    elif prompt_version == "v2_tr":
        lines = [
            "Hem persona kartını hem de katılımcı bağlamını birlikte kullan.",
            "Persona kartı profil düzeyindeki eğilimleri, katılımcı bağlamı ise bireysel varyantı tanımlar.",
            "Tüm maddelere mekanik biçimde aynı yanıtı verme.",
            "Örüntüyü makul tut: ilişkili maddeler küçük farklılıklar gösterebilir, ancak genel yanıt tarzı kendi içinde tutarlı olmalıdır.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "v3_tr_gpt":
        lines = [
            "Hem persona kartını hem de katılımcı bağlamını birlikte kullan.",
            "Persona kartı profil düzeyindeki eğilimleri verir; katılımcı bağlamı ise aynı profil içindeki bireysel varyasyonu belirler.",
            "Katılımcı bağlamı en az birkaç maddede görünür etki yaratmalıdır; yalnızca profil ortalamasına göre mekanik tekrar üretme.",
            "Yanıtları gerçekçi dağıt: gerektiğinde 1 ve 5 değerlerini kullanmaktan kaçınma, ancak bunu ancak madde içeriği ve bağlam gerçekten destekliyorsa yap.",
            "Aynı alt ölçek içindeki maddeler bire bir aynı olmak zorunda değildir; benzer yönde olabilir ama bazı maddelerde belirgin ayrışma görülebilir.",
            "Özellikle sosyal kaygı ve iyi oluş maddelerinde orta banda yığılma üretme; bağlam uygunsa daha düşük ve daha yüksek seçeneklere çık.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca profil ortalamasından yaptığın bireysel sapmaları tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "v3_tr_gpt_b":
        lines = [
            "Hem persona kartını hem de katılımcı bağlamını birlikte kullan.",
            "Persona kartı profil düzeyindeki eğilimleri verir; katılımcı bağlamı ise aynı profil içindeki bireysel varyasyonu belirler.",
            "Katılımcı bağlamı en az birkaç maddede görünür etki yaratmalıdır; ancak profil yönünü bozacak kadar dağılma üretme.",
            "Yanıtları gerçekçi dağıt: 1 ve 5 değerlerini ancak bağlam güçlü biçimde destekliyorsa kullan.",
            "Aynı alt ölçek içindeki maddeler bire bir aynı olmak zorunda değildir; benzer yönde olabilir ama çoğu durumda sınırlı ve anlamlı ayrışma yeterlidir.",
            "Özellikle sosyal kaygı ve iyi oluş maddelerinde uç seçenekleri zorlamadan bağlama göre çeşitlilik üret; 2, 3 ve 4 bandında kalmak çoğu durumda daha olası olabilir.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca profil ortalamasından yaptığın bireysel sapmaları tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "v3_tr_claude_a":
        lines = [
            "Hem persona kartını hem de katılımcı bağlamını birlikte kullan.",
            "Persona kartı profil düzeyindeki eğilimleri verir; katılımcı bağlamı ise aynı profil içindeki bireysel varyasyonu belirler.",
            "Katılımcı bağlamı en az birkaç maddede görünür etki yaratmalıdır; ancak profil yönünü bozacak kadar dağılma üretme.",
            "Yanıtları gerçekçi dağıt: 1 ve 5 değerlerini ancak bağlam güçlü biçimde destekliyorsa kullan.",
            "Aynı alt ölçek içindeki maddeler bire bir aynı olmak zorunda değildir; benzer yönde olabilir ama çoğu durumda sınırlı ve anlamlı ayrışma yeterlidir.",
            "Özellikle sosyal kaygı ve iyi oluş maddelerinde uç seçenekleri zorlamadan bağlama göre çeşitlilik üret; 2, 3 ve 4 bandında kalmak çoğu durumda daha olası olabilir.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca profil ortalamasından yaptığın bireysel sapmaları tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "v3_tr_gpt_e":
        lines = [
            "Hem persona kartını hem de katılımcı bağlamını birlikte kullan.",
            "Persona kartı profil düzeyindeki eğilimleri verir; katılımcı bağlamı ise aynı profil içindeki bireysel varyasyonu belirler.",
            "Katılımcı bağlamı en az birkaç maddede görünür etki yaratmalıdır; ancak profil yönünü bozacak kadar dağılma üretme.",
            "Yanıtları gerçekçi dağıt: 1 ve 5 değerlerini ancak bağlam güçlü biçimde destekliyorsa kullan.",
            "Aynı alt ölçek içindeki maddeler bire bir aynı olmak zorunda değildir; benzer yönde olabilir ama çoğu durumda sınırlı ve anlamlı ayrışma yeterlidir.",
            "Özellikle sosyal kaygı ve iyi oluş maddelerinde tek bir 3-4 kalıbını tekrar etme; uç seçenekleri zorlamadan bağlama göre çeşitlilik üret.",
            "İyi oluşta 4 değerini varsayılan kabul etme; güçlü olumlu kanıt yoksa 3 çevresinde daha dengeli kal.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca profil ortalamasından yaptığın bireysel sapmaları tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "v3_tr_gpt_g":
        lines = [
            "Hem persona kartını hem de katılımcı bağlamını birlikte kullan.",
            "Persona kartı profil düzeyindeki eğilimleri verir; katılımcı bağlamı ise aynı profil içindeki bireysel varyasyonu belirler.",
            "Katılımcı bağlamı en az birkaç maddede görünür etki yaratmalıdır; ancak profil yönünü bozacak kadar dağılma üretme.",
            "Yanıtları gerçekçi dağıt: 1 ve 5 değerlerini ancak bağlam güçlü biçimde destekliyorsa kullan.",
            "Aynı alt ölçek içindeki maddeler bire bir aynı olmak zorunda değildir; benzer yönde olabilir ama çoğu durumda sınırlı ve anlamlı ayrışma yeterlidir.",
            "Özellikle sosyal kaygı ve iyi oluş maddelerinde aynı 3-4 kalıbını mekanik tekrar etme; güçlü kanıt yoksa orta bant doğal olabilir ama maddeler arasında küçük ve anlamlı ayrışmalar üret.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca profil ortalamasından yaptığın bireysel sapmaları tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "v3_tr_gpt_c":
        lines = [
            "Hem persona kartını hem de katılımcı bağlamını birlikte kullan.",
            "Persona kartı profil düzeyindeki eğilimleri verir; katılımcı bağlamı ise aynı profil içindeki bireysel varyasyonu belirler.",
            "Önce her ölçeği zihninde düşük / orta / yüksek anchor seviyelerinden birine yerleştir; sonra madde yanıtlarını bu iç konuma göre üret.",
            "Bu anchor'lar ölçek anlamını sabitlemek için referans noktalarıdır; çıktıda anchor yazma, yalnızca son yanıtları ver.",
            "Katılımcı bağlamı en az birkaç maddede görünür etki yaratmalıdır, ancak profil yönünü bozacak kadar dağılma üretme.",
            "Aynı alt ölçek içindeki maddeler bire bir aynı olmak zorunda değildir; anchor seviyesiyle uyumlu küçük ve anlamlı ayrışmalar yeterlidir.",
            "Güçlü kanıt varsa 1 ve 5 kullanılabilir; güçlü kanıt yoksa gereksiz uç değer verme.",
            "Özellikle sosyal kaygı ve iyi oluş maddelerinde aynı orta kalıbı tekrar etme; bağlama göre kontrollü farklılaşma üret.",
            "",
            "Dijital bağımlılık anchorları:",
            "- Düşük: kullanımı kontrol edebilen, ayrışmakta zorlanmayan, gündelik işlevleri belirgin bozulmayan örüntü.",
            "- Orta: sık kullanan, zaman zaman dikkat kaybı yaşayan ama her maddede zorlayıcı düzeye çıkmayan örüntü.",
            "- Yüksek: bırakmakta zorlanan, kullanım nedeniyle dikkatini ve gündelik işlevini daha belirgin bozan örüntü.",
            "",
            "Sosyal kaygı anchorları:",
            "- Düşük: sosyal durumlarda görece rahat, yoğun yargılanma korkusu yaşamayan örüntü.",
            "- Orta: bazı sosyal durumlarda gerilim yaşayan ama her durumda kaçınmayan örüntü.",
            "- Yüksek: sosyal değerlendirme ve olumsuz yargılanma korkusu belirgin, kaçınma eğilimi daha güçlü örüntü.",
            "",
            "İyi oluş anchorları:",
            "- Düşük: enerji, amaçlılık ve olumlu duygu düzeyi daha zayıf örüntü.",
            "- Orta: olumlu ve olumsuz yönleri karışık, dengeli ama sınırlı iyilik hali örüntüsü.",
            "- Yüksek: olumlu duygu, amaçlılık ve kendini iyi hissetme düzeyi daha güçlü örüntü.",
            "",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca hangi ölçek anchorlarının bağlam nedeniyle öne çıktığını tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "v3_tr_gpt_d":
        lines = [
            "Hem persona kartını hem de katılımcı bağlamını birlikte kullan.",
            "Persona kartı profil düzeyindeki eğilimleri verir; katılımcı bağlamı ise aynı profil içindeki bireysel varyasyonu belirler.",
            "Aşağıdaki kısa birinci tekil backstory'yi birinci tekil iç referans olarak kullan; çıktıda bu backstory'yi yeniden yazma.",
            "Katılımcı bağlamı en az birkaç maddede görünür etki yaratmalıdır; ancak profil yönünü bozacak kadar dağılma üretme.",
            "Yanıtları gerçekçi dağıt: 1 ve 5 değerlerini ancak bağlam güçlü biçimde destekliyorsa kullan.",
            "Aynı alt ölçek içindeki maddeler bire bir aynı olmak zorunda değildir; benzer yönde olabilir ama çoğu durumda sınırlı ve anlamlı ayrışma yeterlidir.",
            "Özellikle sosyal kaygı ve iyi oluş maddelerinde uç seçenekleri zorlamadan bağlama göre çeşitlilik üret; 2, 3 ve 4 bandında kalmak çoğu durumda daha olası olabilir.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca profil ortalamasından yaptığın bireysel sapmaları tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "direct_stats_v1":
        lines = [
            "Hem persona kartını, hem katılımcı bağlamını, hem de ampirik profil istatistiklerini birlikte kullan.",
            "Ampirik profil istatistikleri profilin eğitim splitindeki merkezini gösterir; bunları ölçek düzeyi referansı olarak kullan.",
            "Bu istatistikleri bire bir kopyalama; tek tek madde yanıtı üretirken makul ve kendi içinde tutarlı bir örüntü kur.",
            "Katılımcı bağlamı profil ortalamasından küçük ama görünür sapmalar üretebilir.",
            "Önce her alt ölçeğin genel seviyesini bu istatistikler etrafında zihninde konumlandır, sonra maddeleri tek tek yanıtla.",
            "Aynı alt ölçek içindeki maddeler bire bir aynı olmak zorunda değildir; fakat ölçek yönü ampirik profil merkeziyle uyumlu kalmalıdır.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca profil istatistiklerine göre hangi ölçeklerde bağlamsal sapma yaptığını tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "direct_stats_two_pass_v1":
        lines = [
            "Hem persona kartını, hem katılımcı bağlamını, hem ampirik profil istatistiklerini, hem de hedef ölçek planını birlikte kullan.",
            "Hedef ölçek planı her alt ölçek için hedef ortalamayı verir; madde yanıtlarını öncelikle bu plan etrafında kur.",
            "Hedef ortalamaları bire bir mekanik kopyalama; madde düzeyinde küçük ve anlamlı farklılaşmalar üret.",
            "extremity_preference değeri 0'a yakınsa orta banda daha yakın kal, 1'e yakınsa güçlü kanıt olan maddelerde daha uç seçeneklere açıl.",
            "Katılımcı bağlamı profil merkezinden ve hedef plandan küçük ama görünür sapmalar yaratabilir.",
            "Aynı alt ölçek içindeki maddeler bire bir aynı olmak zorunda değildir; fakat gerçekleşen ölçek yönü hedef planla uyumlu kalmalıdır.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca hedef ölçek planına göre hangi maddelerde bağlamsal sapma yaptığını tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "v4_tr_hybrid":
        lines = [
            "Hem persona kartını hem de katılımcı bağlamını birlikte kullan.",
            "Persona kartı profil düzeyindeki eğilimleri verir; katılımcı bağlamı ise aynı profil içindeki bireysel varyasyonu belirler.",
            "Katılımcı bağlamı maddelerde görünür olabilir, ancak yalnızca gerçekten desteklenen maddelerde sapma üret.",
            "Orta bandı otomatik tekrar etme, fakat güçlü kanıt yoksa gereksiz uç değer de verme.",
            "Aynı alt ölçek içindeki maddeler bire bir aynı olmak zorunda değildir; benzer yönde olabilir ama sadece anlamlı yerlerde ayrışsın.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca profil ortalamasından yaptığın bireysel sapmaları tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    elif prompt_version == "v4_tr_hybrid_b":
        lines = [
            "Hem persona kartını hem de katılımcı bağlamını birlikte kullan.",
            "Persona kartı profil düzeyindeki eğilimleri verir; katılımcı bağlamı ise aynı profil içindeki bireysel varyasyonu belirler.",
            "Katılımcı bağlamı maddelerde görünür olabilir, ancak yalnızca gerçekten desteklenen maddelerde sapma üret.",
            "Profil yönünü bozacak kadar dağılma üretme; güçlü kanıt yoksa daha sınırlı ayrışma tercih et.",
            "Özellikle sosyal kaygı ve iyi oluş maddelerinde uç seçenekleri zorlamadan bağlama göre küçük ve anlamlı farklılaşmalar üret.",
            "Her madde kimliğini tam olarak bir kez kullan ve 1 ile 5 arasında bir tam sayı ata.",
            "reasoning_note alanında yalnızca profil ortalamasından yaptığın bireysel sapmaları tek cümlede açıkla.",
            "",
            "Anket maddeleri:",
        ]
    else:
        raise ValueError(f"Unsupported prompt version: {prompt_version}")

    for item_id, question_text in dataset.question_spec:
        lines.append(f"- {item_id}: {question_text}")
    return "\n".join(lines)


def _calibration_signal_prompt(
    prompt_version: str = "v2_tr",
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    scale_lines = [
        f"- {scale.mean_column}: {dataset.feature_label(scale.mean_column)}"
        for scale in dataset.scales
    ]
    if dataset.locale == "tr":
        prompt_lines = [
            "Aşağıdaki persona kartı ve katılımcı bağlamına göre profil ortalamasından bireysel sapmaları özetle.",
            "Tam madde yanıtı üretme; yalnızca kompakt kalibrasyon sinyali ver.",
            "scale_shifts nesnesi içinde her ölçek için -1.5 ile 1.5 arasında bir kayma değeri ver.",
            "extremity_preference alanını 0 ile 1 arasında seç: 0 daha orta banda yakın, 1 daha uç seçeneklere açık demektir.",
            "reasoning_note tek cümle olsun ve yalnızca bağlamın profil ortalamasından neden sapabileceğini anlatsın.",
        ]
        if prompt_version == "v4_tr_hybrid":
            prompt_lines.extend(
                [
                    "Varsayılan olarak tüm shift alanlarını 0'a yakın tut; yalnızca güçlü ve açık bağlamsal kanıt varsa belirgin sapma ver.",
                    "Güçlü kanıt yoksa birden fazla ölçeği aynı anda oynatma; çoğu durumda en fazla tek bir ana ölçeğe küçük/orta bir kayma ver.",
                    "extremity_preference değerini çoğu durumda 0.35 ile 0.55 arasında tut.",
                ]
            )
        elif prompt_version == "v4_tr_hybrid_b":
            prompt_lines.extend(
                [
                    "Varsayılan olarak tüm shift alanlarını 0'a yakın tut; yalnızca güçlü ve açık bağlamsal kanıt varsa belirgin sapma ver.",
                    "Güçlü kanıt yoksa birden fazla ölçeği aynı anda oynatma; çoğu durumda en fazla tek bir ana ölçeğe küçük/orta bir kayma ver.",
                    "Özellikle sosyal kaygı ve iyi oluş için varsayılanı küçük ve orta sapmalar olsun.",
                    "Çok güçlü kanıt yoksa bu iki ölçekte aynı anda belirgin kayma verme.",
                    "extremity_preference değerini çoğu durumda 0.25 ile 0.45 arasında tut.",
                ]
            )
        elif prompt_version == "v3_tr_gpt":
            prompt_lines.append(
                "Katılımcı bağlamı görünür etki yaratmalı; yalnızca profil özetini tekrar eden sıfır sinyali verme."
            )
        prompt_lines.extend(["", "Ölçek anahtarları:", *scale_lines])
        return "\n".join(prompt_lines)

    prompt_lines = [
        "Summarize respondent-specific deviations from the profile center using compact calibration signals.",
        "Do not generate item responses here; return only the calibration signal.",
        "Provide a scale_shifts object with one shift value between -1.5 and 1.5 for each scale key below.",
        "Set extremity_preference between 0 and 1: 0 means closer to the midpoint, 1 means more open to extreme options.",
        "Keep reasoning_note to one sentence describing why the context may deviate from the profile center.",
    ]
    if prompt_version in {"v4_tr_hybrid", "v4_tr_hybrid_b"}:
        prompt_lines.extend(
            [
                "Keep shifts near zero by default; only use visible deviations when the context provides clear evidence.",
                "Without strong evidence, avoid moving multiple scales at once.",
            ]
        )
    prompt_lines.extend(["", "Scale keys:", *scale_lines])
    return "\n".join(prompt_lines)


def _build_respondent_context(
    row: pd.Series,
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> dict[str, object]:
    context: dict[str, object] = {}
    for column in dataset.context_columns:
        if column not in row.index:
            continue
        value = row[column]
        if _is_missing_scalar(value):
            continue
        context[column] = value.item() if hasattr(value, "item") else value
    return context


def _render_profile_narrative(
    narrative: dict[str, object],
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    lines = ["Profil özeti:" if dataset.locale == "tr" else "Profile summary:"]
    if "profile_id" in narrative:
        lines.append(
            f"- {'Profil kimliği' if dataset.locale == 'tr' else 'Profile id'}: {narrative['profile_id']}"
        )
    if "label" in narrative:
        lines.append(f"- {'Etiket' if dataset.locale == 'tr' else 'Label'}: {narrative['label']}")
    if "summary" in narrative:
        lines.append(
            f"- {'Kısa açıklama' if dataset.locale == 'tr' else 'Short summary'}: {narrative['summary']}"
        )
    if narrative.get("demographic_anchor"):
        lines.append(
            f"- {'Demografik dayanak' if dataset.locale == 'tr' else 'Demographic anchor'}: {narrative['demographic_anchor']}"
        )
    if narrative.get("size") is not None:
        lines.append(
            f"- {'Eğitim örneklemindeki kişi sayısı' if dataset.locale == 'tr' else 'Training-split profile size'}: {narrative['size']}"
        )

    feature_means = narrative.get("feature_means")
    if isinstance(feature_means, dict) and feature_means:
        lines.append(
            "- Sayısal profil göstergeleri:"
            if dataset.locale == "tr"
            else "- Numeric profile indicators:"
        )
        for key, value in feature_means.items():
            label = dataset.feature_label(str(key))
            lines.append(f"  - {label}: {value}")

    return "\n".join(lines)


def _render_compact_profile_stats(
    narrative: dict[str, object],
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    lines = [
        "Ampirik profil istatistikleri:"
        if dataset.locale == "tr"
        else "Empirical profile statistics:"
    ]
    if narrative.get("size") is not None:
        lines.append(
            f"- {'Eğitim splitindeki profil büyüklüğü' if dataset.locale == 'tr' else 'Training-split profile size'}: {narrative['size']}"
        )

    feature_means = narrative.get("feature_means")
    if isinstance(feature_means, dict) and feature_means:
        preferred_order = list(dataset.scale_columns) + ["usage_hours_midpoint"]
        remaining_keys = [
            key for key in feature_means.keys() if key not in preferred_order
        ]
        for key in preferred_order + remaining_keys:
            if key not in feature_means:
                continue
            value = feature_means[key]
            label = dataset.feature_label(str(key))
            if key == "usage_hours_midpoint":
                unit = "saat" if dataset.locale == "tr" else "hours"
                lines.append(f"- {label}: {value} {unit}")
            else:
                lines.append(f"- {label}: {value}")

    return "\n".join(lines)


def _render_direct_stats_two_pass_plan(
    plan: DirectStatsTwoPassPlan,
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    lines = [
        "Hedef ölçek planı:" if dataset.locale == "tr" else "Target scale plan:"
    ]
    for scale in dataset.scales:
        label = dataset.feature_label(scale.mean_column)
        if dataset.locale == "tr":
            target_label = (
                label.replace("ortalaması", "hedef ortalaması")
                if "ortalaması" in label
                else f"{label} hedef ortalaması"
            )
        else:
            target_label = (
                label.replace("mean", "target mean")
                if "mean" in label.lower()
                else f"{label} target mean"
            )
        lines.append(f"- {target_label}: {plan.target_scale_means[scale.mean_column]}")
    lines.append(f"- extremity_preference: {plan.extremity_preference}")
    if plan.reasoning_note:
        lines.append(f"- reasoning_note: {plan.reasoning_note}")
    return "\n".join(lines)


def _render_persona_card(
    persona: PersonaCard,
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    lines = [
        "Persona kartı:" if dataset.locale == "tr" else "Persona card:",
        f"- {'Profil kimliği' if dataset.locale == 'tr' else 'Profile id'}: {persona.profile_id}",
        f"- {'Etiket' if dataset.locale == 'tr' else 'Label'}: {persona.label}",
        f"- {'Özet' if dataset.locale == 'tr' else 'Summary'}: {persona.summary}",
        f"- {'Demografik dayanak' if dataset.locale == 'tr' else 'Demographic anchor'}: {persona.demographic_anchor}",
        "- Davranış göstergeleri:" if dataset.locale == "tr" else "- Behavior markers:",
    ]
    for marker in persona.behavior_markers:
        lines.append(f"  - {marker}")
    lines.append(f"- {'Ton' if dataset.locale == 'tr' else 'Tone'}: {persona.tone}")
    return "\n".join(lines)


def _membership_probabilities(row: pd.Series) -> dict[int, float] | None:
    """Profile id -> membership probability, when the run passes soft membership."""
    columns = [c for c in row.index if str(c).startswith(MEMBERSHIP_PREFIX)]
    if not columns:
        return None
    return {int(str(c)[len(MEMBERSHIP_PREFIX):]): float(row[c]) for c in columns}


def _render_soft_membership(
    membership: dict[int, float],
    persona_by_profile: dict[int, PersonaCard],
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    """Every profile's persona card with the respondent's membership probability.

    Profiles are listed in a fixed order, by profile id, so the order carries no
    information about which profile is the likeliest.
    """
    tr = dataset.locale == "tr"
    lines = [
        "Profil üyeliği:" if tr else "Profile membership:",
        (
            "Katılımcının profili bilinmiyor. Aşağıdaki olasılıklar yalnızca katılımcı "
            "bağlamından tahmin edilmiştir. Her aday profil, üyelik olasılığı ve persona "
            "kartıyla birlikte verilmiştir. Persona kartlarını bu olasılıklara göre "
            "ağırlıklandırarak kullan."
        )
        if tr
        else (
            "The respondent's profile is not known. The probabilities below were estimated "
            "from the respondent context alone. Each candidate profile is listed with its "
            "membership probability and its persona card. Weigh the persona cards by these "
            "probabilities."
        ),
    ]
    blocks = ["\n".join(lines)]
    total = len(membership)
    for position, profile_id in enumerate(sorted(membership), start=1):
        header = (
            f"Aday profil {position}/{total}. Üyelik olasılığı: {membership[profile_id]:.2f}"
            if tr
            else f"Candidate profile {position} of {total}. Membership probability: {membership[profile_id]:.2f}"
        )
        blocks.append(
            header + "\n" + _render_persona_card(persona_by_profile[profile_id], dataset=dataset)
        )
    return "\n\n".join(blocks)


def _render_weighted_profile_stats(
    membership: dict[int, float],
    narrative_by_profile: dict[int, dict[str, object]],
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    """Profile statistics averaged with the membership probabilities as weights.

    Same lines and number format as the hard version, without the profile size, which
    has no meaning for a weighted average.
    """
    weighted: dict[str, float] = {}
    for profile_id, probability in membership.items():
        means = narrative_by_profile[profile_id].get("feature_means")
        if not isinstance(means, dict):
            raise ValueError("soft membership requires feature_means for every profile.")
        for key, value in means.items():
            weighted[key] = weighted.get(key, 0.0) + probability * float(value)
    lines = [
        "Ampirik profil istatistikleri (üyelik olasılıklarıyla ağırlıklandırılmış):"
        if dataset.locale == "tr"
        else "Empirical profile statistics (weighted by the membership probabilities):"
    ]
    preferred_order = list(dataset.scale_columns) + ["usage_hours_midpoint"]
    for key in preferred_order + [k for k in weighted if k not in preferred_order]:
        if key not in weighted:
            continue
        label = dataset.feature_label(str(key))
        if key == "usage_hours_midpoint":
            unit = "saat" if dataset.locale == "tr" else "hours"
            lines.append(f"- {label}: {weighted[key]} {unit}")
        else:
            lines.append(f"- {label}: {weighted[key]}")
    return "\n".join(lines)


SOFT_MEMBERSHIP_PROMPTS = {"v3_tr_gpt_b", "direct_stats_v1"}


def _build_soft_user_prompt(
    membership: dict[int, float],
    persona_by_profile: dict[int, PersonaCard],
    narrative_by_profile: dict[int, dict[str, object]] | None,
    respondent_context: dict[str, object],
    questionnaire: str,
    prompt_version: str,
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    """The BPP or SADP prompt with the single persona card replaced by all candidates.

    Respondent context and questionnaire instructions are those of the hard version. For
    SADP the profile statistics become their membership-weighted average, so the two
    soft arms differ by the statistics block alone, as the two hard arms do.
    """
    if prompt_version not in SOFT_MEMBERSHIP_PROMPTS:
        raise ValueError(f"soft membership is not defined for prompt_version '{prompt_version}'.")
    missing = [p for p in membership if p not in persona_by_profile]
    if missing:
        raise ValueError(f"soft membership requires a persona card for every profile; missing {missing}.")
    parts = [
        _render_soft_membership(membership, persona_by_profile, dataset=dataset),
        _render_respondent_context(respondent_context, dataset=dataset),
    ]
    if prompt_version == "direct_stats_v1":
        if narrative_by_profile is None or any(p not in narrative_by_profile for p in membership):
            raise ValueError("direct_stats_v1 requires profile_narratives for every profile_id.")
        parts.append(_render_weighted_profile_stats(membership, narrative_by_profile, dataset=dataset))
    parts.append(questionnaire)
    return "\n\n".join(parts)


def _render_respondent_context(
    context: dict[str, object],
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    lines = ["Katılımcı bağlamı:" if dataset.locale == "tr" else "Respondent context:"]
    for key, value in context.items():
        label = dataset.context_labels.get(key, key)
        if key == "usage_hours_midpoint":
            unit = "saat" if dataset.locale == "tr" else "hours"
            lines.append(f"- {label}: {value} {unit}")
        else:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _render_first_person_backstory(
    persona: PersonaCard,
    context: dict[str, object],
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    if dataset.locale != "tr":
        opening_bits: list[str] = []
        if context.get("age_band"):
            opening_bits.append(f"in the {context['age_band']} age band")
        if context.get("education"):
            opening_bits.append(f"with {context['education']} education")
        elif context.get("program"):
            opening_bits.append(f"studying in {context['program']}")
        opening = " ".join(opening_bits)
        first_sentence = (
            f"I am a respondent {opening}."
            if opening
            else "I am a respondent with a similar profile."
        )
        lines = ["Short first-person backstory:", first_sentence]
        if context.get("residence"):
            lines.append(f"My living context is mostly {context['residence']}.")
        if context.get("usage_band"):
            if context.get("primary_platform"):
                lines.append(
                    f"My daily social-media use is usually {context['usage_band']}, and I mostly use {context['primary_platform']}."
                )
            else:
                lines.append(f"My daily social-media use is usually {context['usage_band']}.")
        lines.append(f"My overall pattern is closest to {persona.summary.lower()}.")
        if persona.behavior_markers:
            lines.append(
                "I would describe myself most as: "
                + "; ".join(str(marker) for marker in persona.behavior_markers[:3])
                + "."
            )
        lines.append(f"My general tone is {persona.tone.lower()}.")
        return "\n".join(lines)

    age_band = context.get("age_band")
    program = context.get("program")
    residence = context.get("residence")
    usage_band = context.get("usage_band")
    platforms = context.get("platforms")

    opening_parts: list[str] = []
    if age_band:
        opening_parts.append(f"{age_band} yaş aralığında")
    if program:
        opening_parts.append(f"{program} düzeyinde okuyan")
    opening = " ".join(opening_parts)
    if opening:
        first_sentence = f"Ben {opening} bir öğrenciyim."
    else:
        first_sentence = "Ben bu profile yakın özellikler taşıyan bir öğrenciyim."

    lines = [
        "Kısa birinci tekil backstory:",
        first_sentence,
    ]
    if residence:
        lines.append(f"Kaldığım yer ve yaşam düzenim çoğunlukla {residence}.")
    if usage_band:
        if platforms:
            lines.append(
                f"Günlük medya kullanımım çoğu zaman {usage_band}; en sık {platforms} kullanırım."
            )
        else:
            lines.append(f"Günlük medya kullanımım çoğu zaman {usage_band} düzeyindedir.")
    lines.append(f"Genel örüntüm {persona.summary.lower()} çizgisine yakındır.")
    if persona.behavior_markers:
        lines.append(
            "Kendimi en çok şu yönlerle tanımlarım: "
            + "; ".join(str(marker) for marker in persona.behavior_markers[:3])
            + "."
        )
    lines.append(f"İnsanlarla ilişkide tonum çoğunlukla {persona.tone.lower()} olur.")
    return "\n".join(lines)


def _build_direct_user_prompt(
    persona: PersonaCard,
    respondent_context: dict[str, object],
    questionnaire: str,
    prompt_version: str,
    *,
    profile_narrative: dict[str, object] | None = None,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    if prompt_version == "demographic_only_v1":
        return "\n\n".join(
            [
                _render_respondent_context(respondent_context, dataset=dataset),
                questionnaire,
            ]
        )
    if prompt_version == "direct_stats_v1":
        if profile_narrative is None:
            raise ValueError("direct_stats_v1 requires profile_narratives for every profile_id.")
        return "\n\n".join(
            [
                _render_persona_card(persona, dataset=dataset),
                _render_respondent_context(respondent_context, dataset=dataset),
                _render_compact_profile_stats(profile_narrative, dataset=dataset),
                questionnaire,
            ]
        )
    if prompt_version == "v3_tr_gpt_d":
        return "\n\n".join(
            [
                _render_persona_card(persona, dataset=dataset),
                _render_respondent_context(respondent_context, dataset=dataset),
                _render_first_person_backstory(persona, respondent_context, dataset=dataset),
                questionnaire,
            ]
        )
    # Every persona-conditioned arm renders the persona card and respondent context as
    # native-language prose. ``v3_tr_gpt_b`` (BPP) was originally omitted from this set and
    # alone fell through to the JSON fallback below, so it differed in prompt format from
    # the arms it is compared against.
    if prompt_version in {
        "v2_tr",
        "v3_tr_gpt",
        "v3_tr_gpt_b",
        "v3_tr_gpt_e",
        "v3_tr_claude_a",
    }:
        return "\n\n".join(
            [
                _render_persona_card(persona, dataset=dataset),
                _render_respondent_context(respondent_context, dataset=dataset),
                questionnaire,
            ]
        )
    return "\n\n".join(
        [
            "Persona card:",
            persona.model_dump_json(indent=2),
            "Respondent context:",
            json.dumps(respondent_context, ensure_ascii=False, indent=2),
            questionnaire,
        ]
    )


def _direct_stats_two_pass_plan_system_prompt(dataset: DatasetSettings) -> str:
    if dataset.locale == "tr":
        return (
            "Sen, persona kartı, katılımcı bağlamı ve ampirik profil istatistiklerinden "
            "kompakt bir hedef ölçek planı çıkaran dikkatli bir araştırma asistanısın. "
            "Bu aşamada yalnızca JSON döndür ve tam madde yanıtı üretme."
        )
    return (
        "You are a careful research assistant who extracts a compact target scale plan "
        "from a persona card, respondent context, and empirical profile statistics. "
        "Return JSON only and do not generate item-level survey answers at this stage."
    )


def _build_direct_stats_two_pass_plan_user_prompt(
    persona: PersonaCard,
    respondent_context: dict[str, object],
    *,
    profile_narrative: dict[str, object],
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    if dataset.locale == "tr":
        instructions = [
            "Görev: Persona kartı, katılımcı bağlamı ve ampirik profil istatistiklerine göre kompakt bir hedef ölçek planı üret.",
            "Tam madde yanıtı üretme; yalnızca target_scale_means, extremity_preference ve reasoning_note alanlarını doldur.",
            f"target_scale_means içinde her ölçek için {dataset.response_min} ile {dataset.response_max} arasında bir hedef ortalama ver.",
            "Hedef ortalamalar ampirik profil merkezine yakın kalmalı; bağlam güçlü olduğunda küçük ama görünür sapmalar uygulanabilir.",
            "extremity_preference değerini 0 ile 1 arasında seç: 0 orta banda daha yakın, 1 uç seçeneklere daha açık demektir.",
            "reasoning_note tek cümle olsun ve yalnızca hangi ölçeklerin profil merkezinden neden hafif saptığını anlatsın.",
        ]
    else:
        instructions = [
            "Task: Produce a compact target scale plan from the persona card, respondent context, and empirical profile statistics.",
            "Do not generate item-level survey answers; fill only target_scale_means, extremity_preference, and reasoning_note.",
            f"Inside target_scale_means, provide a target mean between {dataset.response_min} and {dataset.response_max} for each scale.",
            "Target means should stay close to the empirical profile center, with only small but visible deviations when the context strongly supports them.",
            "Choose extremity_preference between 0 and 1: 0 stays closer to the midpoint, 1 allows more extreme options.",
            "Keep reasoning_note to one sentence describing why the context shifts any scale away from the profile center.",
        ]
    return "\n\n".join(
        [
            _render_persona_card(persona, dataset=dataset),
            _render_respondent_context(respondent_context, dataset=dataset),
            _render_compact_profile_stats(profile_narrative, dataset=dataset),
            "\n".join(instructions),
        ]
    )


def _build_direct_stats_two_pass_response_user_prompt(
    persona: PersonaCard,
    respondent_context: dict[str, object],
    questionnaire: str,
    *,
    profile_narrative: dict[str, object],
    target_plan: DirectStatsTwoPassPlan,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    return "\n\n".join(
        [
            _render_persona_card(persona, dataset=dataset),
            _render_respondent_context(respondent_context, dataset=dataset),
            _render_compact_profile_stats(profile_narrative, dataset=dataset),
            _render_direct_stats_two_pass_plan(target_plan, dataset=dataset),
            questionnaire,
        ]
    )


def _build_latent_plan_user_prompt(
    persona: PersonaCard,
    respondent_context: dict[str, object],
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> str:
    return build_tune_f_plan_prompt(
        persona_card_block=_render_persona_card(persona, dataset=dataset),
        respondent_context_block=_render_respondent_context(respondent_context, dataset=dataset),
        first_person_backstory=_render_first_person_backstory(persona, respondent_context, dataset=dataset),
    )


def _generate_direct_response_candidate(
    client: LMStudioClient,
    *,
    user_prompt: str,
    response_schema: type[BaseModel],
    preset: InferencePreset,
    model_id: str,
    question_spec: list[tuple[str, str]],
    dataset: DatasetSettings = DEFAULT_DATASET,
    prompt_version: str = "v1",
) -> tuple[dict[str, int | None], RunMetadata]:
    parsed, metadata = client.generate_structured(
        system_prompt=_survey_system_prompt(dataset, prompt_version),
        user_prompt=user_prompt,
        schema=response_schema,
        preset=preset,
        model_id=model_id,
    )
    responses_model = cast(BaseModel, getattr(parsed, "responses"))
    responses = cast(dict[str, object], responses_model.model_dump())
    normalized_responses = {
        item_id: _coerce_likert_value(
            responses.get(item_id),
            response_min=dataset.response_min,
            response_max=dataset.response_max,
        )
        for item_id, _ in question_spec
    }
    return normalized_responses, metadata


def _generate_direct_stats_two_pass_plan_candidate(
    client: LMStudioClient,
    *,
    user_prompt: str,
    preset: InferencePreset,
    model_id: str,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> tuple[DirectStatsTwoPassPlan, RunMetadata]:
    plan_schema = build_direct_stats_two_pass_plan_schema(dataset)
    parsed, metadata = client.generate_structured(
        system_prompt=_direct_stats_two_pass_plan_system_prompt(dataset),
        user_prompt=user_prompt,
        schema=plan_schema,
        preset=preset,
        model_id=model_id,
    )
    return parsed.to_direct_stats_two_pass_plan(), metadata


def _generate_latent_plan_candidate(
    client: LMStudioClient,
    *,
    user_prompt: str,
    preset: InferencePreset,
    model_id: str,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> tuple[LatentTraitPlan, RunMetadata]:
    return client.generate_structured(
        system_prompt=_latent_plan_system_prompt(dataset),
        user_prompt=user_prompt,
        schema=LatentTraitPlan,
        preset=preset,
        model_id=model_id,
    )


def _response_distance(
    left: dict[str, int | None],
    right: dict[str, int | None],
    *,
    question_spec: list[tuple[str, str]],
) -> int:
    total = 0
    for item_id, _ in question_spec:
        left_value = left.get(item_id)
        right_value = right.get(item_id)
        if left_value is None or right_value is None:
            total += 4
            continue
        total += abs(left_value - right_value)
    return total


def _calibration_signal_distance(left: CalibrationSignal, right: CalibrationSignal) -> float:
    shift_keys = set(left.scale_shifts) | set(right.scale_shifts)
    return sum(
        abs(float(left.scale_shifts.get(key, 0.0)) - float(right.scale_shifts.get(key, 0.0)))
        for key in shift_keys
    ) + abs(left.extremity_preference - right.extremity_preference)


def _select_self_consistent_candidate_index(
    candidates: list[dict[str, int | None]],
    *,
    question_spec: list[tuple[str, str]],
) -> int:
    if len(candidates) <= 1:
        return 0
    candidate_scores = []
    for index, candidate in enumerate(candidates):
        score = sum(
            _response_distance(candidate, other, question_spec=question_spec)
            for other_index, other in enumerate(candidates)
            if other_index != index
        )
        candidate_scores.append(score)
    return min(range(len(candidate_scores)), key=lambda index: (candidate_scores[index], index))


def _select_self_consistent_signal_index(
    candidates: list[CalibrationSignal],
) -> int:
    if len(candidates) <= 1:
        return 0
    candidate_scores = []
    for index, candidate in enumerate(candidates):
        score = sum(
            _calibration_signal_distance(candidate, other)
            for other_index, other in enumerate(candidates)
            if other_index != index
        )
        candidate_scores.append(score)
    return min(range(len(candidate_scores)), key=lambda index: (candidate_scores[index], index))


def _aggregate_run_metadata(
    metadatas: list[RunMetadata], *, selected_index: int
) -> RunMetadata:
    selected = metadatas[selected_index]
    return RunMetadata(
        model_id=selected.model_id,
        preset_name=selected.preset_name,
        prompt_tokens=sum(metadata.prompt_tokens for metadata in metadatas),
        completion_tokens=sum(metadata.completion_tokens for metadata in metadatas),
        total_tokens=sum(metadata.total_tokens for metadata in metadatas),
        response_id=selected.response_id,
    )


def _combine_run_metadata(*metadatas: RunMetadata) -> RunMetadata:
    selected = metadatas[-1]
    preset_names = [metadata.preset_name for metadata in metadatas if metadata.preset_name]
    if len(set(preset_names)) == 1:
        preset_name = preset_names[0]
    else:
        preset_name = "+".join(preset_names)
    return RunMetadata(
        model_id=selected.model_id,
        preset_name=preset_name,
        prompt_tokens=sum(metadata.prompt_tokens for metadata in metadatas),
        completion_tokens=sum(metadata.completion_tokens for metadata in metadatas),
        total_tokens=sum(metadata.total_tokens for metadata in metadatas),
        response_id=selected.response_id,
    )


def _build_profile_narrative_lookup(
    profile_narratives: Iterable[dict[str, object]] | None,
) -> dict[int, dict[str, object]]:
    if profile_narratives is None:
        return {}
    lookup: dict[int, dict[str, object]] = {}
    for narrative in profile_narratives:
        profile_id = narrative.get("profile_id")
        if profile_id is None:
            continue
        lookup[int(profile_id)] = narrative
    return lookup


def _respondent_seed(
    random_state: int | None,
    *,
    respondent_id: object,
    run_label: str,
) -> int:
    seed_basis = f"{random_state}:{run_label}:{respondent_id}"
    digest = hashlib.sha256(seed_basis.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _apply_scale_means_to_record(
    record: dict[str, object],
    *,
    dataset: DatasetSettings,
) -> None:
    for scale in dataset.scales:
        record[scale.mean_column] = pd.Series(
            [record.get(item_id) for item_id in scale.item_columns],
            dtype="float64",
        ).mean()


def generate_persona_cards(
    client: LMStudioClient,
    profile_narratives: Iterable[dict[str, object]],
    preset: InferencePreset,
    *,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> tuple[list[PersonaCard], list[RunMetadata]]:
    narratives = list(profile_narratives)
    cards: list[PersonaCard] = []
    metadata_records: list[RunMetadata] = []
    with make_progress(
        total=len(narratives),
        desc="Generating persona cards",
        unit="card",
        leave=False,
    ) as progress:
        for narrative in narratives:
            if dataset.locale == "tr":
                user_prompt = (
                    "Aşağıdaki ampirik öğrenci profil özetinden yapılandırılmış bir persona kartı oluştur.\n\n"
                    "label, summary, demographic_anchor, behavior_markers ve tone alanlarını Türkçe yaz.\n\n"
                    f"{_render_profile_narrative(narrative, dataset=dataset)}"
                )
            else:
                user_prompt = (
                    "Create a structured persona card from the empirical respondent profile summary below.\n\n"
                    "Write label, summary, demographic_anchor, behavior_markers, and tone in English.\n\n"
                    f"{_render_profile_narrative(narrative, dataset=dataset)}"
                )
            parsed, metadata = client.generate_structured(
                system_prompt=_persona_system_prompt(dataset),
                user_prompt=user_prompt,
                schema=PersonaCard,
                preset=preset,
            )
            card = parsed.model_copy(
                update={"profile_id": _coerce_int_like(narrative["profile_id"])}
            )
            cards.append(card)
            metadata_records.append(metadata)
            progress.update(1)
    return cards, metadata_records


def _simulate_single_response(
    client: LMStudioClient,
    row: pd.Series,
    *,
    persona_by_profile: dict[int, PersonaCard],
    profile_narrative_by_profile: dict[int, dict[str, object]] | None,
    questionnaire: str,
    response_schema: type[BaseModel],
    preset: InferencePreset,
    run_label: str,
    prompt_version: str,
    model_id: str,
    dataset: DatasetSettings,
    question_spec: list[tuple[str, str]],
    self_consistency_samples: int = 1,
) -> tuple[dict[str, object], RunMetadata]:
    profile_id = int(row["profile_id"])
    persona = persona_by_profile[profile_id]
    respondent_context = _build_respondent_context(row, dataset=dataset)
    profile_narrative = (
        profile_narrative_by_profile.get(profile_id)
        if profile_narrative_by_profile is not None
        else None
    )
    selected_plan: DirectStatsTwoPassPlan | None = None
    if prompt_version == "direct_stats_two_pass_v1":
        if profile_narrative is None:
            raise ValueError(
                "direct_stats_two_pass_v1 requires profile_narratives for every profile_id."
            )
        plan_user_prompt = _build_direct_stats_two_pass_plan_user_prompt(
            persona,
            respondent_context,
            profile_narrative=profile_narrative,
            dataset=dataset,
        )
        if self_consistency_samples <= 1:
            selected_plan, plan_metadata = _generate_direct_stats_two_pass_plan_candidate(
                client,
                user_prompt=plan_user_prompt,
                preset=preset,
                model_id=model_id,
                dataset=dataset,
            )
            response_user_prompt = _build_direct_stats_two_pass_response_user_prompt(
                persona,
                respondent_context,
                questionnaire,
                profile_narrative=profile_narrative,
                target_plan=selected_plan,
                dataset=dataset,
            )
            responses, response_metadata = _generate_direct_response_candidate(
                client,
                user_prompt=response_user_prompt,
                response_schema=response_schema,
                preset=preset,
                model_id=model_id,
                question_spec=question_spec,
                dataset=dataset,
                prompt_version=prompt_version,
            )
            metadata = _combine_run_metadata(plan_metadata, response_metadata)
            selected_index = 0
        else:
            candidate_payloads: list[dict[str, int | None]] = []
            candidate_plans: list[DirectStatsTwoPassPlan] = []
            candidate_metadatas: list[RunMetadata] = []
            for _ in range(self_consistency_samples):
                candidate_plan, plan_metadata = _generate_direct_stats_two_pass_plan_candidate(
                    client,
                    user_prompt=plan_user_prompt,
                    preset=preset,
                    model_id=model_id,
                    dataset=dataset,
                )
                response_user_prompt = _build_direct_stats_two_pass_response_user_prompt(
                    persona,
                    respondent_context,
                    questionnaire,
                    profile_narrative=profile_narrative,
                    target_plan=candidate_plan,
                    dataset=dataset,
                )
                candidate_responses, response_metadata = _generate_direct_response_candidate(
                    client,
                    user_prompt=response_user_prompt,
                    response_schema=response_schema,
                    preset=preset,
                    model_id=model_id,
                    question_spec=question_spec,
                    dataset=dataset,
                    prompt_version=prompt_version,
                )
                candidate_payloads.append(candidate_responses)
                candidate_plans.append(candidate_plan)
                candidate_metadatas.append(
                    _combine_run_metadata(plan_metadata, response_metadata)
                )
            selected_index = _select_self_consistent_candidate_index(
                candidate_payloads,
                question_spec=question_spec,
            )
            responses = candidate_payloads[selected_index]
            selected_plan = candidate_plans[selected_index]
            metadata = _aggregate_run_metadata(
                candidate_metadatas,
                selected_index=selected_index,
            )
    else:
        membership = _membership_probabilities(row)
        if membership is not None:
            user_prompt = _build_soft_user_prompt(
                membership,
                persona_by_profile,
                profile_narrative_by_profile,
                respondent_context,
                questionnaire,
                prompt_version,
                dataset=dataset,
            )
        else:
            user_prompt = _build_direct_user_prompt(
                persona,
                respondent_context,
                questionnaire,
                prompt_version,
                profile_narrative=profile_narrative,
                dataset=dataset,
            )
        if self_consistency_samples <= 1:
            responses, metadata = _generate_direct_response_candidate(
                client,
                user_prompt=user_prompt,
                response_schema=response_schema,
                preset=preset,
                model_id=model_id,
                question_spec=question_spec,
                dataset=dataset,
                prompt_version=prompt_version,
            )
            selected_index = 0
        else:
            candidate_payloads: list[dict[str, int | None]] = []
            candidate_metadatas: list[RunMetadata] = []
            for _ in range(self_consistency_samples):
                candidate_responses, candidate_metadata = _generate_direct_response_candidate(
                    client,
                    user_prompt=user_prompt,
                    response_schema=response_schema,
                    preset=preset,
                    model_id=model_id,
                    question_spec=question_spec,
                    dataset=dataset,
                    prompt_version=prompt_version,
                )
                candidate_payloads.append(candidate_responses)
                candidate_metadatas.append(candidate_metadata)
            selected_index = _select_self_consistent_candidate_index(
                candidate_payloads,
                question_spec=question_spec,
            )
            responses = candidate_payloads[selected_index]
            metadata = _aggregate_run_metadata(
                candidate_metadatas,
                selected_index=selected_index,
            )

    record: dict[str, object] = {
        "respondent_id": row["respondent_id"],
        "profile_id": profile_id,
        "persona_label": persona.label,
        "run_label": run_label,
        "prompt_version": prompt_version,
        "model_id": metadata.model_id,
        "preset_name": metadata.preset_name,
        "prompt_tokens": metadata.prompt_tokens,
        "completion_tokens": metadata.completion_tokens,
        "total_tokens": metadata.total_tokens,
        "self_consistency_samples": self_consistency_samples,
        "self_consistency_selected_index": selected_index,
    }
    if selected_plan is not None:
        record["two_pass_plan_extremity_preference"] = selected_plan.extremity_preference
        record["two_pass_plan_reasoning_note"] = selected_plan.reasoning_note
        for scale_name, value in selected_plan.target_scale_means.items():
            record[f"two_pass_target_{scale_name}"] = value
    record.update(respondent_context)
    for item_id, _ in question_spec:
        record[item_id] = responses.get(item_id)
    _apply_scale_means_to_record(record, dataset=dataset)
    return record, metadata


def _simulate_single_calibrated_response(
    client: LMStudioClient,
    row: pd.Series,
    *,
    persona_by_profile: dict[int, PersonaCard],
    preset: InferencePreset,
    run_label: str,
    prompt_version: str,
    model_id: str,
    dataset: DatasetSettings,
    calibration_artifacts: CalibrationArtifacts,
    donor_pool: pd.DataFrame,
    calibration_strategy: str,
    random_state: int | None,
    self_consistency_samples: int = 1,
) -> tuple[dict[str, object], RunMetadata | None, dict[str, object]]:
    profile_id = int(row["profile_id"])
    persona = persona_by_profile[profile_id]
    respondent_context = _build_respondent_context(row, dataset=dataset)
    signal_metadata: RunMetadata | None = None
    selected_index = 0

    if calibration_strategy == "statistical_only":
        signal = neutral_calibration_signal(
            calibration_artifacts.scale_columns,
            scale_shift_aliases=calibration_artifacts.scale_shift_aliases,
            reasoning_note=(
                "İstatistiksel kalibrasyon kullanıldı."
                if dataset.locale == "tr"
                else "Statistical calibration was used."
            ),
        )
    else:
        user_prompt = "\n\n".join(
            [
                _render_persona_card(persona, dataset=dataset),
                _render_respondent_context(respondent_context, dataset=dataset),
                _calibration_signal_prompt(prompt_version=prompt_version, dataset=dataset),
            ]
        )
        calibration_schema = build_calibration_signal_schema(dataset)
        if self_consistency_samples <= 1:
            parsed_signal, signal_metadata = client.generate_structured(
                system_prompt=_calibration_signal_system_prompt(dataset),
                user_prompt=user_prompt,
                schema=calibration_schema,
                preset=preset,
                model_id=model_id,
            )
            signal = parsed_signal.to_calibration_signal()
        else:
            candidate_signals: list[CalibrationSignal] = []
            candidate_metadatas: list[RunMetadata] = []
            for _ in range(self_consistency_samples):
                parsed_signal, candidate_metadata = client.generate_structured(
                    system_prompt=_calibration_signal_system_prompt(dataset),
                    user_prompt=user_prompt,
                    schema=calibration_schema,
                    preset=preset,
                    model_id=model_id,
                )
                candidate_signals.append(parsed_signal.to_calibration_signal())
                candidate_metadatas.append(candidate_metadata)
            selected_index = _select_self_consistent_signal_index(candidate_signals)
            signal = candidate_signals[selected_index]
            signal_metadata = _aggregate_run_metadata(
                candidate_metadatas,
                selected_index=selected_index,
            )

    calibrated = decode_calibrated_response(
        row,
        signal,
        calibration_artifacts,
        donor_pool,
        random_state=_respondent_seed(
            random_state,
            respondent_id=row["respondent_id"],
            run_label=run_label,
        ),
    )
    record: dict[str, object] = {
        "respondent_id": row["respondent_id"],
        "profile_id": profile_id,
        "persona_label": persona.label,
        "run_label": run_label,
        "prompt_version": prompt_version,
        "model_id": signal_metadata.model_id if signal_metadata is not None else "statistical_only",
        "preset_name": signal_metadata.preset_name if signal_metadata is not None else "statistical_only",
        "prompt_tokens": signal_metadata.prompt_tokens if signal_metadata is not None else 0,
        "completion_tokens": signal_metadata.completion_tokens if signal_metadata is not None else 0,
        "total_tokens": signal_metadata.total_tokens if signal_metadata is not None else 0,
        "self_consistency_samples": self_consistency_samples,
        "self_consistency_selected_index": selected_index,
        "calibration_strategy": calibration_strategy,
        "calibration_donor_id": calibrated.selected_donor_id,
        "calibration_signal_extremity_preference": signal.extremity_preference,
        "calibration_reasoning_note": signal.reasoning_note,
    }
    record.update(
        {
            f"calibration_signal_{scale_name}": float(
                signal.shift_for_scale(
                    scale_name,
                    alias=calibration_artifacts.scale_shift_aliases.get(scale_name),
                )
            )
            for scale_name in calibration_artifacts.scale_columns
        }
    )
    record.update(
        {
            f"calibration_signal_{alias}_shift": float(
                signal.shift_for_scale(scale_name, alias=alias)
            )
            for scale_name, alias in calibration_artifacts.scale_shift_aliases.items()
        }
    )
    record.update(respondent_context)
    record.update(calibrated.item_responses)
    for scale_name, value in calibrated.base_scale_values.items():
        record[f"base_{scale_name}"] = value
    for scale_name, value in calibrated.target_scale_values.items():
        record[f"target_{scale_name}"] = value
    for scale_name, value in calibrated.realized_scale_values.items():
        record[scale_name] = value

    signal_record: dict[str, object] = {
        "respondent_id": row["respondent_id"],
        "profile_id": profile_id,
        "run_label": run_label,
        "calibration_strategy": calibration_strategy,
        "model_id": record["model_id"],
        "extremity_preference": signal.extremity_preference,
        "reasoning_note": signal.reasoning_note,
        "selected_donor_id": calibrated.selected_donor_id,
        "self_consistency_samples": self_consistency_samples,
        "self_consistency_selected_index": selected_index,
    }
    signal_record.update(
        {
            f"{scale_name}_shift": float(signal.scale_shifts.get(scale_name, 0.0))
            for scale_name in calibration_artifacts.scale_columns
        }
    )
    signal_record.update(
        {
            f"base_{scale_name}": value
            for scale_name, value in calibrated.base_scale_values.items()
        }
    )
    signal_record.update(
        {
            f"target_{scale_name}": value
            for scale_name, value in calibrated.target_scale_values.items()
        }
    )
    signal_record.update(
        {
            f"realized_{scale_name}": value
            for scale_name, value in calibrated.realized_scale_values.items()
        }
    )
    signal_record.update(respondent_context)
    return record, signal_metadata, signal_record


def _simulate_single_latent_logprob_response(
    client: LMStudioClient,
    row: pd.Series,
    *,
    persona_by_profile: dict[int, PersonaCard],
    preset: InferencePreset,
    ordinal_preset: InferencePreset,
    run_label: str,
    prompt_version: str,
    item_scoring_mode: str,
    model_id: str,
    dataset: DatasetSettings,
    question_spec: list[tuple[str, str]],
    trait_plan_samples: int = 1,
) -> tuple[dict[str, object], RunMetadata]:
    profile_id = int(row["profile_id"])
    persona = persona_by_profile[profile_id]
    respondent_context = _build_respondent_context(row, dataset=dataset)
    plan_user_prompt = _build_latent_plan_user_prompt(
        persona,
        respondent_context,
        dataset=dataset,
    )

    if trait_plan_samples <= 1:
        latent_plan, plan_metadata = _generate_latent_plan_candidate(
            client,
            user_prompt=plan_user_prompt,
            preset=preset,
            model_id=model_id,
            dataset=dataset,
        )
        selected_plan_index = 0
    else:
        candidate_plans: list[LatentTraitPlan] = []
        candidate_metadatas: list[RunMetadata] = []
        for _ in range(trait_plan_samples):
            candidate_plan, candidate_metadata = _generate_latent_plan_candidate(
                client,
                user_prompt=plan_user_prompt,
                preset=preset,
                model_id=model_id,
                dataset=dataset,
            )
            candidate_plans.append(candidate_plan)
            candidate_metadatas.append(candidate_metadata)
        selected_plan_index = select_self_consistent_plan_index(candidate_plans)
        latent_plan = candidate_plans[selected_plan_index]
        plan_metadata = _aggregate_run_metadata(
            candidate_metadatas,
            selected_index=selected_plan_index,
        )

    respondent_context_block = _render_respondent_context(respondent_context, dataset=dataset)
    latent_plan_block = render_latent_trait_plan(latent_plan)
    item_results = [
        score_item_with_latent_plan(
            client,
            persona_card_block=_render_persona_card(persona, dataset=dataset),
            respondent_context_block=respondent_context_block,
            latent_plan_block=latent_plan_block,
            item_id=item_id,
            item_text=question_text,
            preset=ordinal_preset,
            item_scoring_mode=item_scoring_mode,
            model_id=model_id,
        )
        for item_id, question_text in question_spec
    ]
    item_prompt_tokens = sum(result.metadata.prompt_tokens for result in item_results)
    item_completion_tokens = sum(result.metadata.completion_tokens for result in item_results)
    item_total_tokens = sum(result.metadata.total_tokens for result in item_results)
    metadata = RunMetadata(
        model_id=model_id,
        preset_name=f"{preset.name}+{ordinal_preset.name}",
        prompt_tokens=plan_metadata.prompt_tokens + item_prompt_tokens,
        completion_tokens=plan_metadata.completion_tokens + item_completion_tokens,
        total_tokens=plan_metadata.total_tokens + item_total_tokens,
        response_id=plan_metadata.response_id,
    )

    record: dict[str, object] = {
        "respondent_id": row["respondent_id"],
        "profile_id": profile_id,
        "persona_label": persona.label,
        "run_label": run_label,
        "prompt_version": prompt_version,
        "model_id": metadata.model_id,
        "preset_name": metadata.preset_name,
        "prompt_tokens": metadata.prompt_tokens,
        "completion_tokens": metadata.completion_tokens,
        "total_tokens": metadata.total_tokens,
        "response_mode": "latent_logprob",
        "item_scoring_mode": item_scoring_mode,
        "trait_plan_samples": trait_plan_samples,
        "trait_plan_selected_index": selected_plan_index,
        "latent_plan_digital_level": latent_plan.digital_level,
        "latent_plan_social_level": latent_plan.social_level,
        "latent_plan_wellbeing_level": latent_plan.wellbeing_level,
        "latent_plan_extremity_style": latent_plan.extremity_style,
        "latent_plan_midpoint_tendency": latent_plan.midpoint_tendency,
        "latent_plan_acquiescence_tendency": latent_plan.acquiescence_tendency,
        "latent_plan_reasoning_note": latent_plan.reasoning_note,
        "mean_item_entropy": pd.Series(
            [result.entropy for result in item_results],
            dtype="float64",
        ).mean(),
        "mean_item_top1_probability": pd.Series(
            [result.top1_probability for result in item_results],
            dtype="float64",
        ).mean(),
    }
    record.update(respondent_context)
    for result in item_results:
        record[result.item_id] = result.score
    _apply_scale_means_to_record(record, dataset=dataset)
    return record, metadata


def simulate_survey_responses(
    client: LMStudioClient,
    holdout: pd.DataFrame,
    persona_cards: Iterable[PersonaCard],
    preset: InferencePreset,
    *,
    max_rows: int | None = None,
    run_label: str = "primary",
    prompt_version: str = "v1",
    parallel_workers: int = 1,
    self_consistency_samples: int = 1,
    response_mode: str = "direct_json",
    item_scoring_mode: str = "argmax",
    trait_plan_samples: int = 1,
    ordinal_preset: InferencePreset | None = None,
    profile_narratives: Iterable[dict[str, object]] | None = None,
    calibration_artifacts: CalibrationArtifacts | None = None,
    donor_pool: pd.DataFrame | None = None,
    calibration_strategy: str = "direct",
    random_state: int | None = None,
    save_raw_outputs: bool = True,
    dataset: DatasetSettings = DEFAULT_DATASET,
) -> tuple[pd.DataFrame, list[RunMetadata]]:
    persona_by_profile = {card.profile_id: card for card in persona_cards}
    profile_narrative_by_profile = _build_profile_narrative_lookup(profile_narratives)
    selected = holdout.head(max_rows) if max_rows is not None else holdout
    question_spec = list(dataset.question_spec)
    use_calibrated_path = (
        calibration_strategy != "direct"
        and calibration_artifacts is not None
        and donor_pool is not None
    )
    use_latent_logprob_path = response_mode == "latent_logprob"
    if use_latent_logprob_path and use_calibrated_path:
        raise ValueError("latent_logprob mode does not support calibration strategies.")
    if use_latent_logprob_path and ordinal_preset is None:
        raise ValueError("ordinal_preset is required when response_mode='latent_logprob'.")
    if use_latent_logprob_path and dataset.dataset_id != DEFAULT_DATASET.dataset_id:
        raise ValueError("latent_logprob mode is currently only supported for the default dataset.")
    if prompt_version in {"direct_stats_v1", "direct_stats_two_pass_v1"} and (
        use_calibrated_path or use_latent_logprob_path
    ):
        raise ValueError(
            f"{prompt_version} is only supported with response_mode='direct_json' and calibration_strategy='direct'."
        )

    if any(str(c).startswith(MEMBERSHIP_PREFIX) for c in selected.columns) and (
        use_calibrated_path
        or use_latent_logprob_path
        or prompt_version not in SOFT_MEMBERSHIP_PROMPTS
    ):
        raise ValueError(
            f"soft membership is only defined for direct generation with {sorted(SOFT_MEMBERSHIP_PROMPTS)}."
        )

    questionnaire = None
    response_schema = None
    if not use_calibrated_path and not use_latent_logprob_path:
        questionnaire = _questionnaire_prompt(prompt_version=prompt_version, dataset=dataset)
        response_schema = build_survey_response_schema(
            tuple(item_id for item_id, _ in question_spec),
            response_min=dataset.response_min,
            response_max=dataset.response_max,
        )
    model_id = client.discover_loaded_model().id
    rows = [row for _, row in selected.iterrows()]
    if prompt_version in {"direct_stats_v1", "direct_stats_two_pass_v1"}:
        required_profile_ids = {int(row["profile_id"]) for row in rows}
        missing_profile_ids = sorted(
            profile_id
            for profile_id in required_profile_ids
            if profile_id not in profile_narrative_by_profile
        )
        if missing_profile_ids:
            raise ValueError(
                f"{prompt_version} requires profile_narratives for every profile_id."
            )

    with make_progress(
        total=len(rows),
        desc=f"Simulating {run_label} responses",
        unit="resp",
        leave=False,
    ) as progress:
        if parallel_workers <= 1 or len(rows) <= 1:
            results = []
            for row in rows:
                if use_calibrated_path:
                    results.append(
                        _simulate_single_calibrated_response(
                            client,
                            row,
                            persona_by_profile=persona_by_profile,
                            preset=preset,
                            run_label=run_label,
                            prompt_version=prompt_version,
                            model_id=model_id,
                        dataset=dataset,
                            calibration_artifacts=calibration_artifacts,
                            donor_pool=donor_pool,
                            calibration_strategy=calibration_strategy,
                            random_state=random_state,
                            self_consistency_samples=self_consistency_samples,
                        )
                    )
                elif use_latent_logprob_path:
                    direct_record, metadata = _simulate_single_latent_logprob_response(
                        client,
                        row,
                        persona_by_profile=persona_by_profile,
                        preset=preset,
                        ordinal_preset=cast(InferencePreset, ordinal_preset),
                        run_label=run_label,
                        prompt_version=prompt_version,
                        item_scoring_mode=item_scoring_mode,
                        model_id=model_id,
                        dataset=dataset,
                        question_spec=question_spec,
                        trait_plan_samples=trait_plan_samples,
                    )
                    results.append((direct_record, metadata, None))
                else:
                    direct_record, metadata = _simulate_single_response(
                        client,
                        row,
                        persona_by_profile=persona_by_profile,
                        profile_narrative_by_profile=profile_narrative_by_profile,
                        questionnaire=cast(str, questionnaire),
                        response_schema=cast(type[BaseModel], response_schema),
                        preset=preset,
                        run_label=run_label,
                        prompt_version=prompt_version,
                        model_id=model_id,
                        dataset=dataset,
                        question_spec=question_spec,
                        self_consistency_samples=self_consistency_samples,
                    )
                    results.append((direct_record, metadata, None))
                progress.update(1)
        else:
            worker_count = min(parallel_workers, len(rows))
            ordered_results: list[
                tuple[dict[str, object], RunMetadata | None, dict[str, object] | None] | None
            ] = [None] * len(rows)
            worker_function = (
                _simulate_single_calibrated_response
                if use_calibrated_path
                else (
                    _simulate_single_latent_logprob_response
                    if use_latent_logprob_path
                    else _simulate_single_response
                )
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_index = {
                    executor.submit(
                        worker_function,
                        client,
                        row,
                        **(
                            {
                                "persona_by_profile": persona_by_profile,
                                "preset": preset,
                                "run_label": run_label,
                                "prompt_version": prompt_version,
                                "model_id": model_id,
                                "dataset": dataset,
                                "calibration_artifacts": calibration_artifacts,
                                "donor_pool": donor_pool,
                                "calibration_strategy": calibration_strategy,
                                "random_state": random_state,
                                "self_consistency_samples": self_consistency_samples,
                            }
                            if use_calibrated_path
                            else (
                                {
                                    "persona_by_profile": persona_by_profile,
                                    "preset": preset,
                                    "ordinal_preset": cast(InferencePreset, ordinal_preset),
                                    "run_label": run_label,
                                    "prompt_version": prompt_version,
                                    "item_scoring_mode": item_scoring_mode,
                                    "model_id": model_id,
                                    "dataset": dataset,
                                    "question_spec": question_spec,
                                    "trait_plan_samples": trait_plan_samples,
                                }
                                if use_latent_logprob_path
                                else {
                                    "persona_by_profile": persona_by_profile,
                                    "profile_narrative_by_profile": profile_narrative_by_profile,
                                    "questionnaire": cast(str, questionnaire),
                                    "response_schema": cast(type[BaseModel], response_schema),
                                    "preset": preset,
                                    "run_label": run_label,
                                    "prompt_version": prompt_version,
                                    "model_id": model_id,
                                    "dataset": dataset,
                                    "question_spec": question_spec,
                                    "self_consistency_samples": self_consistency_samples,
                                }
                            )
                        ),
                    ): index
                    for index, row in enumerate(rows)
                }
                for future in as_completed(future_to_index):
                    future_result = future.result()
                    if use_calibrated_path:
                        ordered_results[future_to_index[future]] = future_result
                    else:
                        direct_record, metadata = future_result
                        ordered_results[future_to_index[future]] = (
                            direct_record,
                            metadata,
                            None,
                        )
                    progress.update(1)
            results = [result for result in ordered_results if result is not None]

    records = [record for record, _, _ in results]
    metadata_records = [metadata for _, metadata, _ in results if metadata is not None]
    signal_records = [signal_record for _, _, signal_record in results if signal_record is not None]

    frame = pd.DataFrame(records)
    if save_raw_outputs and signal_records:
        frame.attrs["calibration_signals"] = signal_records
    frame.attrs["calibration_strategy"] = calibration_strategy
    return frame, metadata_records
