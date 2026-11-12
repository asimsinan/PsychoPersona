from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from .lmstudio import InferencePreset, LMStudioClient, RunMetadata


TraitLevel = Literal["very_low", "low", "medium", "high", "very_high"]
StyleLevel = Literal["low", "medium", "high"]

_TRAIT_LEVEL_ORDER = {
    "very_low": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "very_high": 4,
}
_STYLE_LEVEL_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


class LatentTraitPlan(BaseModel):
    digital_level: TraitLevel
    social_level: TraitLevel
    wellbeing_level: TraitLevel
    extremity_style: StyleLevel
    midpoint_tendency: StyleLevel
    acquiescence_tendency: StyleLevel
    salient_cues: list[str]
    reasoning_note: str | None = None


@dataclass(slots=True)
class ItemScoreResult:
    item_id: str
    score: int
    option_probabilities: dict[str, float]
    entropy: float
    top1_probability: float
    chosen_option: str
    metadata: RunMetadata


def build_tune_f_plan_prompt(
    *,
    persona_card_block: str,
    respondent_context_block: str,
    first_person_backstory: str,
) -> str:
    instructions = [
        "Görev: Aşağıdaki persona kartı, katılımcı bağlamı ve kısa birinci tekil backstory'ye göre kompakt bir latent plan üret.",
        "Bu aşamada madde yanıtı üretme; yalnızca latent plan alanlarını doldur.",
        "Informative parsimony uygula: yalnızca gerçekten ayırt edici ipuçlarını kullan, gereksiz süslü açıklama ekleme.",
        "Trait düzeylerini beşli anchor mantığıyla seç: çok düşük, düşük, orta, yüksek, çok yüksek.",
        "Yanıt stili alanlarını üç düzeyde seç: low, medium, high.",
        "salient_cues alanına yalnızca en belirleyici 2-4 bağlamsal ipucunu yaz.",
        "",
        "Dijital bağımlılık anchorları:",
        "- çok düşük: kullanımını rahat yöneten, bırakmakta zorlanmayan, günlük işlevi belirgin bozulmayan örüntü.",
        "- düşük: çoğu durumda kontrollü kullanım, ara sıra dikkat dağılması olsa da baskın sorun olmayan örüntü.",
        "- orta: sık kullanım ile özdenetim arasında karışık sinyaller veren dengeli örüntü.",
        "- yüksek: kullanım nedeniyle dikkat, zaman yönetimi veya kontrol kaybı belirginleşen örüntü.",
        "- çok yüksek: bırakmakta çok zorlanan ve gündelik işlevi belirgin bozulan örüntü.",
        "",
        "Sosyal kaygı anchorları:",
        "- çok düşük: sosyal değerlendirme baskısı çok zayıf, çoğu durumda rahat örüntü.",
        "- düşük: sosyal durumlarda genel olarak rahat ama bazı anlarda hafif gerilim yaşayabilen örüntü.",
        "- orta: sosyal gerilim ile rahatlık arasında karışık sinyaller taşıyan örüntü.",
        "- yüksek: değerlendirilme korkusu ve çekingenlik belirgin olan örüntü.",
        "- çok yüksek: olumsuz yargılanma korkusu ve kaçınma eğilimi çok güçlü örüntü.",
        "",
        "İyi oluş anchorları:",
        "- çok düşük: enerji, amaçlılık ve olumlu duygu belirgin biçimde zayıf örüntü.",
        "- düşük: iyi oluş sinyalleri sınırlı, olumlu duygu ve işlevsellik daha kırılgan örüntü.",
        "- orta: olumlu ve olumsuz yönleri karışık, dengeli ama sınırlı iyi oluş örüntüsü.",
        "- yüksek: çoğu göstergede olumlu duygu, amaçlılık ve dayanıklılık daha güçlü örüntü.",
        "- çok yüksek: çok güçlü olumlu duygu, amaçlılık ve psikolojik dayanıklılık örüntüsü.",
        "",
        "Yanıt stili alanları:",
        "- extremity_style: uç seçeneklere gitme eğilimi.",
        "- midpoint_tendency: orta seçeneklerde kalma eğilimi.",
        "- acquiescence_tendency: maddelere daha onaylayıcı yaklaşma eğilimi.",
    ]
    return "\n\n".join(
        [
            persona_card_block,
            respondent_context_block,
            first_person_backstory,
            "\n".join(instructions),
        ]
    )


def render_latent_trait_plan(plan: LatentTraitPlan) -> str:
    lines = [
        "Latent plan:",
        f"- digital_level: {plan.digital_level}",
        f"- social_level: {plan.social_level}",
        f"- wellbeing_level: {plan.wellbeing_level}",
        f"- extremity_style: {plan.extremity_style}",
        f"- midpoint_tendency: {plan.midpoint_tendency}",
        f"- acquiescence_tendency: {plan.acquiescence_tendency}",
        "- salient_cues:",
    ]
    for cue in plan.salient_cues:
        lines.append(f"  - {cue}")
    if plan.reasoning_note:
        lines.append(f"- reasoning_note: {plan.reasoning_note}")
    return "\n".join(lines)


def build_tune_f_item_prompt(
    *,
    persona_card_block: str,
    respondent_context_block: str,
    latent_plan_block: str,
    item_id: str,
    item_text: str,
) -> str:
    instructions = [
        "Aşağıda yalnızca tek bir anket maddesi vardır.",
        f"Madde kimliği: {item_id}",
        f"Madde metni: {item_text}",
        "Latent planı bu maddeye uygula ve yalnızca 1, 2, 3, 4 veya 5 döndür.",
        "Başka hiçbir açıklama, JSON, cümle veya işaret ekleme.",
    ]
    return "\n\n".join(
        [
            persona_card_block,
            respondent_context_block,
            latent_plan_block,
            "\n".join(instructions),
        ]
    )


def argmax_score(option_probabilities: dict[str, float]) -> int:
    if not option_probabilities:
        raise ValueError("option_probabilities cannot be empty.")
    best_choice = min(
        option_probabilities,
        key=lambda choice: (-option_probabilities[choice], int(choice)),
    )
    return int(best_choice)


def posterior_mean_round(option_probabilities: dict[str, float]) -> int:
    if not option_probabilities:
        raise ValueError("option_probabilities cannot be empty.")
    posterior_mean = sum(int(choice) * probability for choice, probability in option_probabilities.items())
    rounded = math.floor(posterior_mean + 0.5)
    return max(1, min(5, rounded))


def probability_entropy(option_probabilities: dict[str, float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in option_probabilities.values()
        if probability > 0.0
    )


def top1_probability(option_probabilities: dict[str, float]) -> float:
    if not option_probabilities:
        return 0.0
    return max(option_probabilities.values())


def score_item_with_latent_plan(
    client: LMStudioClient,
    *,
    persona_card_block: str,
    respondent_context_block: str,
    latent_plan_block: str,
    item_id: str,
    item_text: str,
    preset: InferencePreset,
    item_scoring_mode: Literal["argmax", "posterior_mean_round"] = "posterior_mean_round",
    model_id: str | None = None,
) -> ItemScoreResult:
    option_probabilities, metadata, chosen_option = client.score_discrete_choices(
        system_prompt=(
            "Sen tek maddelik Likert yanıtı üreten dikkatli bir araştırma asistanısın. "
            "Yalnızca 1, 2, 3, 4 veya 5 döndür."
        ),
        user_prompt=build_tune_f_item_prompt(
            persona_card_block=persona_card_block,
            respondent_context_block=respondent_context_block,
            latent_plan_block=latent_plan_block,
            item_id=item_id,
            item_text=item_text,
        ),
        choices=["1", "2", "3", "4", "5"],
        preset=preset,
        model_id=model_id,
    )
    if item_scoring_mode == "argmax":
        score = argmax_score(option_probabilities)
    else:
        score = posterior_mean_round(option_probabilities)
    return ItemScoreResult(
        item_id=item_id,
        score=score,
        option_probabilities=option_probabilities,
        entropy=probability_entropy(option_probabilities),
        top1_probability=top1_probability(option_probabilities),
        chosen_option=chosen_option,
        metadata=metadata,
    )


def _plan_distance(left: LatentTraitPlan, right: LatentTraitPlan) -> int:
    return (
        abs(_TRAIT_LEVEL_ORDER[left.digital_level] - _TRAIT_LEVEL_ORDER[right.digital_level])
        + abs(_TRAIT_LEVEL_ORDER[left.social_level] - _TRAIT_LEVEL_ORDER[right.social_level])
        + abs(_TRAIT_LEVEL_ORDER[left.wellbeing_level] - _TRAIT_LEVEL_ORDER[right.wellbeing_level])
        + abs(_STYLE_LEVEL_ORDER[left.extremity_style] - _STYLE_LEVEL_ORDER[right.extremity_style])
        + abs(_STYLE_LEVEL_ORDER[left.midpoint_tendency] - _STYLE_LEVEL_ORDER[right.midpoint_tendency])
        + abs(
            _STYLE_LEVEL_ORDER[left.acquiescence_tendency]
            - _STYLE_LEVEL_ORDER[right.acquiescence_tendency]
        )
    )


def select_self_consistent_plan_index(plans: list[LatentTraitPlan]) -> int:
    if len(plans) <= 1:
        return 0
    plan_scores = []
    for index, plan in enumerate(plans):
        score = sum(
            _plan_distance(plan, other)
            for other_index, other in enumerate(plans)
            if other_index != index
        )
        plan_scores.append(score)
    return min(range(len(plan_scores)), key=lambda index: (plan_scores[index], index))
