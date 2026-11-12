from __future__ import annotations

from .dataset_spec import (
    AuditReferenceSettings,
    DatasetSettings,
    FeatureDescriptorSettings,
    MetadataFieldSettings,
    RegressionPairSettings,
    ScaleSettings,
)


def _current_turkish_dataset() -> DatasetSettings:
    digital_raw_columns = [
        " [Uyumadan önce yataktayken, dijital cihazımda oyun oynar veya mesajlaşırım.]",
        " [Dijital cihazlarımı kullanamazsam sıkılırım.]",
        " [Dijital cihazları kullanmayı planlandığımdan daha çok zaman ayırıyorum.]",
        " [Dijital cihazlarımı bir yere bıraktığımda, tekrar kullanma isteği duyuyorum.]",
        " [Yemek yerken dijital cihazlarımı kullanırım.]",
        " [Sokakta yürürken dijital cihazımda oyun oynar veya mesajlaşırım.]",
        " [Birisiyle sohbet ederken bile, gözüm dijital cihazımdadır.]",
        " [Dijital cihazlarımı kullanmazsam huysuz biri olurum.]",
        " [Arkadaşlarımın sosyal medyada ne yazdığını bilmediğimde huzursuz hissederim.]",
        " [Dersler sırasında dijital cihazım üzerinden gözümü ayırmam.]",
    ]
    social_raw_columns = [
        " [Başkalarının benim hakkımda ne söyleyeceğinden endişelenirim.]",
        " [Başkalarının benden hoşlanmayacağından endişelenirim.]",
        " [Başkalarının benden hoşlanmayacağından korkarım.]",
        " [Başkalarının benim hakkımda ne düşündüğünden endişelenirim.]",
        " [Yeni insanlarla tanıştığımda tedirgin olurum.]",
        " [Tanımadığım insanların yanında utanırım.]",
        " [Çok iyi tanımadığım insanlarla konuşurken tedirgin olurum.]",
        " [Bazı insanların yanındayken tedirgin olurum.]",
        " [Başkalarından benimle bir şeyler yapmalarını istemekte zorlanırım.]",
        " [Başkaları hayır diyebilirler diye benimle bir şeyler yapmalarını teklif etmeye çekinirim (Ör: sinemaya gitmek gibi).]",
        " [Bir grup insanla beraberken durgun olurum.]",
        " [İyi tanıdığım insanların yanındayken bile utanırım.]",
    ]
    wellbeing_raw_columns = [
        "Lütfen aşağıdaki önermeleri size en uygun olacak şekilde işaretleyiniz. [Amaçlı ve anlamlı bir yaşam sürdürüyorum.]",
        "Lütfen aşağıdaki önermeleri size en uygun olacak şekilde işaretleyiniz. [Sosyal ilişkilerim destekleyici ve tatmin edicidir.]",
        "Lütfen aşağıdaki önermeleri size en uygun olacak şekilde işaretleyiniz. [Günlük aktivitelerime bağlı ve ilgiliyim.]",
        "Lütfen aşağıdaki önermeleri size en uygun olacak şekilde işaretleyiniz. [Başkalarının mutlu ve iyi olmasına aktif olarak katkıda bulunurum.]",
        "Lütfen aşağıdaki önermeleri size en uygun olacak şekilde işaretleyiniz. [Benim için önemli olan etkinliklerde yetenekli ve yeterliyim.]",
        "Lütfen aşağıdaki önermeleri size en uygun olacak şekilde işaretleyiniz. [Ben iyi bir insanım ve iyi bir hayat yaşıyorum.]",
        "Lütfen aşağıdaki önermeleri size en uygun olacak şekilde işaretleyiniz. [Geleceğim hakkında iyimserim.]",
        "Lütfen aşağıdaki önermeleri size en uygun olacak şekilde işaretleyiniz. [İnsanlar bana saygı duyar.]",
    ]
    agreement_map = {
        "Kesinlikle Katılmıyorum": 1.0,
        "Katılmıyorum": 2.0,
        "Kararsızım": 3.0,
        "Katılıyorum": 4.0,
        "Tamamen Katılıyorum": 5.0,
    }
    return DatasetSettings(
        dataset_id="anket_tr",
        locale="tr",
        file_format="xlsx",
        metadata_fields=[
            MetadataFieldSettings(
                name="timestamp",
                source_column="Zaman damgası",
                label="Zaman damgası",
                include_in_context=False,
            ),
            MetadataFieldSettings(
                name="gender",
                source_column="Cinsiyetiniz:",
                label="Cinsiyet",
            ),
            MetadataFieldSettings(
                name="age_band",
                source_column="Yaşınız",
                label="Yaş grubu",
                transform="age_band_default",
            ),
            MetadataFieldSettings(
                name="program",
                source_column="Eğitim gördüğünüz program",
                label="Program",
                transform="program_tr",
            ),
            MetadataFieldSettings(
                name="gpa_band",
                source_column="Genel Not Ortalamanız",
                label="Not ortalaması aralığı",
            ),
            MetadataFieldSettings(
                name="residence",
                source_column="Üniversite okurken kaldığınız yer",
                label="Kalınan yer",
            ),
            MetadataFieldSettings(
                name="platforms",
                source_column="Dijital araçlarla en çok hangi platformu/platformları kullanıyorsunuz? (1'den fazla seçebilirsiniz)",
                label="Sık kullanılan platformlar",
            ),
            MetadataFieldSettings(
                name="usage_band",
                source_column="Dijital medya araçları (sosyal medya, web siteleri, radyo, tv, podcast, dijital dizi/film vb.) ile günde ortalama kaç saat vakit geçiriyorsunuz?",
                label="Günlük medya kullanım aralığı",
            ),
            MetadataFieldSettings(
                name="usage_hours_midpoint",
                source_column="Dijital medya araçları (sosyal medya, web siteleri, radyo, tv, podcast, dijital dizi/film vb.) ile günde ortalama kaç saat vakit geçiriyorsunuz?",
                label="Günlük medya kullanım süresi orta noktası",
                transform="mapped_number",
                value_map={
                    "0-3 saat": 1.5,
                    "1-2 saat": 1.5,
                    "4-6 saat": 5.0,
                    "7-9 saat": 8.0,
                    "10 saat ve üzeri": 10.0,
                },
            ),
        ],
        scales=[
            ScaleSettings(
                name="digital_addiction",
                label="Dijital bağımlılık ortalaması",
                mean_column="digital_addiction_mean",
                raw_columns=digital_raw_columns,
                item_columns=[f"digital_item_{idx:02d}" for idx in range(1, 11)],
                response_map=agreement_map,
                shift_key="digital",
            ),
            ScaleSettings(
                name="social_anxiety",
                label="Sosyal kaygı ortalaması",
                mean_column="social_anxiety_mean",
                raw_columns=social_raw_columns,
                item_columns=[f"social_item_{idx:02d}" for idx in range(1, 13)],
                response_map=agreement_map,
                shift_key="social",
            ),
            ScaleSettings(
                name="wellbeing",
                label="İyi oluş ortalaması",
                mean_column="wellbeing_mean",
                raw_columns=wellbeing_raw_columns,
                item_columns=[f"wellbeing_item_{idx:02d}" for idx in range(1, 9)],
                response_map=agreement_map,
                shift_key="wellbeing",
            ),
        ],
        feature_descriptors={
            "digital_addiction_mean": FeatureDescriptorSettings(
                label="Dijital bağımlılık ortalaması",
                high_text="daha yüksek dijital kullanım",
                low_text="daha düşük dijital kullanım",
            ),
            "social_anxiety_mean": FeatureDescriptorSettings(
                label="Sosyal kaygı ortalaması",
                high_text="daha yüksek sosyal kaygı",
                low_text="daha düşük sosyal kaygı",
            ),
            "wellbeing_mean": FeatureDescriptorSettings(
                label="İyi oluş ortalaması",
                high_text="daha yüksek iyi oluş",
                low_text="daha düşük iyi oluş",
            ),
            "usage_hours_midpoint": FeatureDescriptorSettings(
                label="Günlük medya kullanım süresi orta noktası",
                high_text="daha yüksek günlük medya kullanımı",
                low_text="daha düşük günlük medya kullanımı",
            ),
        },
        profile_anchor_columns=["gender", "program", "usage_band"],
        subgroup_columns=["gender", "program", "usage_band"],
        stratify_categorical_columns=["gender", "program", "usage_band"],
        stratify_score_columns=["digital_addiction_mean", "wellbeing_mean"],
        regression_pairs=[
            RegressionPairSettings(
                name="social_anxiety_on_digital",
                x="digital_addiction_mean",
                y="social_anxiety_mean",
            ),
            RegressionPairSettings(
                name="wellbeing_on_digital",
                x="digital_addiction_mean",
                y="wellbeing_mean",
            ),
        ],
        response_min=1,
        response_max=5,
        audit_reference=AuditReferenceSettings(
            paper_sample_size=669,
            expected_category_counts={
                "usage_band": {
                    "0-3 saat": 200,
                    "4-6 saat": 352,
                    "7-9 saat": 84,
                    "10 saat ve üzeri": 33,
                },
                "gender": {"Kadın": 335, "Erkek": 334},
                "program": {"Ön lisans": 409, "Lisans": 260},
            },
            note="Original Turkish paper benchmark reference.",
        ),
    )


def _bangladesh_sns_dataset() -> DatasetSettings:
    four_point_frequency = {
        "Never": 0.0,
        "Rarely": 1.0,
        "Sometimes": 2.0,
        "Often": 3.0,
    }
    phq_map = {
        "Not at all": 0.0,
        "Several days": 1.0,
        "Half of days": 2.0,
        "More than half of the days": 2.0,
        "Half days": 2.0,
        "Nearly everyday": 3.0,
        "Nearly every day": 3.0,
    }
    return DatasetSettings(
        dataset_id="bangladesh_sns",
        locale="en",
        file_format="xlsx",
        sheet_name="Raw Data",
        metadata_fields=[
            MetadataFieldSettings(
                name="timestamp",
                source_column="Timestamp",
                label="Timestamp",
                include_in_context=False,
            ),
            MetadataFieldSettings(
                name="age_band",
                source_column="21. Please write your age in years (number).",
                label="Age band",
                transform="age_band_default",
            ),
            MetadataFieldSettings(
                name="gender",
                source_column="22. Gender",
                label="Gender",
            ),
            MetadataFieldSettings(
                name="education",
                source_column="25. Education",
                label="Education",
                transform="mapped_text",
                value_map={
                    "Primary level": "Primary",
                    "Secondary Level": "Secondary",
                    "Higher secondary level": "Higher secondary",
                    "Graduate Level": "Graduate or above",
                    "Graduate or above": "Graduate or above",
                    "Masters or above": "Graduate or above",
                },
            ),
            MetadataFieldSettings(
                name="profession",
                source_column="26. Profession",
                label="Profession",
                transform="mapped_text",
                value_map={
                    "Student": "Student",
                    "Student ": "Student",
                    "Unemployed including students": "Student / unemployed",
                    "Bekar": "Student / unemployed",
                    "No": "Student / unemployed",
                    "Private job": "Private job",
                    "Private job and self taught makeup artist": "Private job",
                    "Govt. job": "Government job",
                    "Businessman": "Business",
                    "Business+student": "Business / student",
                    "Self-employed/Entrepreneur/Business": "Business",
                    "Doctor": "Healthcare",
                    "Doctor ": "Healthcare",
                    "doctor": "Healthcare",
                    "Intern doctor": "Healthcare",
                    "Dental surgeon ": "Healthcare",
                    "Dentist ": "Healthcare",
                    "Teacher ": "Teacher",
                    "Housewife": "Homemaker",
                    "Housewife ": "Homemaker",
                    "housewife": "Homemaker",
                    "Homemaker ": "Homemaker",
                    "Housemaker": "Homemaker",
                    "Hhousewife": "Homemaker",
                    "House  wife ": "Homemaker",
                    "Author": "Other",
                    "Daily labor/driver/housemaid": "Other",
                    "Home tutor ": "Other",
                    "TA": "Other",
                    "Used to work and now searching for job.": "Other",
                    "Recently unemployed, now housewife": "Homemaker",
                    "I don't wanna do any job": "Other",
                },
            ),
            MetadataFieldSettings(
                name="monthly_income",
                source_column="27. Monthly income",
                label="Monthly income",
            ),
            MetadataFieldSettings(
                name="residence",
                source_column="28. Area of residence",
                label="Residence area",
            ),
            MetadataFieldSettings(
                name="primary_platform",
                source_column="2. Which social media account do you use usually?",
                label="Primary social platform",
                transform="mapped_text",
                value_map={
                    "Facebook": "Facebook",
                    "Imo": "Imo",
                    "imo": "Imo",
                    "Instagram": "Instagram",
                    "Snapchat": "Snapchat",
                    "Tiktok": "TikTok",
                    "Tiktok ": "TikTok",
                    "Twitter": "Twitter",
                    "WeChat": "WeChat",
                    "WhatsApp": "WhatsApp",
                },
            ),
            MetadataFieldSettings(
                name="usage_band",
                source_column="7. How much time do you spend daily in social media?",
                label="Daily social-media use",
            ),
            MetadataFieldSettings(
                name="usage_hours_midpoint",
                source_column="7. How much time do you spend daily in social media?",
                label="Daily social-media hours midpoint",
                transform="mapped_number",
                value_map={
                    "Less than 1 hour": 0.5,
                    "1-3 hours": 2.0,
                    "3-5 hours": 4.0,
                    "More than 5 hours": 6.0,
                },
            ),
        ],
        scales=[
            ScaleSettings(
                name="ucla_loneliness",
                label="UCLA-8 loneliness mean",
                mean_column="ucla_loneliness_mean",
                raw_columns=[
                    "1. In the past 30 days, do you feel lack of companionship?",
                    "2. In the past 30 days, there is no one I can turn to",
                    "3. In the past 30 days, I feel left out.",
                    "4. In the last 30 days, I feel isolated from others.",
                    "5. In the last 30 days, I am unhappy being so withdrawn.",
                    "6. In the last 30 days, people are around me but not with me.",
                    "7. In the last 30 days, I am an outgoing person.",
                    "8. In the last 30 days, I can find companionship when I want it.",
                ],
                item_columns=[f"ucla_item_{idx:02d}" for idx in range(1, 9)],
                response_map=four_point_frequency,
                reverse_item_columns=["ucla_item_07", "ucla_item_08"],
                shift_key="ucla",
            ),
            ScaleSettings(
                name="phq9",
                label="PHQ-9 depression mean",
                mean_column="phq9_mean",
                raw_columns=[
                    "1 In the last 30 days, little interest or pleasure in doing things.",
                    "2. In the last 30 days, feeling down, depressed or hopeless.",
                    "3. In the last 30 days, trouble falling or staying asleep, sleeping too much",
                    "4. In the last 30 days, Feeling tired or having little energy.",
                    "5. In the last 30 days, poor appetite or over-eating.",
                    "6. In the last 30 days, feeling bad about yourself-or that you are a failure or have let yourself or your family down.",
                    "7. In the last 30 days, trouble concentrating on things, such as reading the newspaper or watching television.",
                    "8. In the last 30 days, moving or speaking so slowly or the opposite-moving around a lot more than usual.",
                    "9. In the last 30 days, thoughts that you would be better off dead, or of hurting yourself.",
                ],
                item_columns=[f"phq9_item_{idx:02d}" for idx in range(1, 10)],
                response_map=phq_map,
                shift_key="phq9",
            ),
            ScaleSettings(
                name="gad7",
                label="GAD-7 anxiety mean",
                mean_column="gad7_mean",
                raw_columns=[
                    "1. In the last 30 days, I am feeling nervous, anxious, or on edge",
                    "2. In the last 30 days, I an not being able to stop or control worrying",
                    "3. In the last 30 days, I am worrying too much about different things.",
                    "4. In the last 30 days, I felt trouble in relaxing.",
                    "5. In the last 30 days, I am being so restless that it's hard to sit still",
                    "6. In the last 30 days, I becoming easily annoyed or irritable.",
                    "7. In the last 30 days, I am feeling afraid as if something awful might happen.",
                ],
                item_columns=[f"gad7_item_{idx:02d}" for idx in range(1, 8)],
                response_map=phq_map,
                shift_key="gad7",
            ),
        ],
        feature_descriptors={
            "ucla_loneliness_mean": FeatureDescriptorSettings(
                label="UCLA-8 loneliness mean",
                high_text="higher loneliness",
                low_text="lower loneliness",
            ),
            "phq9_mean": FeatureDescriptorSettings(
                label="PHQ-9 depression mean",
                high_text="higher depressive symptoms",
                low_text="lower depressive symptoms",
            ),
            "gad7_mean": FeatureDescriptorSettings(
                label="GAD-7 anxiety mean",
                high_text="higher anxiety symptoms",
                low_text="lower anxiety symptoms",
            ),
            "usage_hours_midpoint": FeatureDescriptorSettings(
                label="Daily social-media hours midpoint",
                high_text="heavier daily social-media use",
                low_text="lighter daily social-media use",
            ),
        },
        profile_anchor_columns=["gender", "education", "usage_band"],
        subgroup_columns=["gender", "education", "usage_band"],
        stratify_categorical_columns=["gender", "education", "usage_band"],
        stratify_score_columns=["phq9_mean", "gad7_mean"],
        regression_pairs=[
            RegressionPairSettings(
                name="gad7_on_phq9",
                x="phq9_mean",
                y="gad7_mean",
            ),
            RegressionPairSettings(
                name="ucla_on_phq9",
                x="phq9_mean",
                y="ucla_loneliness_mean",
            ),
        ],
        response_min=0,
        response_max=3,
    )


DEFAULT_DATASET_SETTINGS = _current_turkish_dataset()
BANGLADESH_SNS_DATASET_SETTINGS = _bangladesh_sns_dataset()
DATASET_PRESETS = {
    "default": DEFAULT_DATASET_SETTINGS,
    "anket_tr": DEFAULT_DATASET_SETTINGS,
    DEFAULT_DATASET_SETTINGS.dataset_id: DEFAULT_DATASET_SETTINGS,
    "bangladesh_sns": BANGLADESH_SNS_DATASET_SETTINGS,
    BANGLADESH_SNS_DATASET_SETTINGS.dataset_id: BANGLADESH_SNS_DATASET_SETTINGS,
}


def clone_dataset_settings(settings: DatasetSettings) -> DatasetSettings:
    return DatasetSettings.model_validate(settings.model_dump(mode="json"))


def default_dataset_settings() -> DatasetSettings:
    return clone_dataset_settings(DEFAULT_DATASET_SETTINGS)


def bangladesh_sns_dataset_settings() -> DatasetSettings:
    return clone_dataset_settings(BANGLADESH_SNS_DATASET_SETTINGS)


def dataset_settings_for_preset(name: str) -> DatasetSettings:
    try:
        settings = DATASET_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_PRESETS))
        raise KeyError(f"Unknown dataset preset '{name}'. Available presets: {available}") from exc
    return clone_dataset_settings(settings)

