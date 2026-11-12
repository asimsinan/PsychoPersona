# PsychoPersona

**A held-out benchmark for testing whether LLM personas can reproduce human psychometric survey responses.**

PsychoPersona loads real survey data, builds respondent profiles, asks a language model to simulate held-out answers, and scores how closely those synthetic responses match human structure—not just whether they *sound* plausible.

---

## What this project does

1. **Loads** human survey responses (Likert-style items and scale scores).
2. **Splits** data into train / validation / final test (final test stays locked until you freeze a method).
3. **Builds** empirical respondent profiles from the training split.
4. **Generates** persona cards and simulated survey answers via an LLM (local LM Studio or OpenRouter).
5. **Evaluates** synthetic vs. human data on psychometric metrics (correlations, profile fit, respondent-level error, stability, diversity).

The main question is not “Can an LLM fill out a form?” but “Does the synthetic data preserve the measurement structure of held-out humans?”

---

## What this repository contains

| Part | Content |
| --- | --- |
| `src/`, `scripts/`, `configs/` | The benchmark code, the analysis scripts and every run configuration. |
| `results/runs/experiment/` | 890 archived runs. Each has its manifest, persona cards, synthetic responses, evaluation report and stability sweeps. |
| `results/analysis/` | The aggregate files behind the tables and figures of the paper. |
| `datasets/anket.xlsx` | The Turkish survey, de-identified at collection. |

Three things are left out on purpose.

- **Records of human respondents inside the runs.** A run directory holds no answer and no
  context field of a respondent, only the respondent identifier and the assigned profile.
  `scripts/restore_run_data.py` rebuilds these files from the survey files (see
  [Reproducing the analysis](#reproducing-the-analysis-without-api-cost)).
- **The Bangladesh survey file.** It is supplementary material of its source article
  (Islam et al., 2021, *Journal of Community Psychology*, 49(7), 2276-2297,
  doi 10.1002/jcop.22675) and is not redistributed. Download the supplementary spreadsheet
  from the article and save it as `datasets/bangladesh_sns/mmc2.xlsx`. For Bangladesh runs
  the respondent context next to each synthetic record is removed as well.
- **The raw request and response archive of the language models** (about 350 MB). Every
  archived run can be regenerated from its configuration.

---



## Quick start

Run all commands from the **project root** (this directory).

### 1. Install

Requires [uv](https://docs.astral.sh/uv/) and **Python 3.13+**.

```bash
uv sync
```



### 2. Choose an LLM backend


| Backend                 | Best for                    | Config example                                                |
| ----------------------- | --------------------------- | ------------------------------------------------------------- |
| **LM Studio** (local)   | Development, no API cost    | `configs/runs/paper_validation_baseline.yaml`                 |
| **OpenRouter** (hosted) | GPT, Claude, Gemini via API | `configs/runs/paper_validation_openrouter_gpt54_tuned_b.yaml` |
| **OpenRouter** (open-weight) | Qwen3.8 27B, DeepSeek V4.1 Flash on one pinned host | `configs/runs/r2/r2_tr_qwen38_27b_sadp_s104.yaml` |


**Local (LM Studio):** start LM Studio, load a model, enable the local server (default `http://127.0.0.1:1234/v1`), then:

```bash
bash scripts/run_pipeline.sh configs/runs/paper_validation_baseline.yaml
```

**Hosted (OpenRouter):**

```bash
export OPENROUTER_API_KEY="your-key-here"
bash scripts/run_pipeline.sh configs/runs/paper_validation_openrouter_gpt54_tuned_b.yaml
```



### 3. Run a full experiment

```bash
bash scripts/run_pipeline.sh configs/runs/paper_validation_openrouter_gpt54_tuned_b.yaml
```

The script prints the run directory under `results/runs/experiment/`. Each run includes metrics, plots, persona cards, and a full LLM request archive.

To resume after an interruption:

```bash
bash scripts/run_pipeline.sh configs/runs/your_config.yaml results/runs/experiment/<timestamp>_experiment
```

---



## How the pipeline works

```text
Raw survey (Excel)
       │
       ▼
 Score items → train / validation / final_test split
       │
       ├── Audit (baseline psychometrics on human data)
       │
       ├── Fit profiles (clustering on train)
       │
       ├── Generate persona cards (LLM)
       │
       ├── Simulate survey responses (LLM, held-out rows)
       │
       ├── Evaluate vs. human holdout
       │
       └── Stability sweeps (repeatability)
```

**Split discipline (important):**

- `validation` — compare methods, tune prompts, inspect failures.
- `final_test` — run only after you lock a configuration; do not peek while choosing methods.

**Profile assignment (important):** every profile-conditioned method can be run under two
settings, set by `profiles.assignment` in the config.


| Setting            | Meaning                                                                   | What the numbers mean                                                                              |
| ------------------ | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `oracle` (default) | Held-out respondents are assigned to profiles using their own scale means | An upper bound. Profile membership is derived from the answers the simulation is meant to produce. |
| `coldstart`        | Membership is predicted from pre-response metadata only                   | What a deployed study could actually obtain, since no responses exist yet.                         |


Report both. An oracle-only result overstates what the method delivers in practice, and the
gap between the two settings is itself a finding.

**Profile construction:** two further options change how the profile layer is built. They
exist to test how far the results depend on it.


| Option                | Values                    | Meaning                                                                                                                                  |
| --------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `profiles.method`     | `gmm` (default), `kmeans` | Clustering used to form the profiles.                                                                                                    |
| `profiles.membership` | `hard` (default), `soft`  | `soft` passes the classifier's membership probabilities to the prompt in place of one predicted profile. Allowed with `coldstart` only. |


`soft` is refused under `oracle` assignment, because posterior probabilities are computed
from the held-out respondent's own answers and would leak more of them than a hard label.

---



## Project layout

```text
configs/
├── default.yaml          # Base settings (paths, splits, LM presets)
└── runs/                 # Run configs (validation + final_test)
    └── r2/               # Repeated splits, open-weight models, alternative profiles
src/saykopersona/         # Python package
├── data.py               # Loading, scoring, splits
├── profiles.py           # Empirical profile clustering
├── coldstart.py          # Metadata-only profile prediction (cold-start setting)
├── simulation.py         # Persona cards + survey simulation
├── evaluation.py         # Psychometric and distributional metrics
├── calibration.py        # Hybrid calibration (LLM + statistical decoder)
├── audit.py              # Human-data psychometrics
├── pipeline.py           # End-to-end run
└── experiment.py         # Resumable step-by-step runner
scripts/                  # Pipeline wrappers and analysis
datasets/                 # Survey files (the Bangladesh file is added by the reader)
results/
├── runs/experiment/      # Per-run outputs (archived runs, and new ones when you execute)
└── analysis/             # Aggregated CSVs backing the paper's tables
```

Survey data paths are set in YAML configs.

- Turkish dataset: `datasets/anket.xlsx`, de-identified at collection and included here.
- Bangladesh dataset: `datasets/bangladesh_sns/mmc2.xlsx` (set `dataset: bangladesh_sns`
in config). Obtain it from the source article and place it at that path before running
Bangladesh configs.

Configurations written before the survey files moved into `datasets/` still name the old
location. The loader finds the file of the same name in `datasets/`.

---



## Simulation methods


| Label                               | Abbrev.   | Idea                                          | Selector                                                                              |
| ----------------------------------- | --------- | --------------------------------------------- | ------------------------------------------------------------------------------------- |
| Demographic-Only Prompting          | **DOP**   | Demographics only, no persona card            | `prompt_version: demographic_only_v1`                                                 |
| Balanced Profile Prompting          | **BPP**   | Standard portable persona prompt              | `prompt_version: v3_tr_gpt_b`                                                         |
| First-Person Narrative Prompting    | **FPNP**  | First-person grounding variant                | `prompt_version: v3_tr_gpt_d`                                                         |
| Self-Consistency Prompting          | **SCP**   | Medoid-style self-consistency                 | `self_consistency_samples: 3`                                                         |
| Bias-Balanced Prompting             | **BBP**   | Model-specific bias correction                | `prompt_version: v3_tr_gpt_e` (GPT, Gemini) `prompt_version: v3_tr_claude_a` (Claude) |
| Statistics-Aware Direct Prompting   | **SADP**  | Adds compact profile statistics to the prompt | `prompt_version: direct_stats_v1`                                                     |
| Two-Pass Statistics-Aware Prompting | **TPSAP** | Plan scales first, then fill items            | `prompt_version: direct_stats_two_pass_v1`                                            |
| Hybrid Calibration                  | **HC**    | LLM signal + statistical decoder              | `prompt_version: v4_tr_hybrid_b` + `calibration.enabled: true`                        |


SCP is the one arm selected by a config knob rather than a `prompt_version`. BBP uses
two selectors because its response-band correction is model-specific.

Other `prompt_version` values in the code (`v3_tr_gpt_c`, `v3_tr_gpt_g`, and the legacy
`v1`, `v1_compact`, `v2`, `v2_tr`, `v3_tr_gpt`, `v4_tr_hybrid`) are development-only
variants and are not part of the benchmark.

See the module docstring in `src/saykopersona/simulation.py` for the full mapping,
including a note on how persona-card serialization differs for BPP.

---



## Evaluation metrics


| Metric                           | Lower or higher is better? | Meaning                                      |
| -------------------------------- | -------------------------- | -------------------------------------------- |
| Correlation RMSE                 | Lower                      | Scale–scale correlation structure vs. humans |
| Within-profile RMSE              | Lower                      | Fit inside respondent profile groups         |
| Profile recovery                 | Higher                     | Correct profile assignment                   |
| Unique response ratio            | Higher                     | Response diversity (not collapsed modes)     |
| Stability correlation            | Higher                     | Repeatability across stability runs          |
| Respondent-scale RMSE / item MAE | Lower                      | Closeness to individual held-out respondents |


Group-level distributional metrics, computed per item and averaged (the target used by
SimBench and similar benchmarks):


| Metric      | Lower or higher is better? | Meaning                                                                 |
| ----------- | -------------------------- | ----------------------------------------------------------------------- |
| `TVD`       | Lower                      | Total variation distance between response histograms                    |
| `JSD_bits`  | Lower                      | Jensen–Shannon divergence, in bits                                      |
| `WD_norm`   | Lower                      | 1-Wasserstein distance, normalized by scale width (respects ordinality) |
| `CoverageR` | Higher                     | Share of human-used response categories the synthetic data uses         |


Treat respondent-level scores as **approximation**, not exact reconstruction of human item vectors.
A method can lead on the distributional metrics and trail on the respondent-level ones, so
report more than one target.

### Measurement properties, subgroups and response style

Agreement on scores does not show that synthetic records measure what the instrument
measures. Four offline analyses compare the human and the synthetic answers of the same
respondents, pooled over the repeated splits.


| Analysis               | What it reports                                                                                   | Script                             |
| ---------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Item and scale         | Cronbach's alpha, inter-item and item-rest correlations, unused categories, differential item functioning | `r2_psychometric_properties.py`    |
| Measurement invariance | Configural, metric and scalar models per scale, with random human halves as the reference         | `r2_invariance.R` (lavaan, semTools) |
| Subgroup fidelity      | Item MAE by demographic subgroup against a permutation reference                                  | `r2_subgroup_fidelity.py`          |
| Response style         | Use of the lowest, highest and middle categories, entropy, and the human-to-human reference       | `r2_response_style.py`             |


### Non-LLM baselines

Twelve baselines run under the same held-out protocol, as calibration anchors rather than as
competing methods. Each is matched to the information an LLM method receives.


| Information          | Baselines                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| Nothing              | `marginal`, graded response model, Gaussian copula                                                     |
| Profile              | `profile_center` (per-profile mean), `within_profile`, graded response model, Gaussian copula         |
| Metadata             | `nn_metadata`, proportional-odds regression (sampled and median)                                       |
| Profile and metadata | proportional-odds regression (sampled and median)                                                      |


```bash
# the four original baselines against one run
PYTHONPATH=src python scripts/compute_nonllm_baselines.py \
  --run-dir results/runs/experiment/<id> --dataset anket_tr --replicates 50 --out <path>

# all twelve, on the paper split and on 50 random splits
PYTHONPATH=src:scripts python scripts/r2_stronger_baselines.py
```

The estimators are in `scripts/r2_ordinal_models.py`, and `scripts/r2_check_ordinal_models.py`
compares them with mirt, polycor and statsmodels.

If an LLM arm does not beat `profile_center` on a metric, it has shown no added value there.

---



## Commands

```bash
# Full resumable experiment (recommended)
bash scripts/run_pipeline.sh configs/runs/your_config.yaml

# Single step of a run, resumable
PYTHONPATH=src python scripts/run_experiment.py --config configs/runs/your_config.yaml
```

Create run configs from the default template:

```bash
uv run python scripts/create_paper_configs.py
```

---



## Configuration tips

Edit or copy files under `configs/runs/`. Common fields:

```yaml
dataset_path: ../../datasets/anket.xlsx
output_root: ../../results
split:
  validation_size: 0.1
  final_test_size: 0.1
  random_state: 42
lmstudio:
  base_url: https://openrouter.ai/api/v1   # or http://127.0.0.1:1234/v1
  model_id: openai/gpt-5.4
  api_key_env: OPENROUTER_API_KEY
simulation:
  prompt_version: v3_tr_gpt_b              # selects prompting method
  evaluation_split: validation             # or final_test
  parallel_workers: 4
```

For Bangladesh runs, add `dataset: bangladesh_sns` and point `dataset_path` at the Bangladesh Excel file.

Three optional OpenRouter settings fix how a model is served. The run manifest records them.

```yaml
lmstudio:
  service_tier: flex                       # provider service tier
  reasoning: {enabled: false}              # hidden reasoning on or off
  provider:                                # pin one host and one numerical precision
    only: [deepinfra]
    quantizations: [bf16]
    allow_fallbacks: false
```

The open-weight models are run with reasoning disabled. With reasoning enabled they spend the
response-length limit of the benchmark on hidden reasoning and return no answer.

---



## Reproducing the analysis without API cost

Everything except LLM generation runs offline against the archived runs in
`results/runs/`. No `OPENROUTER_API_KEY` is needed.

**Step 1. Restore the human-data files.** The archived runs are distributed without
respondent records. This command rebuilds, for every run, the scored responses, the
train, validation and final-test splits and the full profile-assignment files, using the
split settings stored in the run manifest.

```bash
PYTHONPATH=src python scripts/restore_run_data.py
```

Runs of the Bangladesh dataset are skipped until `datasets/bangladesh_sns/mmc2.xlsx` is in
place. The script compares the SHA-256 of each survey file with the one recorded in the
run manifests and skips a run when they differ. `--remove` deletes the rebuilt files again.

**Step 2. Run the analyses.** Each script writes its CSV files to `results/analysis/`, so
`git diff results/analysis` afterwards shows whether the distributed files were
reproduced. The first three commands recompute the dataset psychometrics, the bootstrap
intervals and the demographic-only summary.

```bash
export PYTHONPATH=src:scripts
python scripts/compute_psychometrics.py
python scripts/compute_uncertainty.py
python scripts/collect_dop_results.py

python scripts/build_run_index.py                # which run backs which table cell
python scripts/r2_main_grid.py                   # every metric of every main-grid run
python scripts/c2_llm_split_robustness.py        # repeated splits, all model families
python scripts/simbench_distributional.py        # group-level distributional metrics
python scripts/r2_stronger_baselines.py          # twelve non-LLM baselines, 50 splits
python scripts/r2_profile_sensitivity.py         # alternative clusterings, assignment uncertainty
python scripts/r2_alt_profile_runs.py            # k-means and soft-membership runs
python scripts/r2_psychometric_properties.py     # reliability, category use, item functioning
Rscript scripts/r2_invariance.R                  # measurement invariance (R, lavaan, semTools)
python scripts/r2_subgroup_fidelity.py           # subgroup fidelity
python scripts/r2_response_style.py              # response style, human-to-human reference
python scripts/r2_figures.py                     # figures, written to results/analysis/figures/
python scripts/r2_manuscript_tables.py           # tables, written to results/analysis/tables/
```

`r2_generation_summary.py` and `harvest_variance_and_cost.py` read token counts and costs
from the request archive, which is not distributed. Their outputs,
`r2_generation_summary.csv`, `generation_variance.csv` and `computational_cost.csv`, are
included as computed.

Only `scripts/run_pipeline.sh` spends API credit.

---



## License

Research code for academic use.