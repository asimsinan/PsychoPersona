"""Create the run configs for the round-2 revision experiments.

Every config is derived from a config that produced a September result, and then checked
against the settings recorded in that result's run manifest. Only the fields the new
experiment is meant to change may differ: the model, the routing options, the split
seed, the stability sweep, the profile assignment, and the file paths. Anything else
differing is an error, because a silent change of prompt or decoding settings would make
the new runs incomparable with the reported ones.

Blocks
------
splits_claude, splits_gemini   DOP, BPP, SADP, HC on seeds 101-110 (Reviewer 1, comment 4)
splits_gpt_rest                FPNP, SCP, BBP, TPSAP on GPT-5.4, seeds 101-110
splits_open                    DOP, BPP, SADP, HC on the two open-weight models
open_main                      open-weight models, both splits, oracle assignment
open_coldstart                 open-weight models, BPP, SADP, HC under predicted membership
alt_profiles                   BPP and SADP on GPT-5.4 under two alternative profile
                               constructions (Reviewer 1, comment 3): k-means profiles
                               with oracle assignment, the profile count held at the
                               three the mixture selects, and soft membership under
                               cold-start assignment

Usage
-----
    PYTHONPATH=src python scripts/create_round2_configs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from saykopersona.config import load_config  # noqa: E402

RUNS = REPO / "configs" / "runs"
OUT = RUNS / "r2"
STATE = REPO / "results" / "analysis" / "r2_state"
SEEDS = range(101, 111)

PIN = lambda q: {"only": ["deepinfra"], "quantizations": [q], "allow_fallbacks": False}
FAMILIES = {
    "gpt54": {"label": "GPT-5.4", "model_id": "openai/gpt-5.4"},
    "claude_opus46": {"label": "Claude Opus 4.6", "model_id": "anthropic/claude-opus-4.6"},
    # Flex serves the same snapshot at lower priority; pinned to the one host whose Flex
    # endpoint is reliably up, so every Gemini run takes the same serving path.
    "gemini31pro": {"label": "Gemini 3.1 Pro", "model_id": "google/gemini-3.1-pro-preview",
                    "service_tier": "flex",
                    "provider": {"only": ["google-ai-studio"], "allow_fallbacks": False}},
    "qwen38_27b": {"label": "GPT-5.4", "model_id": "qwen/qwen3.8-27b",
                   "reasoning": {"enabled": False}, "provider": PIN("bf16")},
    "deepseek_v41_flash": {"label": "GPT-5.4", "model_id": "deepseek/deepseek-v4.1-flash-1017",
                           "reasoning": {"enabled": False}, "provider": PIN("fp8")},
}
OPEN = ("qwen38_27b", "deepseek_v41_flash")
DATASETS = {"tr": "anket_tr", "bd": "bangladesh_sns"}

# arm -> main-grid file stem (Turkish, Bangladesh)
MAIN = {
    "dop": ("demographic_only_v1", "bangladesh_demographic_only_v1"),
    "bpp": ("tuned_b", "bangladesh_tuned_b"),
    "sadp": ("direct_stats_v1", "bangladesh_direct_stats_v1"),
    "hc": ("hybrid_tuned_b", "bangladesh_hybrid_tuned_b"),
    "fpnp": ("fpnp", "bangladesh_fpnp"),
    "scp": ("scp", "bangladesh_scp"),
    "bbp": ("bbp", "bangladesh_bbp"),
    "tpsap": ("direct_stats_two_pass_v1", "bangladesh_tpsap"),
}

# Fields a round-2 config is allowed to change relative to the September run.
FREE = {"dataset_path", "paper_path", "output_root", "lmstudio.model_id",
        "lmstudio.service_tier", "lmstudio.reasoning", "lmstudio.provider",
        "split.random_state", "simulation.stability_runs", "profiles.assignment",
        # added in round 2, absent from the September manifests; default to the old behaviour
        "profiles.method", "profiles.membership"}


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = prefix + k
        if isinstance(v, dict) and not any(key == f or key.startswith(f + ".") for f in FREE):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _settings(dump: dict) -> dict:
    return {k: v for k, v in _flat(dump).items()
            if not any(k == f or k.startswith(f + ".") for f in FREE)}


_INDEX = pd.read_csv(REPO / "results" / "analysis" / "run_index.csv")


def canonical(ds: str, arm: str, split: str, label: str) -> dict:
    row = _INDEX[(_INDEX.dataset == DATASETS[ds]) & (_INDEX.arm == arm.upper())
                 & (_INDEX.split == split) & (_INDEX.model == label)]
    if len(row) != 1:
        raise SystemExit(f"no unique September run for {ds} {arm} {split} {label}")
    manifest = REPO / "results" / "runs" / "experiment" / row.iloc[0].run / "run_manifest.json"
    return _settings(json.loads(manifest.read_text(encoding="utf-8"))["config"])


def build(template: Path, ds: str, fam: str, *, seed: int | None = None,
          stability_runs: int | None = None, assignment: str | None = None,
          profiles: dict | None = None) -> dict:
    cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
    cfg["output_root"] = "../../../results"
    if ds == "tr":
        cfg["dataset_path"] = "../../../datasets/anket.xlsx"
        cfg.pop("paper_path", None)       # the Turkish source PDF is not on disk
    else:
        cfg["dataset_path"] = "../../../datasets/bangladesh_sns/mmc2.xlsx"
        cfg["paper_path"] = "../../../datasets/bangladesh_sns/mmc1.docx"
    family = FAMILIES[fam]
    lm = cfg["lmstudio"]
    lm["model_id"] = family["model_id"]
    for key in ("service_tier", "reasoning", "provider"):
        if key in family:
            lm[key] = family[key]
    if seed is not None:
        cfg.setdefault("split", {})["random_state"] = seed
    if stability_runs is not None:
        cfg["simulation"]["stability_runs"] = stability_runs
    if assignment is not None:
        cfg.setdefault("profiles", {})["assignment"] = assignment
    if profiles:
        cfg.setdefault("profiles", {}).update(profiles)
    return cfg


def write(name: str, cfg: dict, expected: dict) -> Path:
    path = OUT / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True), encoding="utf-8")
    got = _settings(load_config(path).model_dump(mode="json"))
    diff = {k: (expected.get(k), got.get(k)) for k in sorted(set(expected) | set(got))
            if expected.get(k) != got.get(k)}
    if diff:
        raise SystemExit(f"{name}: settings differ from the September run: {diff}")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    blocks: dict[str, list[Path]] = {}

    def add(block: str, path: Path) -> None:
        blocks.setdefault(block, []).append(path.relative_to(REPO))

    # Repeated splits, the four arms carrying the paper's contrasts.
    for fam, block in (("claude_opus46", "splits_claude"), ("gemini31pro", "splits_gemini"),
                       ("qwen38_27b", "splits_open"), ("deepseek_v41_flash", "splits_open")):
        for ds in DATASETS:
            for arm in ("dop", "bpp", "sadp", "hc"):
                template = RUNS / "c2" / f"c2_{ds}_gpt54_{arm}_s101.yaml"
                expected = canonical(ds, arm, "final_test", FAMILIES[fam]["label"])
                for seed in SEEDS:
                    cfg = build(template, ds, fam, seed=seed, stability_runs=0)
                    add(block, write(f"r2_{ds}_{fam}_{arm}_s{seed}", cfg, expected))

    # Repeated splits, the remaining four arms on GPT-5.4.
    for ds, col in (("tr", 0), ("bd", 1)):
        for arm in ("fpnp", "scp", "bbp", "tpsap"):
            template = RUNS / f"paper_final_test_openrouter_gpt54_{MAIN[arm][col]}.yaml"
            expected = canonical(ds, arm, "final_test", "GPT-5.4")
            for seed in SEEDS:
                cfg = build(template, ds, "gpt54", seed=seed, stability_runs=0)
                add("splits_gpt_rest", write(f"r2_{ds}_gpt54_{arm}_s{seed}", cfg, expected))

    # Open-weight models on the main grid, oracle and cold-start.
    for fam in OPEN:
        for ds, col in (("tr", 0), ("bd", 1)):
            for split in ("validation", "final_test"):
                for arm in ("dop", "bpp", "sadp", "hc"):
                    template = RUNS / f"paper_{split}_openrouter_gpt54_{MAIN[arm][col]}.yaml"
                    expected = canonical(ds, arm, split, "GPT-5.4")
                    cfg = build(template, ds, fam)
                    add("open_main", write(f"r2_{ds}_{fam}_{arm}_{split}", cfg, expected))
                    if arm != "dop":      # DOP receives no profile, so it has no cold-start row
                        cfg = build(template, ds, fam, assignment="coldstart")
                        add("open_coldstart",
                            write(f"r2_{ds}_{fam}_{arm}_{split}_coldstart", cfg, expected))

    # Alternative profile constructions on GPT-5.4. Everything else equals the September
    # run of the same arm and split, which is the run each of these is compared with.
    for ds, col in (("tr", 0), ("bd", 1)):
        for split in ("validation", "final_test"):
            for arm in ("bpp", "sadp"):
                template = RUNS / f"paper_{split}_openrouter_gpt54_{MAIN[arm][col]}.yaml"
                expected = canonical(ds, arm, split, "GPT-5.4")
                fixed = {**expected, "profiles.cluster_min": 3, "profiles.cluster_max": 3}
                cfg = build(template, ds, "gpt54",
                            profiles={"method": "kmeans", "cluster_min": 3, "cluster_max": 3})
                add("alt_profiles", write(f"r2_{ds}_gpt54_{arm}_{split}_kmeans", cfg, fixed))
                cfg = build(template, ds, "gpt54", assignment="coldstart",
                            profiles={"membership": "soft"})
                add("alt_profiles", write(f"r2_{ds}_gpt54_{arm}_{split}_coldstart_soft", cfg, expected))

    total = 0
    for block, paths in blocks.items():
        (STATE / f"{block}.txt").write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")
        print(f"{block:18s} {len(paths):4d} configs")
        total += len(paths)
    print(f"{'total':18s} {total:4d} configs written to {OUT.relative_to(REPO)}, every one "
          "checked against its September run")


if __name__ == "__main__":
    main()
