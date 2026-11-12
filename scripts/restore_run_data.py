"""Rebuild the human-data files that the public repository does not carry.

The archived runs in ``results/runs/experiment`` are distributed without any record of a
human respondent. Two kinds of file are left out and are rebuilt here from the raw survey
files, so that the offline analyses can be rerun.

``data/``       scored responses, the three split files and the two summaries. They are
                written by the benchmark's own data step from the configuration stored in
                the run manifest, so the split is the one the run used.
``profiles/``   the distributed ``*_profile_assignments.csv`` hold the respondent
                identifier and the assigned profile. The respondent's record is joined
                back, which restores the file the pipeline wrote.

The Turkish survey file is part of the repository. The Bangladesh file is third-party
supplementary material and has to be obtained from its source article first (see the
README). Runs of a dataset whose file is absent are skipped and counted.

The SHA-256 of each survey file is compared with the one recorded in the run manifest,
and a run is skipped when they differ.

Usage
-----
    PYTHONPATH=src python scripts/restore_run_data.py
    PYTHONPATH=src python scripts/restore_run_data.py --bangladesh /path/to/mmc2.xlsx
    PYTHONPATH=src python scripts/restore_run_data.py --remove      # delete the rebuilt files again
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from saykopersona.config import ResearchConfig
from saykopersona.data import build_stratified_three_way_split, load_raw_survey, score_survey_dataframe
from saykopersona.pipeline import _write_data_outputs

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs" / "experiment"
SPLITS = ("train", "validation", "final_test")


def dataset_id(config: dict) -> str:
    ds = config.get("dataset")
    return ds["dataset_id"] if isinstance(ds, dict) else "anket_tr"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_assignments(frame: pd.DataFrame, record_columns: list[str]) -> pd.DataFrame:
    """Keep the identifier and the columns the profile step added."""
    return frame[[c for c in frame.columns if c == "respondent_id" or c not in record_columns]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turkish", type=Path, default=REPO / "datasets" / "anket.xlsx")
    ap.add_argument("--bangladesh", type=Path, default=REPO / "datasets" / "bangladesh_sns" / "mmc2.xlsx")
    ap.add_argument("--remove", action="store_true", help="delete data/ and strip the assignment files again")
    args = ap.parse_args()
    source = {"anket_tr": args.turkish, "bangladesh_sns": args.bangladesh}
    digest = {ds: sha256(path) for ds, path in source.items() if path.exists()}

    frames: dict[tuple, tuple] = {}
    done = {"restored": 0, "already complete": 0, "survey file absent": 0, "survey file differs from the manifest": 0,
            "removed": 0}
    for manifest_path in sorted(RUNS.glob("*/run_manifest.json")):
        run = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = manifest["config"]
        ds = dataset_id(cfg)
        assignment_files = [run / "profiles" / f"{s}_profile_assignments.csv" for s in SPLITS]

        if args.remove:
            if (run / "data").exists():
                columns = list(pd.read_csv(run / "data" / "scored_responses.csv", nrows=0).columns)
                for f in assignment_files:
                    if f.exists():
                        strip_assignments(pd.read_csv(f), columns).to_csv(f, index=False)
                shutil.rmtree(run / "data")
                done["removed"] += 1
            continue

        if ds not in digest:
            done["survey file absent"] += 1
            continue
        if manifest.get("dataset_sha256") and manifest["dataset_sha256"] != digest[ds]:
            done["survey file differs from the manifest"] += 1
            continue
        if (run / "data" / "scored_responses.csv").exists():
            done["already complete"] += 1
            continue

        config = ResearchConfig.model_validate({**cfg, "dataset_path": str(source[ds])})
        key = (ds, config.split.validation_size, config.split.final_test_size, config.split.random_state)
        if key not in frames:
            raw = load_raw_survey(config.dataset_path, dataset=config.dataset)
            scored = score_survey_dataframe(raw, dataset=config.dataset)
            frames[key] = (raw, scored, *build_stratified_three_way_split(
                scored, validation_size=config.split.validation_size, final_test_size=config.split.final_test_size,
                random_state=config.split.random_state, dataset=config.dataset))
        raw, scored, train, validation, final_test = frames[key]
        _write_data_outputs(run, raw, scored, train, validation, final_test, config=config)

        for name, f in zip(SPLITS, assignment_files):
            if not f.exists():
                continue
            stored = pd.read_csv(f)
            if any(c in stored.columns for c in scored.columns if c != "respondent_id"):
                continue                                   # already the full file
            records = pd.read_csv(run / "data" / f"{name}_split.csv").set_index("respondent_id")
            full = records.loc[stored.respondent_id].reset_index().join(stored.drop(columns="respondent_id"))
            full.to_csv(f, index=False)
        done["restored"] += 1

    for label, n in done.items():
        if n:
            print(f"{n:5d}  runs, {label}")
    if done["survey file absent"]:
        missing = [str(p) for ds, p in source.items() if ds not in digest]
        print("not found: " + ", ".join(missing))


if __name__ == "__main__":
    main()
