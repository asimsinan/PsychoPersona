"""Metadata-only profile assignment for the deployable cold-start setting.

The oracle setting assigns a held-out respondent to a training profile with
``profiles.predict_profiles``, which matches the respondent's own scale means against
the mixture centroids. Those scale means are derived from the very answers a simulation
arm is supposed to produce, so the assignment carries target information and the
resulting evaluation measures profile-known rather than deployable performance.

This module supplies the alternative. A classifier is fitted on the training split
alone, predicting the mixture profile label from the pre-response context fields that
already reach the prompt, and held-out respondents are assigned by that classifier
instead. Nothing derived from a held-out respondent's answers enters the assignment.

The predictor is fitted inside the run, from the training split the run itself produced,
rather than read from a previously archived run. Profiles are refitted whenever the
split changes, so a predictor pinned to one archived run would be trained against a
label space that no longer exists. Fitting in-run keeps the classifier and the mixture
in the same label space by construction, which also makes repeated-split designs work
without further bookkeeping.

Accuracy here bounds how well a cold-start arm can recover *profile membership*. It does
not bound respondent-level error, because a wrongly assigned profile still yields a
coherent persona card, so item-level accuracy has to be measured by running the arms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_SEED = 20260907


@dataclass(slots=True)
class ColdStartPredictor:
    """A fitted metadata-to-profile classifier and the columns it consumes."""

    model: Pipeline
    categorical: list[str]
    numeric: list[str]
    train_accuracy: float
    n_train: int
    selected: str
    cv_scores: dict[str, float]
    cv_se: dict[str, float]
    multi_select: list[str]

    @property
    def feature_columns(self) -> list[str]:
        return [*self.categorical, *self.numeric]


MULTI_SELECT_SEPARATOR = ","


def context_columns(dataset) -> tuple[list[str], list[str]]:
    """Pre-response context fields, split into categorical and numeric.

    Only fields already exposed to the prompt as respondent context are eligible. A
    field the prompt never sees would let the assignment use information the simulated
    respondent does not have, which is the same defect in a different place.
    """
    names = [
        field.name
        for field in dataset.metadata_fields
        if getattr(field, "include_in_context", False)
    ]
    numeric = [name for name in names if name.endswith("_midpoint")]
    categorical = [name for name in names if name not in numeric]
    return categorical, numeric


def detect_multi_select(frame: pd.DataFrame, categorical: list[str]) -> list[str]:
    """Categorical fields that hold a list rather than a single value.

    The Turkish platform field is a comma-separated multi-select with 91 distinct
    combinations across 537 training rows. Treated as one categorical it is close to
    useless, because ``min_frequency`` folds 66 of those combinations into a single
    "infrequent" bucket and the survivors share no structure, so "Instagram" and
    "Instagram, WhatsApp" are unrelated categories. The Bangladesh income field also
    contains commas, but as thousands separators inside four ordered bands, so the test
    requires several distinct values per field as well as a comma in most rows.
    """
    out = []
    for column in categorical:
        if column not in frame.columns:
            continue
        text = frame[column].astype(str)
        if (text.str.contains(MULTI_SELECT_SEPARATOR).mean() > 0.5
                and text.str.split(MULTI_SELECT_SEPARATOR).explode().str.strip().nunique() >= 3
                and text.nunique() > 10):
            out.append(column)
    return out


def expand_multi_select(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Replace each multi-select column with one indicator per token, plus a count."""
    if not columns:
        return frame, []
    out = frame.copy()
    added: list[str] = []
    for column in columns:
        text = out[column].astype(str)
        tokens = sorted({t.strip() for value in text for t in value.split(MULTI_SELECT_SEPARATOR)
                         if t.strip()})
        for token in tokens:
            name = f"{column}__{token}"
            out[name] = text.apply(
                lambda v, tok=token: int(tok in [t.strip() for t in v.split(MULTI_SELECT_SEPARATOR)])
            )
            added.append(name)
        count = f"{column}__count"
        out[count] = text.apply(
            lambda v: sum(1 for t in v.split(MULTI_SELECT_SEPARATOR) if t.strip())
        )
        added.append(count)
    return out, added


def _candidate(categorical: list[str], numeric: list[str], kind: str, seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
    if kind == "majority":
        clf = DummyClassifier(strategy="most_frequent")
    elif kind == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif kind == "forest_balanced":
        clf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced", random_state=seed
        )
    elif kind == "forest_plain":
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=seed)
    else:
        raise ValueError(f"unknown candidate '{kind}'")
    return Pipeline([("pre", pre), ("clf", clf)])


CANDIDATES = ("majority", "logreg", "forest_balanced", "forest_plain")


def fit_coldstart_predictor(
    train_profiles: pd.DataFrame,
    dataset,
    random_state: int = DEFAULT_SEED,
) -> ColdStartPredictor:
    """Fit the metadata-to-profile classifier on the training split.

    Four candidates are compared by five-fold cross-validated accuracy on the training
    split, and the best is refitted on the whole split. Selection deliberately uses only
    training data. Choosing by held-out accuracy would tune the cold-start arm on the
    split it is later scored against, which is a milder form of the leakage this whole
    setting exists to remove.

    The candidate set includes the majority-profile classifier, so a dataset whose
    metadata carries no profile signal is allowed to degrade to a single profile. That
    is the honest cold-start outcome rather than a defect, and forcing class-balanced
    predictions to avoid it would report a spread the metadata does not support. The
    selected candidate is recorded on the returned object and written into the run.
    """
    categorical, numeric = context_columns(dataset)
    missing = [c for c in (*categorical, *numeric) if c not in train_profiles.columns]
    if missing:
        raise ValueError(f"training split is missing context columns: {missing}")

    multi = detect_multi_select(train_profiles, categorical)
    frame, expanded = expand_multi_select(train_profiles, multi)
    categorical = [c for c in categorical if c not in multi]
    numeric = [*numeric, *expanded]
    train_profiles = frame

    target = train_profiles["profile_id"].astype(int)
    n_classes = int(target.nunique())
    folds = min(5, int(target.value_counts().min()), max(2, n_classes))
    cv_scores: dict[str, float] = {}
    cv_sd: dict[str, float] = {}
    if folds >= 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        per_fold: dict[str, list[float]] = {}
        for kind in CANDIDATES:
            scores = cross_val_score(
                _candidate(categorical, numeric, kind, random_state),
                train_profiles,
                target,
                cv=splitter,
                scoring="accuracy",
            )
            per_fold[kind] = [float(x) for x in scores]
            cv_scores[kind] = float(scores.mean())
            cv_sd[kind] = float(scores.std(ddof=1) / (len(scores) ** 0.5))
        # Plain cross-validated maximum, which is what a deployer choosing a model would
        # do. Nothing here is tuned against the held-out split.
        #
        # A margin rule was considered and rejected. On the Turkish data the forest leads
        # the majority classifier by 0.006 in cross-validation and trails it by 0.044 on
        # the held-out split, so a threshold could be set that falls back to the constant
        # predictor there. Every threshold that produced that outcome was within a
        # thousandth of one that did not, which means the rule would have been chosen for
        # its effect on this dataset rather than on principle. The scientific conclusion
        # does not depend on the choice, because both candidates sit at or below the
        # majority floor on held-out Turkish data and the finding is the absence of
        # signal either way. Selection, fold scores, standard errors and the realised
        # held-out agreement are all written into the run so the margin is visible.
        selected = max(cv_scores, key=cv_scores.__getitem__)
        per_fold_scores = per_fold
    else:
        selected = "majority"
        per_fold_scores = {}

    model = _candidate(categorical, numeric, selected, random_state)
    model.fit(train_profiles, target)
    fitted = model.predict(train_profiles)
    return ColdStartPredictor(
        model=model,
        categorical=categorical,
        numeric=numeric,
        train_accuracy=float((fitted == target.to_numpy()).mean()),
        n_train=int(len(train_profiles)),
        selected=selected,
        cv_scores={k: round(v, 4) for k, v in cv_scores.items()},
        cv_se={k: round(v, 4) for k, v in cv_sd.items()},
        multi_select=multi,
    )


def predict_coldstart_profiles(
    predictor: ColdStartPredictor,
    frame: pd.DataFrame,
) -> pd.Series:
    """Assign profiles to held-out respondents from metadata alone.

    The same multi-select expansion the predictor was fitted with is reapplied, so a
    token absent from the held-out split still yields its indicator column.
    """
    if predictor.multi_select:
        frame, _ = expand_multi_select(frame, predictor.multi_select)
        for column in predictor.numeric:
            if column not in frame.columns:
                frame[column] = 0
    predicted = predictor.model.predict(frame)
    return pd.Series(predicted, index=frame.index, name="profile_id").astype(int)


MEMBERSHIP_PREFIX = "membership_p_"


def predict_coldstart_membership(
    predictor: ColdStartPredictor,
    frame: pd.DataFrame,
    train_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Membership probabilities of held-out respondents, one column per training profile.

    The probabilities are the classifier's. When cross-validation selected the constant
    classifier, scikit-learn reports certainty for the largest profile, which is not what
    a model without predictors knows. Its probabilities are the training base rates, so
    those are returned for every respondent.
    """
    labels = sorted(int(v) for v in train_profiles["profile_id"].unique())
    if predictor.selected == "majority":
        rates = train_profiles["profile_id"].astype(int).value_counts(normalize=True)
        values = np.tile([float(rates.get(label, 0.0)) for label in labels], (len(frame), 1))
    else:
        prepared = frame
        if predictor.multi_select:
            prepared, _ = expand_multi_select(frame, predictor.multi_select)
            for column in predictor.numeric:
                if column not in prepared.columns:
                    prepared[column] = 0
        proba = predictor.model.predict_proba(prepared)
        classes = [int(c) for c in predictor.model.classes_]
        values = np.zeros((len(frame), len(labels)))
        for j, label in enumerate(classes):
            values[:, labels.index(label)] = proba[:, j]
    columns = [f"{MEMBERSHIP_PREFIX}{label}" for label in labels]
    return pd.DataFrame(values, index=frame.index, columns=columns)
