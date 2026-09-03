"""Inputs for the manuscript figures, read from artifacts/. The asserts check them against the paper."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(HERE), "artifacts")
DEFAULT_OUTPUT = os.path.join(HERE, "output")

LABELS = ("Neutral", "Positive", "Negative")
TASKS = {"absa": "Aspect-based sentiment", "trad": "Traditional sentiment"}
FOLDS = ("train", "val", "test")
FOLD_NAMES = {"train": "Train", "val": "Validation", "test": "Test"}

CORPUS_FLOW = {
    "retrieved_posts": 75439,
    "matched_posts": 1333,
    "matched_sentences": 2071,
    "sampled": 500,
    "excluded": 21,
    "annotated_rows": 479,
    "source_posts": 374,
    "communities": 7,
}

# Rows: overall sentiment. Columns: cannabis-specific sentiment.
OVERALL_TO_CANNABIS = np.array(
    [
        [238, 65, 10],
        [4, 31, 1],
        [62, 35, 33],
    ],
    dtype=int,
)

assert int(OVERALL_TO_CANNABIS.sum()) == CORPUS_FLOW["annotated_rows"]
assert OVERALL_TO_CANNABIS.sum(axis=1).tolist() == [313, 36, 130]
assert OVERALL_TO_CANNABIS.sum(axis=0).tolist() == [304, 131, 44]
assert int(np.trace(OVERALL_TO_CANNABIS)) == 302  # agreement reported in Results


def _read_json(*parts: str) -> dict:
    path = os.path.join(ART, *parts)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_split() -> dict:
    raw = _read_json("manifests", "split.json")
    split = {
        "seed": raw["split_seed"],
        "folds": {
            fold: {
                "n_posts": raw["fold_stats"][fold]["n_posts"],
                "n_rows": raw["fold_stats"][fold]["n_rows"],
                "absa": list(raw["fold_stats"][fold]["absa_counts"]),
                "trad": list(raw["fold_stats"][fold]["trad_counts"]),
            }
            for fold in FOLDS
        },
    }
    assert (raw["n_posts"], raw["n_rows"]) == (374, 479)
    assert sum(split["folds"][f]["n_rows"] for f in FOLDS) == 479
    return split


def load_val_curves() -> pd.DataFrame:
    """Validation macro-F1 by augmentation target, unweighted augmented arm."""
    path = os.path.join(ART, "results", "val_summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    curves = pd.read_csv(path).query("condition == 'augmented_unweighted'")
    curves = curves.sort_values(["task", "target"]).reset_index(drop=True)
    assert set(curves["task"]) == {"absa", "trad"}
    assert len(curves) == 10  # five nested targets per task
    return curves


def load_selected_targets() -> dict:
    """Per-class training-size target chosen on validation, per task."""
    selected = {
        task: int(_read_json("manifests", f"selection_{task}.json")["selection"]["selected_target"])
        for task in TASKS
    }
    assert selected == {"absa": 200, "trad": 300}, selected
    return selected


# --- main text figures ---

CONDITIONS = ("real_unweighted", "real_weighted", "augmented_unweighted", "augmented_weighted")
CONDITION_NAMES = {
    "real_unweighted": "Original\nunweighted",
    "real_weighted": "Original\nweighted",
    "augmented_unweighted": "Generated\nunweighted",
    "augmented_weighted": "Generated\nweighted",
}
SEEDS = (42, 7, 123, 2024, 99)
CONTRAST_NAMES = {
    "C1": "Generated vs original,\nunweighted",
    "C2": "Generated vs original,\nweighted",
    "C3": "Weighted vs unweighted,\noriginal data",
    "C4": "Weighted vs unweighted,\ngenerated data",
}


def load_training_composition() -> dict:
    """Real and generated row counts per class in the validation-selected training sets."""
    selected = load_selected_targets()
    out = {}
    for task, target in selected.items():
        path = os.path.join(ART, "datasets", task, f"train_augmented_{target}.csv")
        frame = pd.read_csv(path, usecols=["label", "synthetic"])
        real = frame[frame["synthetic"] == 0]
        synth = frame[frame["synthetic"] == 1]
        out[task] = {
            "target": target,
            "real": [int((real["label"] == c).sum()) for c in range(3)],
            "synthetic": [int((synth["label"] == c).sum()) for c in range(3)],
        }
    split = load_split()
    for task in TASKS:
        assert out[task]["real"] == split["folds"]["train"][task]
        assert all(r + s >= min(out[task]["target"], r) for r, s in zip(out[task]["real"], out[task]["synthetic"]))
    assert out["absa"]["synthetic"] == [0, 108, 169] and out["trad"]["synthetic"] == [82, 274, 209]
    return out


def load_test_runs() -> pd.DataFrame:
    """One row per (task, condition, seed) DeBERTa test run with macro-F1 and confusion matrix."""
    folder = os.path.join(ART, "results", "test")
    rows = []
    for name in sorted(os.listdir(folder)):
        raw = _read_json("results", "test", name)
        if raw["condition"] not in CONDITIONS:
            continue
        rows.append(
            {
                "task": raw["task"],
                "condition": raw["condition"],
                "seed": int(raw["model_seed"]),
                "macro_f1": raw["metrics"]["fixed_macro_f1"],
                "confusion": np.array(raw["metrics"]["confusion_matrix"], dtype=int),
            }
        )
    runs = pd.DataFrame(rows)
    assert len(runs) == 2 * 4 * 5, len(runs)
    means = runs.groupby(["task", "condition"])["macro_f1"].mean().round(3)
    assert means["absa"]["augmented_unweighted"] == 0.541 and means["trad"]["augmented_unweighted"] == 0.764
    assert means["absa"]["real_unweighted"] == 0.427 and means["trad"]["real_unweighted"] == 0.496
    return runs


def load_test_summary() -> pd.DataFrame:
    """Frozen per-condition test summary, including the shallow baselines."""
    summary = pd.read_csv(os.path.join(ART, "results", "test_summary.csv"))
    assert set(summary["task"]) == {"absa", "trad"}
    return summary.set_index(["task", "condition"])


def load_contrasts() -> pd.DataFrame:
    """Prespecified paired contrasts C1-C4 per task: mean delta, bootstrap CI, per-seed deltas."""
    rows = []
    for task in TASKS:
        raw = _read_json("results", f"contrasts_{task}.json")
        for key, c in raw["contrasts"].items():
            rows.append(
                {
                    "task": task,
                    "contrast": key,
                    "delta": c["mean_delta"],
                    "lo": c["ci_2.5"],
                    "hi": c["ci_97.5"],
                    "per_seed": [c["per_seed_delta"][str(s)] for s in SEEDS],
                }
            )
    contrasts = pd.DataFrame(rows).set_index(["task", "contrast"])
    assert round(contrasts.loc[("absa", "C1"), "delta"], 3) == 0.114
    assert round(contrasts.loc[("trad", "C1"), "delta"], 3) == 0.268
    return contrasts


# Manuscript-stated Gemma macro-F1 means; fill from gemma_prompted_replicates/summary.csv
# once the five replicates land. None skips the agreement check for that task.
GEMMA_MANUSCRIPT_MACRO_F1 = {"absa": 0.752, "trad": 0.610}


def load_gemma() -> dict:
    """Prompted Gemma 4 over five sampling replicates: per-replicate macro-F1,
    mean and SD, and the replicate-averaged confusion matrix per task, 96 test rows."""
    folder = os.path.join(ART, "gemma_prompted_replicates")
    summary = pd.read_csv(os.path.join(folder, "summary.csv")).set_index("task")
    per_rep = pd.read_csv(os.path.join(folder, "per_replicate.csv"))
    out = {}
    for task in TASKS:
        cm = pd.read_csv(os.path.join(folder, f"confusion_matrix_mean_{task}.csv"), index_col=0)
        cm = cm.to_numpy(dtype=float)
        runs = per_rep.loc[per_rep["task"] == task, "macro_f1"].to_numpy(dtype=float)
        assert len(runs) == 5 and int(summary.loc[task, "n_replicates"]) == 5
        assert abs(cm.sum() - 96) < 1e-9 and int(summary.loc[task, "n"]) == 96
        mean = float(summary.loc[task, "macro_f1_mean"])
        assert abs(runs.mean() - mean) < 1e-9
        expected = GEMMA_MANUSCRIPT_MACRO_F1[task]
        assert expected is None or round(mean, 3) == expected, (task, mean)
        out[task] = {
            "macro_f1": mean,
            "sd": float(summary.loc[task, "macro_f1_sd"]),
            "runs": runs,
            "confusion": cm,
        }
    return out


if __name__ == "__main__":
    load_split()
    load_val_curves()
    load_selected_targets()
    load_training_composition()
    load_test_runs()
    load_test_summary()
    load_contrasts()
    load_gemma()
    print("artifacts agree with the manuscript")
