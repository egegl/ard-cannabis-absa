"""Fixed label metrics, post grouped bootstrap, paired contrasts, prediction files."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import config as cfg

FIXED_LABELS = list(cfg.LABELS)

PREDICTION_COLUMNS = [
    "task", "condition", "target", "model_seed", "phase",
    "instance_id", "source_row_id", "post_id",
    "label", "pred",
    "prob_0", "prob_1", "prob_2",
    "logit_0", "logit_1", "logit_2",
]


def fixed_macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, labels=FIXED_LABELS, average="macro", zero_division=0))


def compute_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=FIXED_LABELS, zero_division=0
    )
    return {
        "fixed_macro_f1": fixed_macro_f1(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=FIXED_LABELS, average="weighted", zero_division=0)),
        "per_class": {
            str(l): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, l in enumerate(FIXED_LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=FIXED_LABELS).tolist(),
        "n": int(len(y_true)),
    }


def make_prediction_frame(task, condition, target, model_seed, phase, eval_df, preds, probs=None, logits=None):
    n = len(eval_df)
    frame = pd.DataFrame(
        {
            "task": task,
            "condition": condition,
            "target": target,
            "model_seed": model_seed,
            "phase": phase,
            "instance_id": eval_df["instance_id"].values,
            "source_row_id": eval_df["source_row_id"].values,
            "post_id": eval_df["post_id"].values,
            "label": eval_df["label"].astype(int).values,
            "pred": np.asarray(preds, dtype=int),
        }
    )
    for i in FIXED_LABELS:
        frame[f"prob_{i}"] = np.asarray(probs)[:, i] if probs is not None else [""] * n
        frame[f"logit_{i}"] = np.asarray(logits)[:, i] if logits is not None else [""] * n
    return frame[PREDICTION_COLUMNS]


def write_prediction_frame(frame, path):
    with cfg.open_write(path) as f:
        frame[PREDICTION_COLUMNS].to_csv(f, index=False, lineterminator="\n")


def read_prediction_csv(path):
    return pd.read_csv(path, dtype={"source_row_id": str, "post_id": str}, keep_default_na=False)


def select_target(mean_f1_by_target, tie_tolerance=cfg.TIE_TOLERANCE):
    """smallest target within tie_tolerance of the best mean"""
    best = max(mean_f1_by_target.values())
    eligible = [t for t in sorted(mean_f1_by_target) if mean_f1_by_target[t] >= best - tie_tolerance]
    chosen = eligible[0]
    return {
        "selected_target": chosen,
        "best_mean_val_f1": best,
        "selected_mean_val_f1": mean_f1_by_target[chosen],
        "tie_tolerance": tie_tolerance,
        "mean_val_f1_by_target": {str(t): mean_f1_by_target[t] for t in sorted(mean_f1_by_target)},
        "rule": "smallest target within tie_tolerance of the best mean "
                "(unweighted augmented arm, validation only)",
    }


def _post_row_indices(frame):
    idx = {}
    for i, pid in enumerate(frame["post_id"].values):
        idx.setdefault(pid, []).append(i)
    return idx


def post_grouped_bootstrap(frame, metric=fixed_macro_f1, replicates=cfg.BOOTSTRAP_REPLICATES,
                           seed=cfg.BOOTSTRAP_SEED):
    """resample posts with replacement, score the induced rows"""
    by_post = _post_row_indices(frame)
    posts = sorted(by_post)
    y_true = frame["label"].astype(int).values
    y_pred = frame["pred"].astype(int).values
    rng = np.random.default_rng(seed)
    stats = np.empty(replicates)
    for b in range(replicates):
        chosen = rng.choice(len(posts), size=len(posts), replace=True)
        rows = np.concatenate([by_post[posts[i]] for i in chosen])
        stats[b] = metric(y_true[rows], y_pred[rows])
    return {
        "point": float(metric(y_true, y_pred)),
        "mean": float(stats.mean()),
        "ci_2.5": float(np.percentile(stats, 2.5)),
        "ci_97.5": float(np.percentile(stats, 97.5)),
        "replicates": replicates,
        "seed": seed,
    }


CONTRASTS = (
    ("C1", "augmented_unweighted", "real_unweighted"),
    ("C2", "augmented_weighted", "real_weighted"),
    ("C3", "real_weighted", "real_unweighted"),
    ("C4", "augmented_weighted", "augmented_unweighted"),
)


def paired_contrast(frames_a, frames_b, metric=fixed_macro_f1, replicates=cfg.BOOTSTRAP_REPLICATES,
                    seed=cfg.BOOTSTRAP_SEED):
    """per seed delta of a minus b, one post resample per draw shared by both arms and all seeds"""
    seeds = sorted(frames_a)
    if seeds != sorted(frames_b):
        raise ValueError("both arms need the same seeds")

    ref = frames_a[seeds[0]].sort_values("instance_id").reset_index(drop=True)
    by_post = _post_row_indices(ref)
    posts = sorted(by_post)

    arrays = {}
    for s in seeds:
        fa = frames_a[s].sort_values("instance_id").reset_index(drop=True)
        fb = frames_b[s].sort_values("instance_id").reset_index(drop=True)
        if not (fa["instance_id"].equals(ref["instance_id"]) and fb["instance_id"].equals(ref["instance_id"])):
            raise ValueError("both arms need the same eval instances")
        arrays[s] = (fa["label"].astype(int).values, fa["pred"].astype(int).values, fb["pred"].astype(int).values)

    per_seed_delta = {
        str(s): float(metric(arrays[s][0], arrays[s][1]) - metric(arrays[s][0], arrays[s][2]))
        for s in seeds
    }
    deltas = np.array(list(per_seed_delta.values()))

    rng = np.random.default_rng(seed)
    boot = np.empty(replicates)
    for b in range(replicates):
        chosen = rng.choice(len(posts), size=len(posts), replace=True)
        rows = np.concatenate([by_post[posts[i]] for i in chosen])
        boot[b] = float(np.mean([
            metric(arrays[s][0][rows], arrays[s][1][rows]) - metric(arrays[s][0][rows], arrays[s][2][rows])
            for s in seeds
        ]))
    return {
        "per_seed_delta": per_seed_delta,
        "mean_delta": float(deltas.mean()),
        "sd_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "boot_mean_delta": float(boot.mean()),
        "ci_2.5": float(np.percentile(boot, 2.5)),
        "ci_97.5": float(np.percentile(boot, 97.5)),
        "replicates": replicates,
        "seed": seed,
    }
