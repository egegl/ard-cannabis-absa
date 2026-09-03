"""Gemma (five sampling runs) minus the best DeBERTa condition (five seeds), same post grouped bootstrap.

    python gemma_contrast.py
"""

import argparse
import json
import os

import pandas as pd

import metrics

SEEDS = (42, 7, 123, 2024, 99)
BEST = {"absa": ("augmented_unweighted", 200), "trad": ("augmented_unweighted", 300)}


def contrast(task, gemma_dir, deberta_dir):
    cond, target = BEST[task]
    long = pd.read_csv(os.path.join(gemma_dir, "predictions_long.csv"),
                       dtype={"source_row_id": str, "post_id": str}, keep_default_na=False)
    long = long[long["task"] == task].rename(columns={"true_label": "label", "predicted_label": "pred"})
    reps = sorted(long["replicate"].unique())
    assert len(reps) == len(SEEDS), reps
    # replicate k stands in for seed k, the pairing does not change the mean or the interval
    gemma = {s: long[long["replicate"] == k] for s, k in zip(SEEDS, reps)}
    deberta = {
        s: metrics.read_prediction_csv(os.path.join(deberta_dir, f"test__{task}__{cond}__t{target}__s{s}.csv"))
        for s in SEEDS
    }
    result = metrics.paired_contrast(gemma, deberta)
    result["per_seed_delta"] = {
        f"rep{k}_vs_seed{s}": v for (s, k), v in zip(zip(SEEDS, reps), result["per_seed_delta"].values())
    }
    return {"task": task, "a": "gemma_prompted", "b": f"deberta::{cond}::t{target}", **result}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemma-dir", default="artifacts/gemma_prompted_replicates")
    ap.add_argument("--deberta-dir", default="artifacts/results/predictions")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.gemma_dir
    for task in BEST:
        res = contrast(task, args.gemma_dir, args.deberta_dir)
        with open(os.path.join(out, f"contrast_vs_deberta_{task}.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"{task}: Gemma - DeBERTa = {res['mean_delta']:.3f} (95% CI {res['ci_2.5']:.3f} to {res['ci_97.5']:.3f})")


if __name__ == "__main__":
    main()
