"""Validation sweep, baselines, target selection, test runs, summaries and contrasts.

    python run_models.py sweep [--task absa|trad|all] [--condition ...]
    python run_models.py baselines [--phase val|test]
    python run_models.py select
    python run_models.py test
    python run_models.py collect
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import config as cfg
import metrics

REAL_CONDITIONS = ("real_unweighted", "real_weighted")
AUGMENTED_CONDITIONS = ("augmented_unweighted", "augmented_weighted")


def _read_dataset(paths, task, name):
    return pd.read_csv(paths.dataset_csv(task, name), dtype={"source_row_id": str, "post_id": str},
                       keep_default_na=False)


def _load_train(paths, task, condition, target):
    if condition in REAL_CONDITIONS:
        return _read_dataset(paths, task, "train.csv")
    return _read_dataset(paths, task, f"train_augmented_{target}.csv")


def _class_weights(paths, task, condition):
    if not condition.endswith("_weighted"):
        return None
    return cfg.read_json(paths.class_weights_manifest)["weights"][task]


def _selected_target(paths, task):
    path = paths.selection_manifest(task)
    if not os.path.exists(path):
        return None
    return cfg.read_json(path)["selection"]["selected_target"]


def _safe(condition):
    return condition.replace("::", "-")


def _result_path(paths, phase, task, condition, target, seed):
    root = paths.results_val if phase == "val" else paths.results_test
    return os.path.join(root, f"{task}__{_safe(condition)}__t{target}__s{seed}.json")


def _prediction_path(paths, phase, task, condition, target, seed):
    return os.path.join(paths.predictions, f"{phase}__{task}__{_safe(condition)}__t{target}__s{seed}.csv")


def _record_run(paths, phase, task, condition, target, seed, eval_frame, result_metrics, preds, probs, logits,
                n_train, n_synthetic):
    pred_path = _prediction_path(paths, phase, task, condition, target, seed)
    metrics.write_prediction_frame(
        metrics.make_prediction_frame(task, condition, target, seed, phase, eval_frame, preds, probs, logits),
        pred_path,
    )
    result = {
        "phase": phase,
        "task": task,
        "condition": condition,
        "target": target,
        "model_seed": seed,
        "metrics": result_metrics,
        "n_train": n_train,
        "n_synthetic": n_synthetic,
        "prediction_file": os.path.relpath(pred_path, paths.base),
    }
    cfg.write_text(_result_path(paths, phase, task, condition, target, seed), cfg.dump_json(result))


def _read_results(paths, phase):
    root = paths.results_val if phase == "val" else paths.results_test
    return [cfg.read_json(p) for p in sorted(glob.glob(os.path.join(root, "*.json")))]


def _default_train_fn():
    import train
    return train.train_and_predict


def _run_one(paths, task, condition, target, seed, phase, eval_frame, train_fn):
    if os.path.exists(_result_path(paths, phase, task, condition, target, seed)):
        print(f"{phase}[{task} {condition} t{target} s{seed}]: done, skipping")
        return
    train_df = _load_train(paths, task, condition, target)
    weights = _class_weights(paths, task, condition)
    result_metrics, preds, probs, logits = train_fn(task, train_df, eval_frame, seed, class_weights=weights)
    _record_run(paths, phase, task, condition, target, seed, eval_frame, result_metrics, preds, probs, logits,
                n_train=len(train_df), n_synthetic=int((train_df["synthetic"] == 1).sum()))
    print(f"{phase}[{task} {condition} t{target} s{seed}]: macro_f1 {result_metrics['fixed_macro_f1']:.4f}")


def _val_grid(paths, task):
    grid = [(c, 0) for c in REAL_CONDITIONS]
    grid += [("augmented_unweighted", t) for t in cfg.GEN_TARGETS]
    selected = _selected_target(paths, task)
    if selected is not None:
        grid.append(("augmented_weighted", selected))  # weighted arm uses the selected target
    return grid


def cmd_sweep(paths, task_filter="all", condition_filter="all", train_fn=None):
    train_fn = train_fn or _default_train_fn()
    for task in cfg.TASKS if task_filter == "all" else (task_filter,):
        val_frame = _read_dataset(paths, task, "validation.csv")
        for condition, target in _val_grid(paths, task):
            if condition_filter not in ("all", condition):
                continue
            for seed in cfg.MODEL_SEEDS:
                _run_one(paths, task, condition, target, seed, "val", val_frame, train_fn)


# --- baselines ---

def _tfidf_features(task, frame):
    if task == "absa":
        return (frame["input_text"].astype(str) + " " + frame["aspect"].astype(str)).tolist()
    return frame["input_text"].astype(str).tolist()


def _baseline_runs(paths, task, eval_frame):
    train_real = _load_train(paths, task, "real_unweighted", 0)
    props = train_real["label"].value_counts(normalize=True).reindex(cfg.LABELS, fill_value=0.0).values
    n = len(eval_frame)

    preds = np.zeros(n, dtype=int)
    probs = np.zeros((n, len(cfg.LABELS)))
    probs[:, 0] = 1.0
    yield "baseline::always_neutral", 0, 0, preds, probs, None, len(train_real), 0

    for seed in cfg.MODEL_SEEDS:
        rng = np.random.default_rng(seed)
        preds = rng.choice(cfg.LABELS, size=n, p=props)
        yield "baseline::stratified_random", 0, seed, preds, np.tile(props, (n, 1)), None, len(train_real), 0

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    conditions = [(c, 0) for c in REAL_CONDITIONS]
    selected = _selected_target(paths, task)
    if selected is not None:
        conditions += [(c, selected) for c in AUGMENTED_CONDITIONS]
    for condition, target in conditions:
        train_df = _load_train(paths, task, condition, target)
        weights = _class_weights(paths, task, condition)
        class_weight = {int(l): w for l, w in weights.items()} if weights else None
        for seed in cfg.MODEL_SEEDS:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            x_train = vec.fit_transform(_tfidf_features(task, train_df))
            x_eval = vec.transform(_tfidf_features(task, eval_frame))
            clf = LogisticRegression(max_iter=2000, random_state=seed, class_weight=class_weight)
            clf.fit(x_train, train_df["label"].astype(int).values)
            probs_full = np.zeros((n, len(cfg.LABELS)))
            probs_part = clf.predict_proba(x_eval)
            for i, c in enumerate(clf.classes_):
                probs_full[:, int(c)] = probs_part[:, i]
            yield (f"baseline::tfidf_logreg::{condition}", target, seed, probs_full.argmax(axis=1),
                   probs_full, None, len(train_df), int((train_df["synthetic"] == 1).sum()))


def cmd_baselines(paths, phase="val", task_filter="all"):
    for task in cfg.TASKS if task_filter == "all" else (task_filter,):
        eval_frame = _read_dataset(paths, task, "test.csv" if phase == "test" else "validation.csv")
        for condition, target, seed, preds, probs, logits, n_train, n_synth in _baseline_runs(paths, task, eval_frame):
            if os.path.exists(_result_path(paths, phase, task, condition, target, seed)):
                continue
            result_metrics = metrics.compute_metrics(eval_frame["label"].astype(int).values, preds)
            _record_run(paths, phase, task, condition, target, seed, eval_frame, result_metrics, preds, probs,
                        logits, n_train, n_synth)
            print(f"{phase}[{task} {condition} t{target} s{seed}]: macro_f1 {result_metrics['fixed_macro_f1']:.4f}")


# --- select and test ---

def cmd_select(paths):
    results = [r for r in _read_results(paths, "val") if r["condition"] == "augmented_unweighted"]
    for task in cfg.TASKS:
        by_target = {}
        for t in cfg.GEN_TARGETS:
            scores = [r["metrics"]["fixed_macro_f1"] for r in results if r["task"] == task and r["target"] == t]
            if len(scores) != len(cfg.MODEL_SEEDS):
                raise RuntimeError(f"select[{task}]: target {t} has {len(scores)} of {len(cfg.MODEL_SEEDS)} seeds")
            by_target[t] = float(np.mean(scores))
        selection = metrics.select_target(by_target)
        path = paths.selection_manifest(task)
        if os.path.exists(path):
            print(f"select[{task}]: {path} exists, keeping target {cfg.read_json(path)['selection']['selected_target']}")
            continue
        cfg.write_text(path, cfg.dump_json({"task": task, "selection": selection}))
        print(f"select[{task}]: target {selection['selected_target']} (means {selection['mean_val_f1_by_target']})")


def cmd_test(paths, task_filter="all", train_fn=None):
    train_fn = train_fn or _default_train_fn()
    for task in cfg.TASKS if task_filter == "all" else (task_filter,):
        selected = _selected_target(paths, task)
        if selected is None:
            raise RuntimeError(f"test[{task}]: run select first")
        test_frame = _read_dataset(paths, task, "test.csv")
        conditions = [(c, 0) for c in REAL_CONDITIONS] + [(c, selected) for c in AUGMENTED_CONDITIONS]
        for condition, target in conditions:
            for seed in cfg.MODEL_SEEDS:
                _run_one(paths, task, condition, target, seed, "test", test_frame, train_fn)


# --- reporting ---

def _summary_frame(results):
    frame = pd.DataFrame([
        {
            "task": r["task"], "condition": r["condition"], "target": r["target"],
            "model_seed": r["model_seed"],
            "fixed_macro_f1": r["metrics"]["fixed_macro_f1"],
            "accuracy": r["metrics"]["accuracy"],
            "balanced_accuracy": r["metrics"]["balanced_accuracy"],
            "weighted_f1": r["metrics"]["weighted_f1"],
        }
        for r in results
    ])
    return (
        frame.groupby(["task", "condition", "target"])
        .agg(
            n_seeds=("model_seed", "count"),
            f1_mean=("fixed_macro_f1", "mean"),
            f1_sd=("fixed_macro_f1", "std"),
            acc_mean=("accuracy", "mean"),
            acc_sd=("accuracy", "std"),
            bal_acc_mean=("balanced_accuracy", "mean"),
            weighted_f1_mean=("weighted_f1", "mean"),
        )
        .reset_index()
        .sort_values(["task", "condition", "target"])
    )


def cmd_collect(paths):
    for phase in ("val", "test"):
        results = _read_results(paths, phase)
        if not results:
            continue
        path = os.path.join(paths.results, f"{phase}_summary.csv")
        with cfg.open_write(path) as f:
            _summary_frame(results).to_csv(f, index=False, lineterminator="\n")
        print(f"collect: wrote {path}")

    if not _read_results(paths, "test"):
        return
    for task in cfg.TASKS:
        selected = _selected_target(paths, task)
        frames = {}
        for condition in cfg.CONDITIONS:
            target = 0 if condition in REAL_CONDITIONS else selected
            per_seed = {}
            for seed in cfg.MODEL_SEEDS:
                path = _prediction_path(paths, "test", task, condition, target, seed)
                if os.path.exists(path):
                    per_seed[seed] = metrics.read_prediction_csv(path)
            if len(per_seed) == len(cfg.MODEL_SEEDS):
                frames[condition] = per_seed
        report = {"task": task, "selected_target": selected, "bootstrap": {}, "contrasts": {}}
        for condition, per_seed in frames.items():
            report["bootstrap"][condition] = metrics.post_grouped_bootstrap(pd.concat(per_seed.values(), ignore_index=True))
        for name, cond_a, cond_b in metrics.CONTRASTS:
            if cond_a in frames and cond_b in frames:
                report["contrasts"][name] = {"a": cond_a, "b": cond_b,
                                             **metrics.paired_contrast(frames[cond_a], frames[cond_b])}
        path = os.path.join(paths.results, f"contrasts_{task}.json")
        cfg.write_text(path, cfg.dump_json(report))
        print(f"collect: wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sweep", "baselines", "select", "test", "collect"])
    parser.add_argument("--task", choices=["absa", "trad", "all"], default="all")
    parser.add_argument("--condition", default="all")
    parser.add_argument("--phase", choices=["val", "test"], default="val")
    parser.add_argument("--base-dir", default="artifacts", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    paths = cfg.Paths(args.base_dir)
    if args.command == "sweep":
        cmd_sweep(paths, args.task, args.condition)
    elif args.command == "baselines":
        cmd_baselines(paths, phase=args.phase, task_filter=args.task)
    elif args.command == "select":
        cmd_select(paths)
    elif args.command == "test":
        cmd_test(paths, args.task)
    elif args.command == "collect":
        cmd_collect(paths)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
