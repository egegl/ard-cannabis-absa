"""Split the corpus, write the real folds and class weights.

    python prepare.py
"""

import os
import sys

import pandas as pd

import config as cfg

REQUIRED_COLUMNS = ("post_id", "matches", "matched_sentence2", "absa_label", "trad_label")


def load_source(source_csv=cfg.SOURCE_CSV):
    df = pd.read_csv(source_csv)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns or df[col].isna().any():
            raise ValueError(f"source column {col!r} missing or has nulls")
    if len(df) != cfg.N_ROWS or df["post_id"].nunique() != cfg.N_POSTS:
        raise ValueError(f"expected {cfg.N_ROWS} rows / {cfg.N_POSTS} posts")
    return pd.DataFrame(
        {
            "source_row_id": [cfg.row_id(i) for i in range(len(df))],
            "post_id": df["post_id"].astype(str).values,
            "aspect": df["matches"].astype(str).values,
            "input_text": df["matched_sentence2"].astype(str).values,
            "absa_label": df["absa_label"].astype(int).values,
            "trad_label": df["trad_label"].astype(int).values,
        }
    )


# --- split ---

_EPS_TIE = 1e-12
_EPS_IMPROVE = 1e-9


class _SplitState:
    def __init__(self, posts):
        self.posts = posts
        self.totals = {
            t: [sum(p[t][l] for p in posts.values()) for l in cfg.LABELS] for t in cfg.TASKS
        }
        self.n_rows = sum(p["n"] for p in posts.values())
        self.n_posts = len(posts)
        self.assign = {}
        self.agg = {
            f: {"posts": 0, "rows": 0, "absa": [0, 0, 0], "trad": [0, 0, 0]}
            for f in cfg.FOLDS
        }

    def cost(self):
        # chi square deviation of post counts row counts and both label counts per fold
        c = 0.0
        for f in cfg.FOLDS:
            fr = cfg.FOLD_FRACTIONS[f]
            exp_posts = fr * self.n_posts
            c += (self.agg[f]["posts"] - exp_posts) ** 2 / exp_posts
            exp_rows = fr * self.n_rows
            c += (self.agg[f]["rows"] - exp_rows) ** 2 / exp_rows
            for t in cfg.TASKS:
                for l in cfg.LABELS:
                    e = fr * self.totals[t][l]
                    c += (self.agg[f][t][l] - e) ** 2 / e
        return c

    def _apply(self, post_id, fold, sign):
        p = self.posts[post_id]
        a = self.agg[fold]
        a["posts"] += sign
        a["rows"] += sign * p["n"]
        for t in cfg.TASKS:
            for l in cfg.LABELS:
                a[t][l] += sign * p[t][l]

    def place(self, post_id, fold):
        if post_id in self.assign:
            self._apply(post_id, self.assign[post_id], -1)
        self.assign[post_id] = fold
        self._apply(post_id, fold, +1)

    def cost_if(self, post_id, fold):
        prev = self.assign.get(post_id)
        self.place(post_id, fold)
        c = self.cost()
        if prev is None:
            self._apply(post_id, fold, -1)
            del self.assign[post_id]
        else:
            self.place(post_id, prev)
        return c


def solve_split(df):
    """greedy post level assignment then seeded local search"""
    import numpy as np

    posts = {}
    for r in df.itertuples(index=False):
        p = posts.setdefault(r.post_id, {"n": 0, "absa": [0, 0, 0], "trad": [0, 0, 0]})
        p["n"] += 1
        p["absa"][int(r.absa_label)] += 1
        p["trad"][int(r.trad_label)] += 1

    state = _SplitState(posts)

    def rarity(pid):
        p = posts[pid]
        return sum(p[t][l] / state.totals[t][l] for t in cfg.TASKS for l in cfg.LABELS)

    order = sorted(posts, key=lambda pid: (-rarity(pid), -posts[pid]["n"], pid))
    for pid in order:
        best_fold, best_cost = None, None
        for f in cfg.FOLDS:
            c = state.cost_if(pid, f)
            if best_cost is None or c < best_cost - _EPS_TIE:
                best_fold, best_cost = f, c
        state.place(pid, best_fold)
    greedy_objective = state.cost()

    rng = np.random.default_rng(cfg.SPLIT_SEED)
    pids = sorted(posts)
    passes = 0
    for _ in range(cfg.SPLIT_MAX_PASSES):
        passes += 1
        improved = False
        for i in rng.permutation(len(pids)):
            pid = pids[int(i)]
            current = state.cost()
            for f in cfg.FOLDS:
                if f == state.assign[pid]:
                    continue
                if state.cost_if(pid, f) < current - _EPS_IMPROVE:
                    state.place(pid, f)
                    improved = True
                    break
        idx = rng.permutation(len(pids))
        for k in range(0, len(idx) - 1, 2):
            a, b = pids[int(idx[k])], pids[int(idx[k + 1])]
            fa, fb = state.assign[a], state.assign[b]
            if fa == fb:
                continue
            current = state.cost()
            state.place(a, fb)
            state.place(b, fa)
            if state.cost() >= current - _EPS_IMPROVE:
                state.place(a, fa)
                state.place(b, fb)
            else:
                improved = True
        if not improved:
            break

    for t in cfg.TASKS:
        for f in cfg.FOLDS:
            for l in cfg.LABELS:
                if state.agg[f][t][l] < 1:
                    raise ValueError(f"fold {f} has no {t} label {l} rows")
    return state, greedy_objective, passes


def build_split_manifest(df):
    state, greedy_objective, passes = solve_split(df)
    post_assignments = {pid: state.assign[pid] for pid in sorted(state.assign)}
    fold_row_ids = {f: [] for f in cfg.FOLDS}
    for r in df.itertuples(index=False):
        fold_row_ids[state.assign[r.post_id]].append(r.source_row_id)
    fold_row_ids = {f: sorted(ids) for f, ids in fold_row_ids.items()}

    fold_stats = {}
    for f in cfg.FOLDS:
        fr = cfg.FOLD_FRACTIONS[f]
        fold_stats[f] = {
            "n_posts": state.agg[f]["posts"],
            "n_rows": state.agg[f]["rows"],
            "posts_expected": round(fr * state.n_posts, 4),
            "rows_expected": round(fr * state.n_rows, 4),
            "absa_counts": list(state.agg[f]["absa"]),
            "trad_counts": list(state.agg[f]["trad"]),
            "absa_expected": [round(fr * state.totals["absa"][l], 4) for l in cfg.LABELS],
            "trad_expected": [round(fr * state.totals["trad"][l], 4) for l in cfg.LABELS],
        }

    return {
        "algorithm": "greedy-joint-group-stratified+seeded-local-search",
        "split_seed": cfg.SPLIT_SEED,
        "fractions": cfg.FOLD_FRACTIONS,
        "objective": (
            "sum over folds of chi-square deviation of fold post count, row count, "
            "and per-task per-label row counts from fraction*total expectations; "
            "greedy (rarity-first, earliest-fold ties) then seeded steepest-descent "
            "local search (single-post moves + pairwise swaps, strict improvements only)"
        ),
        "objective_value": round(state.cost(), 8),
        "greedy_objective_value": round(greedy_objective, 8),
        "local_search_passes": passes,
        "n_rows": state.n_rows,
        "n_posts": state.n_posts,
        "post_assignments": post_assignments,
        "fold_row_ids": fold_row_ids,
        "fold_stats": fold_stats,
    }


# --- datasets ---

def write_dataset_csv(frame, path):
    with cfg.open_write(path) as f:
        frame[cfg.DATASET_COLUMNS].to_csv(f, index=False, lineterminator="\n")


def build_real_fold_frame(task, df, split, fold):
    ids = set(split["fold_row_ids"][fold])
    sub = df[df["source_row_id"].isin(ids)].sort_values("source_row_id")
    return pd.DataFrame(
        {
            "instance_id": [cfg.real_instance_id(r) for r in sub["source_row_id"]],
            "source_row_id": sub["source_row_id"].values,
            "post_id": sub["post_id"].values,
            "label": sub[f"{task}_label"].values,
            "aspect": sub["aspect"].values,
            "input_text": sub["input_text"].values,
            "synthetic": 0,
        }
    )


def compute_class_weights(task, df, split):
    train = build_real_fold_frame(task, df, split, "train")
    counts = train["label"].value_counts()
    n = len(train)
    return {str(l): n / (len(cfg.LABELS) * int(counts.get(l, 0))) for l in cfg.LABELS}


# --- commands ---

def cmd_split(paths, df):
    if os.path.exists(paths.split_manifest):
        print(f"split: {paths.split_manifest} exists, keeping it")
        return cfg.read_json(paths.split_manifest)
    manifest = build_split_manifest(df)
    cfg.write_text(paths.split_manifest, cfg.dump_json(manifest))
    for f in cfg.FOLDS:
        s = manifest["fold_stats"][f]
        print(f"  {f}: {s['n_posts']} posts / {s['n_rows']} rows; "
              f"absa {s['absa_counts']}; trad {s['trad_counts']}")
    return manifest


def cmd_datasets(paths, df):
    split = cfg.read_json(paths.split_manifest)
    names = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}
    for task in cfg.TASKS:
        for fold, name in names.items():
            write_dataset_csv(build_real_fold_frame(task, df, split, fold), paths.dataset_csv(task, name))
    weights = {task: compute_class_weights(task, df, split) for task in cfg.TASKS}
    cfg.write_text(
        paths.class_weights_manifest,
        cfg.dump_json({"scheme": "n / (3 * count_c) on the real training fold", "weights": weights}),
    )
    print(f"datasets: wrote real folds for {len(cfg.TASKS)} tasks and class weights")


def main():
    paths = cfg.Paths()
    df = load_source()
    cmd_split(paths, df)
    cmd_datasets(paths, df)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
