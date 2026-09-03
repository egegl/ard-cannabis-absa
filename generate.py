"""Generate synthetic training posts with GPT-5.4 mini and build the augmented training sets.

    python generate.py                  # needs OPENAI_API_KEY, resumes where it left off
    python generate.py build-datasets
"""

import json
import math
import os
import sys

import pandas as pd

import config as cfg
import synth_checks as checks
from prepare import build_real_fold_frame, load_source


def _pool_path(paths, task, label):
    return os.path.join(paths.pools, f"{task}_class{label}_pool.csv")


def _load_pool(paths, task, label):
    path = _pool_path(paths, task, label)
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"source_row_id": str}, keep_default_na=False)
    return pd.DataFrame(columns=cfg.DATASET_COLUMNS)


def _parse(response):
    posts = json.loads(response.output_text)["posts"]
    return [(p["post_text"], p["aspect"]) for p in posts]


def generate(paths):
    from openai import OpenAI

    client = OpenAI()
    df = load_source()
    split = cfg.read_json(paths.split_manifest)
    rows = df.set_index("source_row_id")

    for task in cfg.TASKS:
        train = build_real_fold_frame(task, df, split, "train")
        reference = checks.ReferenceSet(list(train["input_text"]))
        for label in cfg.LABELS:
            class_rows = sorted(train[train["label"] == label]["source_row_id"])
            needed = max(0, max(cfg.GEN_TARGETS) - len(class_rows))
            n_calls = math.ceil(needed / cfg.POSTS_PER_CALL)
            pool = _load_pool(paths, task, label)
            for text in pool["input_text"]:
                reference.add(text)
            done = {int(i.split("::")[3]) for i in pool["instance_id"]}
            new = []
            for ci in range(n_calls):
                if ci in done:
                    continue
                demos = [
                    {"aspect": rows.loc[rid, "aspect"], "input_text": rows.loc[rid, "input_text"]}
                    for rid in cfg.select_demonstrations(task, label, ci, class_rows)
                ]
                response = client.responses.create(**cfg.request_body(cfg.build_prompt(task, label, demos)))
                for item, (post_text, aspect) in enumerate(_parse(response)):
                    reason, detail = checks.check_candidate(post_text, aspect, reference)
                    if reason:
                        print(f"{task} {label} call {ci} item {item}: rejected {reason} ({detail})")
                        continue
                    reference.add(post_text)
                    new.append({
                        "instance_id": cfg.synth_instance_id(task, label, ci, item),
                        "source_row_id": "",
                        "post_id": cfg.synth_post_id(task, label, ci, item),
                        "label": label,
                        "aspect": aspect,
                        "input_text": post_text,
                        "synthetic": 1,
                    })
                print(f"{task} {label}: call {ci + 1}/{n_calls}, {len(pool) + len(new)} accepted so far")
                pool = pd.concat([pool, pd.DataFrame(new)], ignore_index=True)[cfg.DATASET_COLUMNS]
                new = []
                with cfg.open_write(_pool_path(paths, task, label)) as f:
                    pool.to_csv(f, index=False, lineterminator="\n")
            if len(pool) < needed:
                print(f"{task} {label}: pool has {len(pool)} of {needed}, rerun to generate more")


def build_datasets(paths):
    df = load_source()
    split = cfg.read_json(paths.split_manifest)
    for task in cfg.TASKS:
        train = build_real_fold_frame(task, df, split, "train")
        counts = train["label"].value_counts()
        pools = {l: _load_pool(paths, task, l) for l in cfg.LABELS}
        for target in cfg.GEN_TARGETS:
            pieces = [train]
            for l in cfg.LABELS:
                needed = max(0, target - int(counts.get(l, 0)))
                if len(pools[l]) < needed:
                    raise RuntimeError(f"{task} class {l} pool has {len(pools[l])} posts, target {target} needs {needed}")
                pieces.append(pools[l].iloc[:needed])
            augmented = pd.concat(pieces, ignore_index=True)[cfg.DATASET_COLUMNS]
            with cfg.open_write(paths.dataset_csv(task, f"train_augmented_{target}.csv")) as f:
                augmented.to_csv(f, index=False, lineterminator="\n")
            print(f"{task} target {target}: {len(augmented)} rows, {int(augmented['synthetic'].sum())} synthetic")


if __name__ == "__main__":
    paths = cfg.Paths()
    try:
        if sys.argv[1:] == ["build-datasets"]:
            build_datasets(paths)
        elif not sys.argv[1:]:
            generate(paths)
        else:
            sys.exit(__doc__)
    except RuntimeError as exc:
        sys.exit(f"ERROR: {exc}")
