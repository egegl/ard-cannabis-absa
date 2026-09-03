"""Prompted Gemma 4 31B on the 96 test rows, both tasks.

    python gemma_eval.py run --model-path /path/to/gemma-4-31B-it [--replicate K] [--shard-count 8 --shard-index I]
    python gemma_eval.py finalize [--replicate K]
    python gemma_eval.py aggregate

Every response is saved to its own record as it arrives, so a run can be resumed.
Malformed responses are kept and retried with the next seed. Replicates rerun the
same protocol with salted seeds. GEMMA_DECODING=greedy uses greedy decoding.
"""

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

import config as cfg

DECODING = os.environ.get("GEMMA_DECODING", "sampled")
OUTPUT_DIR = "artifacts/gemma_prompted" + ("_greedy" if DECODING == "greedy" else "")
AGGREGATE_DIR = "artifacts/gemma_prompted_replicates"
LABELS = [0, 1, 2]
LABEL_NAMES = {0: "neutral", 1: "positive", 2: "negative"}
TASKS = ("absa", "trad")

# model card sampling settings
GENERATION = {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new_tokens": 1024}
if DECODING == "greedy":
    GENERATION.update({"do_sample": False})
SEED_SALT = "e3ed408fc22050653bb57e0a0d3ae6e1752c3451b0116d5ebf97cce62b843907"

# demonstrations pasted from the original notebooks, given as the training row
# they came from plus how the paste differs (a slice, or one extra space)
NOTEBOOK_EXCERPTS = {
    "trad": {0: {"source_row_id": "row_0074", "slice": (1114, 1353)}},
    "absa": {
        0: {"source_row_id": "row_0032", "aspect": "cbd", "extra_space_at": 208},
        2: {"source_row_id": "row_0130", "aspect": "cbd", "extra_space_at": 124},
    },
}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def output_dir(replicate):
    return OUTPUT_DIR if replicate == 0 else f"{OUTPUT_DIR}_rep{replicate}"


def load_fold(task, fold):
    return pd.read_csv(f"artifacts/datasets/{task}/{fold}.csv", dtype={"source_row_id": str, "post_id": str},
                       keep_default_na=False)


def demonstrations(task, train):
    demos = []
    for label in LABELS:
        rows = train[train["label"] == label]
        spec = NOTEBOOK_EXCERPTS[task].get(label)
        if spec:
            row = rows[rows["source_row_id"] == spec["source_row_id"]].iloc[0]
            text = row["input_text"]
            if "slice" in spec:
                text = text[spec["slice"][0]:spec["slice"][1]]
            if "extra_space_at" in spec:
                p = spec["extra_space_at"]
                text = text[:p] + " " + text[p:]
        else:
            row = min(rows.itertuples(index=False), key=lambda r: sha(f"42|gemma-prompted-v1|{task}|{label}|{r.source_row_id}"))
            text = row.input_text
        demos.append({"label": label, "aspect": row.aspect if hasattr(row, "aspect") else row["aspect"], "text": text})
    return demos


def build_prompt(task, row, demos):
    tail = (
        'Return the result in valid JSON with exactly these two keys: "label" and "explanation". '
        '"label" should either be 0, 1, or 2, and "explanation" should be one sentence. Do not '
        "include anything before or after the JSON object. Return only valid JSON with no comments "
        "and no markdown."
    )
    if task == "trad":
        examples = "\n".join(f"Example for label {d['label']}: {d['text']}" for d in demos)
        return (
            "You are an expert in sociolinguistics of medical and social media language, focused on "
            "determining the sentiment of Reddit posts.\n\n"
            'The label is "0" if the overall sentiment of the text is neutral.\n'
            '"1" if the overall sentiment is positive.\n'
            '"2" if the overall sentiment is negative.\n\n'
            "To assist you with classification, below are three examples:\n"
            f"{examples}\n\n"
            "Please read the following text, and think step by step about the overall sentiment "
            "expressed before deciding on the label:\n"
            f"{row.input_text}\n\n" + tail
        )
    examples = "\n".join(f'Example for label {d["label"]} (aspect: {d["aspect"]}): "{d["text"]}"' for d in demos)
    return (
        "You are an expert in sociolinguistics of medical and social media language, focused on "
        "determining the sentiment of cannabis-related aspects in Reddit posts.\n\n"
        'The label is "0" if the sentiment of the aspect is neutral.\n'
        '"1" if the sentiment of the aspect is positive.\n'
        '"2" if the sentiment of the aspect is negative.\n\n'
        "To assist you with classification, below are three examples:\n"
        f"{examples}\n\n"
        "Please read the following aspect and text, and think step by step about the sentiment "
        "expressed toward the aspect before deciding on the label:\n"
        f'Aspect: "{row.aspect}"\n'
        f'Text: "{row.input_text}"\n\n' + tail
    )


def parse_prediction(final_content, thinking):
    """label and explanation, raises ValueError on anything malformed"""
    if not thinking.strip():
        raise ValueError("no thought channel")
    obj, end = json.JSONDecoder().raw_decode(final_content.lstrip())
    if final_content.lstrip()[end:].strip():
        raise ValueError("text outside the JSON object")
    if not isinstance(obj, dict) or set(obj) != {"label", "explanation"}:
        raise ValueError("JSON must have exactly label and explanation")
    label, explanation = obj["label"], obj["explanation"]
    if isinstance(label, bool) or label not in LABELS:
        raise ValueError("label must be 0, 1 or 2")
    if not isinstance(explanation, str) or not explanation.strip() or "\n" in explanation \
            or re.search(r"[.!?][\"')\]]*\s+\S", explanation.strip()):
        raise ValueError("explanation must be one sentence")
    return label, explanation.strip()


def seed_for(task, instance_id, attempt, replicate):
    material = f"{SEED_SALT}|{task}|{instance_id}|{attempt}" + (f"|rep{replicate}" if replicate else "")
    return int(sha(material)[:8], 16)


class Gemma4:
    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, local_files_only=True,
        ).eval()

    def generate(self, prompt, seed):
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=True,
        ).to(self.model.device)
        n = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            out = self.model.generate(**inputs, **GENERATION)
        raw = self.processor.decode(out[0][n:], skip_special_tokens=False)
        parsed = self.processor.parse_response(raw)
        return raw, str(parsed.get("thinking", "")), str(parsed.get("content", parsed.get("answer", "")))


def run(args):
    out = output_dir(args.replicate)
    model = Gemma4(args.model_path)
    unresolved = []
    for task in TASKS:
        train, test = load_fold(task, "train"), load_fold(task, "test")
        demos = demonstrations(task, train)
        for index, row in enumerate(test.itertuples(index=False)):
            if index % args.shard_count != args.shard_index:
                continue
            path = os.path.join(out, "records", task, f"{row.source_row_id}.json")
            record = cfg.read_json(path) if os.path.exists(path) else {
                "task": task, "instance_id": row.instance_id, "source_row_id": row.source_row_id,
                "post_id": row.post_id, "true_label": int(row.label), "aspect": row.aspect,
                "status": "incomplete", "attempts": [],
            }
            if record["status"] == "complete":
                continue
            prompt = build_prompt(task, row, demos)
            first = len(record["attempts"]) + 1
            for attempt in range(first, first + args.max_attempts):
                seed = seed_for(task, row.instance_id, attempt, args.replicate)
                print(f"{task} {index + 1:03d}/{len(test)} {row.source_row_id}: attempt {attempt}", flush=True)
                raw, thinking, final = model.generate(prompt, seed)
                try:
                    label, explanation = parse_prediction(final, thinking)
                except ValueError as exc:
                    record["attempts"].append({"attempt": attempt, "seed": seed, "valid": False, "error": str(exc),
                                               "raw_response": raw, "thinking": thinking, "final_content": final})
                    cfg.write_text(path, cfg.dump_json(record))
                    continue
                record["attempts"].append({"attempt": attempt, "seed": seed, "valid": True})
                record.update({
                    "status": "complete", "predicted_label": label, "explanation": explanation, "seed": seed,
                    "successful_attempt": attempt, "raw_response": raw, "final_content": final,
                    "thinking": thinking, "thinking_chars": len(thinking),
                })
                cfg.write_text(path, cfg.dump_json(record))
                break
            else:
                unresolved.append(f"{task}/{row.source_row_id}")
    if unresolved:
        sys.exit("still incomplete after retries, rerun: " + ", ".join(unresolved))


def metrics(true, predicted):
    precision, recall, f1, support = precision_recall_fscore_support(true, predicted, labels=LABELS, zero_division=0)
    return {
        "n": len(true),
        "accuracy": float(accuracy_score(true, predicted)),
        "micro_f1": float(f1_score(true, predicted, labels=LABELS, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(true, predicted, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {
            str(l): {"name": LABEL_NAMES[l], "precision": float(precision[i]), "recall": float(recall[i]),
                     "f1": float(f1[i]), "support": int(support[i]),
                     "predicted_count": int(sum(v == l for v in predicted))}
            for i, l in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(true, predicted, labels=LABELS).tolist(),
        "confusion_matrix_orientation": "rows=true label; columns=predicted label; order=0,1,2",
    }


def confusion_frame(matrix):
    cm = pd.DataFrame(matrix, columns=[f"pred_{l}_{LABEL_NAMES[l]}" for l in LABELS])
    cm.insert(0, "true_label", [f"{l}_{LABEL_NAMES[l]}" for l in LABELS])
    return cm


def write_csv(path, frame):
    with cfg.open_write(path) as f:
        frame.to_csv(f, index=False, lineterminator="\n")


def finalize(args):
    out = output_dir(args.replicate)
    summary = []
    for task in TASKS:
        test = load_fold(task, "test")
        rows = []
        for row in test.itertuples(index=False):
            r = cfg.read_json(os.path.join(out, "records", task, f"{row.source_row_id}.json"))
            if r["status"] != "complete":
                sys.exit(f"{task}/{row.source_row_id} is incomplete")
            rows.append({
                "task": task, "instance_id": r["instance_id"], "source_row_id": r["source_row_id"],
                "post_id": r["post_id"], "true_label": r["true_label"], "predicted_label": r["predicted_label"],
                "aspect": row.aspect, "input_text": row.input_text, "explanation": r["explanation"],
                "seed": r["seed"], "successful_attempt": r["successful_attempt"],
                "thinking_present": bool(r["thinking"].strip()), "thinking_chars": r["thinking_chars"],
            })
        preds = pd.DataFrame(rows)
        m = metrics(preds["true_label"].tolist(), preds["predicted_label"].tolist())
        m["task"] = task
        write_csv(os.path.join(out, f"predictions_{task}.csv"), preds)
        cfg.write_text(os.path.join(out, f"metrics_{task}.json"), cfg.dump_json(m))
        write_csv(os.path.join(out, f"per_class_{task}.csv"),
                  pd.DataFrame([{"label": int(l), **v} for l, v in m["per_class"].items()]))
        write_csv(os.path.join(out, f"confusion_matrix_{task}.csv"), confusion_frame(m["confusion_matrix"]))
        summary.append({"task": task, "n": m["n"], "accuracy": m["accuracy"], "micro_f1": m["micro_f1"],
                        "macro_f1": m["macro_f1"]})
        print(f"{task}: accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    write_csv(os.path.join(out, "summary.csv"), pd.DataFrame(summary))


def aggregate(args):
    replicates = [int(v) for v in args.replicates.split(",")]
    long_rows, per_rep = [], []
    for k in replicates:
        for task in TASKS:
            preds = pd.read_csv(os.path.join(output_dir(k), f"predictions_{task}.csv"),
                                dtype={"source_row_id": str, "post_id": str}, keep_default_na=False)
            m = metrics(preds["true_label"].tolist(), preds["predicted_label"].tolist())
            per_rep.append({
                "replicate": k, "task": task, "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                **{f"f1_{LABEL_NAMES[l]}": m["per_class"][str(l)]["f1"] for l in LABELS},
                **{f"precision_{LABEL_NAMES[l]}": m["per_class"][str(l)]["precision"] for l in LABELS},
                **{f"recall_{LABEL_NAMES[l]}": m["per_class"][str(l)]["recall"] for l in LABELS},
                "first_attempt_valid": int((preds["successful_attempt"] == 1).sum()),
            })
            for r in preds.itertuples(index=False):
                long_rows.append({"replicate": k, "task": task, "instance_id": r.instance_id,
                                  "source_row_id": r.source_row_id, "post_id": r.post_id,
                                  "true_label": r.true_label, "predicted_label": r.predicted_label,
                                  "seed": r.seed, "successful_attempt": r.successful_attempt})
    per_rep = pd.DataFrame(per_rep).sort_values(["task", "replicate"]).reset_index(drop=True)
    long = pd.DataFrame(long_rows)
    cols = [c for c in per_rep.columns if c not in ("replicate", "task", "n")]
    summary = []
    for task in TASKS:
        block = per_rep[per_rep["task"] == task]
        row = {"task": task, "n": 96, "n_replicates": len(block)}
        for c in cols:
            row[f"{c}_mean"] = float(block[c].mean())
            row[f"{c}_sd"] = float(block[c].std(ddof=1))
        summary.append(row)
        mats = [confusion_matrix(g["true_label"], g["predicted_label"], labels=LABELS)
                for _, g in long[long["task"] == task].groupby("replicate")]
        write_csv(os.path.join(AGGREGATE_DIR, f"confusion_matrix_mean_{task}.csv"),
                  confusion_frame(np.mean(mats, axis=0)))
    write_csv(os.path.join(AGGREGATE_DIR, "predictions_long.csv"), long)
    write_csv(os.path.join(AGGREGATE_DIR, "per_replicate.csv"), per_rep)
    write_csv(os.path.join(AGGREGATE_DIR, "summary.csv"), pd.DataFrame(summary))
    for row in summary:
        print(f"{row['task']}: macro_f1 {row['macro_f1_mean']:.4f} +/- {row['macro_f1_sd']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replicate", type=int, default=0)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-path", default=os.environ.get("GEMMA_MODEL_PATH", "gemma-4-31B-it"))
    r.add_argument("--max-attempts", type=int, default=3)
    r.add_argument("--shard-count", type=int, default=1)
    r.add_argument("--shard-index", type=int, default=0)
    sub.add_parser("finalize")
    a = sub.add_parser("aggregate")
    a.add_argument("--replicates", default="0,1,2,3,4")
    args = p.parse_args()
    {"run": run, "finalize": finalize, "aggregate": aggregate}[args.command](args)
