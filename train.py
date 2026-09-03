"""Fine tune the task's DeBERTa checkpoint and predict the eval fold."""

import os

import numpy as np
import pandas as pd

import config as cfg
import metrics


def encode_inputs(task, frame):
    texts = frame["input_text"].astype(str).tolist()
    if task == "absa":
        return texts, frame["aspect"].astype(str).tolist()
    return texts, None


def train_and_predict(task, train_df, eval_df, model_seed, class_weights=None, work_dir="artifacts/logs/hf"):
    """returns (metrics, preds, probs, logits) for eval_df"""
    import torch
    from torch import nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    model_name, revision = cfg.MODELS[task]
    hp = cfg.HPARAMS
    set_seed(model_seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, num_labels=len(cfg.LABELS), ignore_mismatched_sizes=True,
    )

    def tokenize(frame):
        texts, pairs = encode_inputs(task, frame)
        return tokenizer(texts, pairs, max_length=hp["max_length"], truncation="only_first", padding=False)

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, frame):
            self.enc = tokenize(frame)
            self.labels = frame["label"].astype(int).tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor([float(class_weights[str(l)]) for l in cfg.LABELS], dtype=torch.float32)

    class _WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weights = weight_tensor.to(outputs.logits.device) if weight_tensor is not None else None
            loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=hp["label_smoothing"])(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"fixed_macro_f1": metrics.fixed_macro_f1(labels, np.argmax(logits, axis=-1))}

    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=os.path.join(work_dir, f"{task}_seed{model_seed}"),
        seed=model_seed,
        data_seed=model_seed,
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp["num_epochs"],
        per_device_train_batch_size=8,
        gradient_accumulation_steps=hp["effective_batch_size"] // 8,
        per_device_eval_batch_size=32,
        warmup_ratio=hp["warmup_ratio"],
        weight_decay=hp["weight_decay"],
        max_grad_norm=hp["max_grad_norm"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="fixed_macro_f1",
        greater_is_better=True,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        logging_steps=50,
        report_to=[],
    )
    trainer = _WeightedTrainer(
        model=model,
        args=args,
        train_dataset=_Dataset(train_df),
        eval_dataset=_Dataset(eval_df),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hp["early_stopping_patience"])],
    )
    trainer.train()

    output = trainer.predict(_Dataset(eval_df))
    logits = np.asarray(output.predictions, dtype=np.float64)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = logits.argmax(axis=1)
    return metrics.compute_metrics(eval_df["label"].astype(int).values, preds), preds, probs, logits
