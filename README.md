Code and results for *Fine-Tuned and Prompted Models for Traditional and
Aspect-Based Sentiment in Autoimmune Rheumatic Disease (ARD) Reddit Discussions*.

The dataset contains 479 annotated examples from seven ARD subreddits, each
labeled for traditional (whole-text) sentiment and for aspect-based sentiment
toward a cannabis term. We compare fine-tuned DeBERTa classifiers, trained in
a 2×2 design (original vs. LLM-augmented training data, unweighted vs.
class-weighted loss, five seeds per cell), against Gemma 4 31B prompted without
fine-tuning (five sampling runs), on the same 96 held-out examples.

## Data

The Reddit text is not distributed with this repository, in line with the
paper's data statement. The following are therefore gitignored: the source
dataset, the per-fold dataset CSVs (`artifacts/datasets/`), and the per-row
Gemma response records (`artifacts/gemma_prompted*/records/`). The Gemma
prediction CSVs omit the `input_text` and `explanation` columns.

All other artifacts are included: the post-level split, class weights, the
generated posts, per-run metrics, per-instance predictions, summaries, and
bootstrap contrasts.

Given the dataset file (`final_sentiment_dataset.csv`) in the repository root,
the withheld files are rebuilt with:

```bash
python prepare.py
python generate.py build-datasets
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The Gemma comparison requires a newer Transformers release than the
fine-tuning code and uses a separate environment (`requirements-gemma4.txt`).

## Pipeline

Completed runs are skipped, so any command can be re-run on a checkout.

Synthetic data generation (`gpt-5.4-mini-2026-03-17` through the Responses API,
needs `OPENAI_API_KEY`; the paper's pools were produced with the same requests
sent as a batch):

```bash
python generate.py
python generate.py build-datasets
```

Fine-tuning and evaluation (`sweep` and `test` require a GPU):

```bash
python run_models.py sweep
python run_models.py baselines
python run_models.py select
python run_models.py test
python run_models.py baselines --phase test
python run_models.py collect
```

Prompted Gemma 4 and the Gemma-vs-DeBERTa contrast:

```bash
GEMMA_MODEL_PATH=/path/to/gemma-4-31B-it sbatch --export=ALL,GEMMA_REPLICATE=0 slurm/gemma4.sbatch   # 0..4
python gemma_eval.py --replicate 0 finalize
python gemma_eval.py aggregate
python gemma_contrast.py
```

Figures:

```bash
python manuscript_figures/figures.py
```

Traditional sentiment fine-tunes `microsoft/deberta-v3-base`; aspect-based
sentiment fine-tunes `yangheng/deberta-v3-base-absa-v1.1`. Both are pinned to
a commit in `config.py`, which also holds the hyperparameters, seeds, and the
two generation prompts. The classification prompts are in `gemma_eval.py`. All
runs used NVIDIA H100 GPUs in bfloat16. The Slurm templates in `slurm/` need a
partition and account for your cluster.

## Results

Macro-F1 on the 96 test examples, mean ± SD across five runs.

| Model | Aspect-based | Traditional |
|---|---:|---:|
| DeBERTa, original, unweighted | 0.427 ± 0.017 | 0.496 ± 0.005 |
| DeBERTa, original, weighted | 0.467 ± 0.040 | 0.740 ± 0.031 |
| DeBERTa, generated, unweighted | 0.541 ± 0.020 | 0.764 ± 0.008 |
| DeBERTa, generated, weighted | 0.536 ± 0.027 | 0.752 ± 0.010 |
| Gemma 4 31B, prompted | 0.752 ± 0.006 | 0.610 ± 0.014 |

Prespecified contrasts with 95% post-grouped bootstrap intervals are in
`artifacts/results/contrasts_{absa,trad}.json`; the Gemma-vs-DeBERTa interval
is in `artifacts/gemma_prompted_replicates/`.

## Repository layout

```
config.py               constants, model pins, hyperparameters, generation prompts
prepare.py              split, real folds, class weights
generate.py             synthetic posts, augmented training sets
synth_checks.py         acceptance checks for generated posts
train.py                DeBERTa fine-tuning
metrics.py              metrics, post-grouped bootstrap, paired contrasts
run_models.py           sweep, baselines, selection, test, collect
gemma_eval.py           prompted Gemma 4 classifier
gemma_contrast.py       Gemma vs. best DeBERTa condition
manuscript_figures/     figure generation
slurm/                  job templates
artifacts/              split, class weights, pools, results, Gemma runs
```
