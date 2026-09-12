# Fine-Tuned and Prompted Models for Traditional and Aspect-Based Sentiment in Autoimmune Rheumatic Disease Reddit Communities

The dataset contains 479 annotated examples from seven ARD subreddits, each
labeled for traditional (whole-text) sentiment and for aspect-based sentiment
toward a cannabis term. We compare fine-tuned DeBERTa classifiers, trained in
a 2×2 design (original vs. LLM-augmented training data, unweighted vs.
class-weighted loss, five seeds per cell), against Gemma 4 31B prompted without
fine-tuning (five sampling runs), on the same 96 held-out examples.

## Data
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
Synthetic data generation (`gpt-5.4-mini-2026-03-17` through the OpenAI API,
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

Prompted Gemma 4 and the Gemma-DeBERTa comparison:

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

`microsoft/deberta-v3-base` is fine-tuned for traditional sentiment, `yangheng/deberta-v3-base-absa-v1.1` for aspect-based sentiment. Both are in `config.py` which also has the hyperparameters, seeds, and the data generation prompts. The classification prompts are in `gemma_eval.py`. All experiments were run on NVIDIA H100 GPUs in bfloat16.
