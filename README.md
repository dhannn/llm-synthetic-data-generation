# Synthetic training data for Filipino hate-speech detection

Can an LLM fill the gap when labelled data is scarce? This project generates synthetic Filipino/Taglish election tweets with Claude and tests whether adding them to a small real training set helps a hate-speech classifier.

**Short answer: not here.** Adding synthetic tweets didn't beat training on real tweets alone, and doubling the real data helped more than any amount of synthetic data.

## How it works

1. **Sample real data** (`src/data_ingestion.py`): a stratified sample of labelled tweets (0 = not hate speech, 1 = hate speech).
2. **Generate synthetic tweets** (`src/synthetic_data_generator.py`): real examples go into a few-shot prompt, and Claude (`claude-3-5-sonnet-20241022`) writes one new tweet per request for a target label. Requests are sent through the **Message Batches API**, so hundreds of generations run as one cheap, asynchronous job.
3. **Evaluate** (`src/model_evaluation.py`): a TF-IDF + SVM pipeline with 10-fold cross-validation. Synthetic tweets are only ever added to the **training** folds; every score is measured on **real** tweets only.

## Results (F1, 10-fold CV)

| Training set | F1 |
|---|---|
| 500 real | 81.46 |
| 1,000 real | 82.39 |
| 2,000 real | 82.36 |
| 500 real + 500 synthetic | 80.40 |
| 500 real + 1,000 synthetic | 81.22 |
| 500 real + 1,500 synthetic | 81.05 |

Reading the synthetic samples, the generations kept circling a few topics and phrasings, so they added volume without adding variety. That's the likely reason they didn't help.

## Repo layout

```
src/        data ingestion, synthetic generation (Claude Batch API), evaluation
notebooks/  01 data · 02 generation runs (test, 500, 1,000) · 03 evaluation
data/       train/test pools and generated samples (data/out/)
```

## Running it

Requires Python with `anthropic`, `pandas` and `scikit-learn`, plus an Anthropic API key in your environment. Run the notebooks in order from the repo root.
