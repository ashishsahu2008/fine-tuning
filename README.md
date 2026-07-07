# Text-to-SQL with QLoRA Fine-Tuning

Fine-tuning **Qwen2.5-3B-Instruct** to translate natural-language questions into SQL
queries, using LoRA/QLoRA on a single free GPU.

**[🤗 Model](https://huggingface.co/ashishsahu2008/qwen2.5-3b-text2sql)** ·
**[🚀 Live demo](https://huggingface.co/spaces/ashishsahu2008/text2sql-demo)**

## Results

Evaluated on a held-out test split (n=200) the model never saw during training.
Both models scored with identical string normalization (lowercase, strip `;`,
canonicalize quotes) so the comparison is apples-to-apples.

| Model                              | Exact-match |
|------------------------------------|:-----------:|
| Qwen2.5-3B-Instruct (base)         |    41.5%    |
| **+ LoRA fine-tuning (this repo)** |  **72.5%**  |

Only **0.96%** of the model's parameters (~30M of 3.1B) were trained.

## Method

- **Base model:** `unsloth/Qwen2.5-3B-Instruct`, loaded in 4-bit (QLoRA).
- **Dataset:** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) —
  each example pairs a `CREATE TABLE` schema and an English question with the target SQL.
  Used a 3,000-row subset (2,700 train / 300 test).
- **Adapters:** LoRA (rank 16, alpha 16) on all attention + MLP projections.
- **Training:** 2 epochs, lr 2e-4, effective batch size 8, ~25 min on a Colab T4.
- **Tooling:** [Unsloth](https://github.com/unslothai/unsloth) + TRL `SFTTrainer`.

## Why fine-tuning helped

The base model often knew the right query but wrapped it in prose
(`SQL QUERY: SELECT ...`) or markdown fences, which fails the "output only SQL"
requirement. Fine-tuning taught it to emit clean, directly-executable SQL — a mix of
better instruction-following and some genuine correctness gains.

## Run it yourself

Training (Colab, free T4):
```bash
pip install unsloth
# then run notebook/train.ipynb top to bottom
```

Local inference:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("your-username/qwen2.5-3b-text2sql")
model = AutoModelForCausalLM.from_pretrained("your-username/qwen2.5-3b-text2sql")
# see app.py for the full prompt format
```

## Limitations & next steps

- **Metric:** exact-match (after normalization) undercounts semantically-correct
  queries written differently. A stronger next step is AST-based comparison with
  `sqlglot`, or true **execution accuracy** on a benchmark with populated databases
  (Spider / BIRD) — `sql-create-context` ships schemas only, no data rows.
- **Scope:** single-table `CREATE TABLE` schemas; complex multi-join databases are
  out of distribution for this training subset.

## Repo structure

```
notebook/train.ipynb   # end-to-end training + evaluation
app.py                 # Gradio demo (deployed to the Space)
requirements.txt       # dependencies for the Space
```
