# Linker RAG: Retrieval-Augmented Generation Strategies for Conditional Transformer-Decoder-Based MOF Linker Design

Reproducibility package for the paper **"Linker RAG"** (Beyza Nur Kara).
This repository contains the full training/inference code, the leakage-safe data
split, the per-record evaluation outputs, and the summary tables reported in the
paper.

The framework couples a single shared **Conditional Transformer Decoder (CTD)**
backbone (graph encoder + property encoder + forward model + decoder) with four
retrieval strategies — **Vanilla Latent-RAG, HyDE-RAG, GraphRAG, Self-RAG** — and
compares them against a **No-RAG** generative baseline for predicting MOF linkers
from MOF properties, using the [Quantum MOF (QMOF) database](https://github.com/Andrew-S-Rosen/QMOF).

---

## Repository structure

```
CODES/
  VANILLA2.py      # No-RAG baseline + Vanilla Latent-RAG
  HYDE2.py         # No-RAG baseline + HyDE-RAG
  GRAPH2.py        # No-RAG baseline + GraphRAG
  SELF2.py         # No-RAG baseline + Self-RAG
DOCUMENTATION/
  LINKER_RAG_FINAL_SUBMISSION.xlsx   # per-record data + summary sheets
RAW_DATA/
  RAW_INFERENCE_METHOD_SEEDS.csv     # 6,000 rows = 400 test x 3 seeds x 5 methods
SUMMARY_TABLES/
  TABLE_1_GENERAL_PERFORMANCE_0.50.csv
  TABLE_2_MULTI_THRESHOLD.csv
METHODOLOGY_AND_VALIDATION.txt        # metric definitions + final results
```

## Key experimental settings (identical across all five configurations)

| Setting | Value |
|---|---|
| Training epochs | 145 (identical across all three seeds) |
| Batch size | 32 |
| Optimizer | AdamW, lr = 3e-4, CosineAnnealingLR |
| Retrieval pool | 380 |
| Final top-K | 90 |
| Random seeds | 42, 123, 999 |
| Held-out test set | 400 canonical-linker-disjoint records (fixed split seed = 42) |
| Flexible match | s_flex = max(MACCS Tanimoto, MCS coverage) |
| Thresholds | 0.50, 0.60, 0.70, 0.85, 0.90 |

Reported match rates are the **single final predicted linker per test molecule
(top-1)** — not a best-of-top-K selection or a hit@K.

## Headline results (mean over 3 seeds, tau = 0.50, n = 400)

| Method | Match count | Success (%) |
|---|---|---|
| Self-RAG | 258.0 | 64.5 |
| GraphRAG | 202.0 | 50.5 |
| HyDE-RAG | 200.0 | 50.0 |
| Vanilla-RAG | 153.3 | 38.3 |
| No-RAG | 55.0 | 13.8 |

## Requirements

- Python 3.10+
- `torch`, `torch-geometric`
- `rdkit`
- `selfies`
- `numpy`, `pandas`, `tqdm`

Install (example):

```bash
pip install torch torch-geometric rdkit selfies numpy pandas tqdm
```

## Data

The models are trained on data derived from the **QMOF database**. Place the QMOF
CSV (`qmof.csv`) in the working directory; the required columns are
`info.pld`, `info.density`, `outputs.pbe.bandgap`,
`info.symmetry.spacegroup_number`, `info.symmetry.pointgroup`, and
`info.mofid.smiles_linkers`. After RDKit validation the working set contains
15,888 records spanning 7,627 distinct canonical linkers; the leakage-safe split
yields 15,488 training and 400 held-out test records (zero linker overlap).

## How to reproduce

Each script runs one RAG variant together with the No-RAG baseline across the
three seeds and writes per-record inference CSVs and summary metrics:

```bash
python CODES/VANILLA2.py
python CODES/HYDE2.py
python CODES/GRAPH2.py
python CODES/SELF2.py
```

(Adjust the dataset path / output directory at the top of each script as needed.)

## Citation

If you use this code or data, please cite the paper (and this repository / its
archived DOI once available).

## License

Released for academic use. See `LICENSE` (add your preferred license, e.g. MIT).