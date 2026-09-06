# MMOE

This module contains the final multi-modal Mixture-of-Experts training stage used in APEX-GraphMoE.

## Pipeline Position

APEX-GraphMoE uses three connected stages:

1. `HUBM` is run first to produce conventional feature embeddings, including visual, textual, ID, and user behavior representations.
2. `LLN` is then run to produce graph-enhanced embeddings from LLM-generated user-interest semantics.
3. `MMOE` is run last. It integrates the HUBM feature embeddings and LLN graph embeddings through a heterogeneous multi-expert architecture with Top-K anti-polarization routing.

In short:

```text
HUBM feature embeddings + LLN graph embeddings -> MMOE final CTR prediction
```

## Files

- `src/multi_mmoe.py`: heterogeneous multi-modal MMoE model with Top-K routing.
- `src/multi_model_train.py`: streaming training and evaluation utilities.
- `src/ulti_main.py`: main training entry point.
- `src/utils.py`: data loading and helper utilities.

## Expected Inputs

The MMOE stage expects precomputed H5 embedding files from HUBM and graph embedding outputs from LLN/LightGCN. These artifacts are large and are not included in this repository.

The original experiment uses more than 100 GB of data and generated embeddings. For follow-up research or reproduction that requires the full data, please contact the first author directly.

## Usage

Install dependencies:

```bash
pip install -r MMOE/requirements.txt
```

Run the final MMOE stage:

```bash
cd MMOE/src
python ulti_main.py
```

Before running, update the data paths in `src/ulti_main.py` so that they point to the local HUBM H5 embeddings and LLN graph embedding directory.
