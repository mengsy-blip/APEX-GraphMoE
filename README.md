# APEX-GraphMoE

This repository contains the implementation and data organization for APEX-GraphMoE.

## Modules

- `HUBM/`: Hierarchical user behavior modeling module used to enhance conventional user and feature embeddings.
- `LLN/`: LLM-enhanced LightGCN network used to construct semantic user-user relations from user interest summaries.
- `MMOE/`: Final multi-modal Mixture-of-Experts training stage that integrates HUBM feature embeddings and LLN graph embeddings.

## Running Order

Run `HUBM` and `LLN` first. `HUBM` produces conventional feature embeddings, while `LLN` produces graph-enhanced embeddings. After these two stages are complete, run `MMOE` for final CTR prediction.

```text
HUBM feature embeddings + LLN graph embeddings -> MMOE final prediction
```

## Data Availability

The processed raw data, intermediate embeddings, and trained checkpoints exceed 100 GB in total, so they are not committed to this repository. If the complete data artifacts are needed for follow-up research or reproduction, please contact the first author directly.
