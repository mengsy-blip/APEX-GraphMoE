# HUBM

HUBM is the user and feature representation enhancement module used in the APEX-GraphMoE paper. It improves multimodal CTR prediction by refining visual, textual, and user behavior features before they are integrated into the downstream multi-expert framework.

The training pipeline contains two stages:

1. Fine-tune Chinese-CLIP on item text-image pairs to obtain task-aware item representations.
2. Train a hierarchical user behavior model with three BERT encoders for bill, service, and query behavior sequences, using multimodal item embeddings and supervised CTR labels.

## Repository Contents

- `src/train_hubm.py`: cleaned English training and evaluation code.
- `configs/hubm_config.json`: default configuration template.
- `results/training_history.csv`: lightweight training-history record from the original experiment.

## Data Availability

The complete dataset used for HUBM training is larger than 100 GB, including raw interaction records, item images, and generated intermediate embeddings. It is therefore not included in this GitHub repository.

For follow-up research or reproduction that requires the full dataset, please contact the first author directly.

## Expected Data Columns

The training CSV should contain:

- `user_id`
- `item_id`
- `item_entity_names`
- `item_title`
- `bill_entity_seq`
- `service_entity_seq`
- `query_entity_seq`
- `label`

Item images are expected to be stored as `{item_id}.png` under the configured image directory.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a small sanity test:

```bash
python src/train_hubm.py --config configs/hubm_config.json --mode test
```

Run full DDP training:

```bash
torchrun --nproc_per_node=2 src/train_hubm.py --config configs/hubm_config.json --mode train
```

The output directory is excluded from Git by default because it contains large checkpoints and embedding files.
