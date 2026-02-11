# RACE

**RACE: Role-Aware Collaborative Embedding for Cold-Start Commuting Flow Prediction**

This repository contains the official implementation of **RACE**, a role-aware and collaborative embedding framework for urban commuting flow prediction under cold-start settings.

RACE explicitly models asymmetric regional roles (origin vs. destination) and distills collaborative signals from the OD matrix into attribute-based representations, enabling accurate flow prediction for unseen regions.

---

## Project Structure

- `main.py`: entry point for model training and evaluation.
- `framework.py`: implementation of the RACE model, including bidirectional embedding enhancement, collaborative encoding, and multi-view prediction.
- `data_load.py`: dataset loading and region-level cold-start split.
- `tools.py`: evaluation utilities, including CPC, MAE, and RMSE computation.

---

## Dataset

We use three representative metropolitan areas in the United States to evaluate RACE: **New York (NYC)**, **San Francisco (SF)**, and **Washington DC (DC)**.

### Data Sources

- **Commuting OD data**: Origin–Destination Employment Statistics (LODES) released by the U.S. Census Bureau.

- **Regional attributes**: Derived from OpenStreetMap (OSM) data.
  
For detailed dataset descriptions and citations, please refer to the paper.

### Data Format

Each dataset folder contains:

```
data/<CITY>/
├── attr.npy
├── dis.npy
└── od.npy
```

- `attr.npy`: region attribute matrix.
- `dis.npy`: pairwise spatial distance matrix between regions.
- `od.npy`: commuting origin–destination flow matrix.

---

## Configurations

For all datasets, we train RACE using the Adam optimizer with a learning rate of 1e-3 and a weight decay of 1e-5 for up to 200 epochs. Early stopping is applied with a patience of 10 epochs, and the batch size is set to 16. The embedding dimensions are set to d_r = 48 for the Attractive and Emissive embeddings, d_c = 48 for the collaborative embedding, and d = 48 for the final region representation.
## Requirements

Install dependencies via:

```shell
pip install -r requirements.txt
```

---

## Run

Example: New York City dataset  
(San Francisco and Washington DC can be run in the same way.)

```shell
python ./model/main.py --dataset NYC --embed_dim 48 --neighbor_k 30
```

