# RACE

**RACE: Role-Aware Collaborative Embedding for Cold-Start Commuting Flow Prediction**

This repository contains the official implementation of **RACE**, a role-aware and collaborative embedding framework for urban commuting flow prediction under cold-start settings.

RACE explicitly models asymmetric regional roles (origin vs. destination) and distills collaborative signals from the OD matrix into attribute-based representations, enabling accurate flow prediction for unseen regions.

---

## Project Structure

```
RACE/
├── data/
│   ├── NYC/
│   ├── SF/
│   └── DC/
│       ├── attr.npy
│       ├── dis.npy
│       └── od.npy
│
├── model/
│   ├── main.py
│   ├── framework.py
│   ├── data_load.py
│   └── tools.py
│
├── requirements.txt
└── README.md
```

### File Description

- **main.py**  
  Entry point for training and evaluation.

- **framework.py**  
  Implementation of the RACE model, including bidirectional embedding enhancement, collaborative encoding, and multi-view prediction.

- **data_load.py**  
  Loads dataset files and performs region-level cold-start split.

- **tools.py**  
  Implements evaluation metrics such as CPC, MAE, and RMSE.

---

## Dataset

We use three real-world commuting datasets:

- **New York (NYC)**
- **San Francisco (SF)**
- **Washington DC (DC)**

### Data Sources

- Commuting OD data: U.S. Census LODES (Origin–Destination Employment Statistics)
- Regional attributes: Derived from OpenStreetMap (POI distributions)

### Data Format

Each dataset folder contains:

```
data/<CITY>/
├── attr.npy
├── dis.npy
└── od.npy
```

- **attr.npy**  
  Region attribute matrix.

- **dis.npy**  
  Pairwise spatial distance matrix between regions.

- **od.npy**  
  Commuting origin–destination flow matrix.

---

## Configurations

For all datasets, we train RACE using:

- Optimizer: Adam  
- Learning rate: 1e-3  
- Weight decay: 1e-5  
- Maximum epochs: 200  
- Early stopping patience: 10  
- Batch size: 16  

Embedding dimensions:

- $begin:math:text$ d\_r \= 48 $end:math:text$ (Attractive / Emissive)
- $begin:math:text$ d\_c \= 48 $end:math:text$ (Collaborative)
- $begin:math:text$ d \= 48 $end:math:text$

---

## Requirements

Install dependencies via:

```shell
pip install -r requirements.txt
```

---

## Run

Example: New York City dataset  
(San Francisco and Washington DC are similar)

```shell
python ./model/main.py --dataset NYC --embed_dim 48 --neighbor_k 30
```

Main arguments:

- `--dataset` : Dataset name under `data/`
- `--embed_dim` : Embedding dimension
- `--neighbor_k` : Number of neighbors in bidirectional enhancement
