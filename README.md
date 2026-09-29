# Galactic Binary Source Separation

This repository is the third component of a four-part project completed during an internship at [L2IT](https://www.l2it.in2p3.fr/):

1. [Dataset generation](https://github.com/TristanDonze/gb-dataset-gen)
2. [Source counting](https://github.com/TristanDonze/gb-sources-counting)
3. [Source separation](https://github.com/TristanDonze/gb-sources-separation)
4. [Parameter estimation](https://github.com/TristanDonze/gb-parameters-estimation)

The project trains a PyTorch model to separate overlapping synthetic galactic-binary signals. Its approach is based on [FLOSS (FLOw matching for Source Separation)](https://arxiv.org/abs/2505.16119), adapted here to galactic-binary signals. It uses conditional flow matching and permutation-invariant source assignment, with an additional slot for residual noise.

## Setup

Python 3.12 or later and [uv](https://docs.astral.sh/uv/) are required. Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/TristanDonze/gb-sources-separation.git
cd gb-sources-separation
uv sync
```

The expected HDF5 datasets are produced by the [dataset generation repository](https://github.com/TristanDonze/gb-dataset-gen). Before training or evaluation, update the dataset paths and experiment settings in [`config.py`](config.py).

## Training

Start a training run with:

```bash
uv run python -u main.py
```

For a background run:

```bash
bash run_nohup.sh
```

Each run is saved under `runs/run_<timestamp>/`, including logs, checkpoints, the model definition, the configuration used, and reproducibility metadata. The main logs are `run.log` and `shell.log`.

## Evaluation

Run these notebooks in order:

1. [`notebooks/evaluate_model.ipynb`](notebooks/evaluate_model.ipynb): set the run directory, checkpoint, dataset, and evaluation parameters in the configuration cell, then execute all cells. The notebook writes a pickle file to `evaluation_results/`.
2. [`notebooks/plot_results.ipynb`](notebooks/plot_results.ipynb): set `PICKLE_PATH` to the file produced by the evaluation notebook, then execute all cells to inspect reconstructions and summary metrics.
