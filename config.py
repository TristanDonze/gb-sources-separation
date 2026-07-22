import os
from pathlib import Path

# Dataset paths
# SNR : 10 - 100
small_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_200K_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
medium_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_500K_uniform_SNR_10_100/dataset.hdf5")
large_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_3M_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
huge_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_10M_uniform_SNR_10_100/dataset.hdf5")

# Medium Config
dataset_path = medium_dataset_path
TRAIN_SIZE = 0.8
MAX_SAMPLES_TRAIN = 400_000
MAX_SAMPLES_VAL = 10_000

# Huge Config
# dataset_path = huge_dataset_path
# TRAIN_SIZE = 0.5
# MAX_SAMPLES_TRAIN = 3_000_000
# MAX_SAMPLES_VAL = 50_000

SPLIT_STRATEGY = "snr" # "random" or "snr"
FIX_ALL_SEEDS = True
SEED = 42

SEED_TRAIN = 42
SEED_VAL = 0
SPLIT_SEED = 2027

# Training Hyperparameters 

MAX_K = 2
CONSTANT_K = 2
TRAIN_ALPHA_SNR = 1.0
BATCH_SIZE = 480
WEIGHT_DECAY = 1e-4
NB_EPOCHS = 30
ENABLE_TIMING = False  # Synchronizes CUDA stages; disable for maximum throughput.

# Scheduler : 

LR = 1e-3
LR_MIN = 1e-6
FACTOR = 0.5
PATIENCE = 5

# Validation Hyperparameters

NB_OF_STEPS = 32
