import os
from pathlib import Path

# Dataset paths
# SNR : 10 - 100
small_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_200K_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
medium_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_500K_uniform_SNR_10_100/dataset.hdf5")
large_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_3M_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
huge_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_10M_uniform_SNR_10_100/dataset.hdf5")

# Medium Config
# dataset_path = medium_dataset_path
# TRAIN_SIZE = 0.8
# MAX_SAMPLES_TRAIN = 400_000
# MAX_SAMPLES_VAL = 10_000

# Large Config
dataset_path = large_dataset_path
TRAIN_SIZE = 0.8
MAX_SAMPLES_TRAIN = 1_250_000
MAX_SAMPLES_VAL = 20_000

# Huge Config
# dataset_path = huge_dataset_path
# TRAIN_SIZE = 0.5
# MAX_SAMPLES_TRAIN = 5_000_000
# MAX_SAMPLES_VAL = 50_000

SPLIT_STRATEGY = "snr" # "random" or "snr"
FIX_ALL_SEEDS = True
SEED = 42
SEED_TRAIN = 42
SEED_VAL = 0
SPLIT_SEED = 2027

# Model Hyperparameters

MAX_K = 10
CONSTANT_K = False
K_TRAIN_PROBS = [0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.19, 0.24]
N_BLOCKS = 8
DROPOUT = 0.15
RESIDUAL_WEIGHT = 1.00
WARM_START_CHECKPOINT = Path(
    "runs/run_20260831_181224/checkpoints/best_checkpoint.pth"
)

# Training Hyperparameters

BATCH_SIZE = 212
NB_EPOCHS = 40
WEIGHT_DECAY = 3e-3
ENABLE_TIMING = False  # Synchronizes CUDA stages; disable for maximum throughput.

# Scheduler :
LR = 3e-4
LR_MIN = 1e-6
LR_DECAY_EPOCHS = 40
FACTOR = 0.5
PATIENCE = 3

# Validation Hyperparameters

NB_OF_STEPS = 32
