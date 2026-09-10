import sys
sys.path.append('..')

import pickle
import torch
import numpy as np
import importlib.util
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from src.losses import ReconstructionPIT_MSELoss, ReconstructionPIT_NMSELoss
from src.dataset import create_train_val_datasets
from src.training import build_initial_state
from src.utils import get_device
from config import (
    small_dataset_path,
    medium_dataset_path,
    large_dataset_path,
    huge_dataset_path,
    TRAIN_SIZE,
    SPLIT_STRATEGY,
    SPLIT_STRATEGY,
    SEED_TRAIN,
    SEED_VAL,
    SPLIT_SEED,
    MAX_K,
    CONSTANT_K,
    BATCH_SIZE,
    NB_OF_STEPS
)

device = get_device()
print(f"Using device : {device}") 

run_name = Path("runs/run_20260828_190432")

run_path =  run_name / "checkpoints"
model_structure_path = run_path / "model_structure.py"
used_config_path = run_path / "used_config.py"
ckpt_path = run_path / "best_checkpoint.pth"
dataset_path = medium_dataset_path

spec = importlib.util.spec_from_file_location(
    "dynamic_model", model_structure_path
)
dynamic_model_module = importlib.util.module_from_spec(spec)
sys.modules["dynamic_model"] = dynamic_model_module
spec.loader.exec_module(dynamic_model_module)

config_spec = importlib.util.spec_from_file_location("run_config", used_config_path)
run_config = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(run_config)


FlowSeparator = getattr(
    dynamic_model_module, "FlowSeparator"
)
model = FlowSeparator(max_k=run_config.MAX_K, n_blocks=run_config.N_BLOCKS).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {number_of_parameters}")

model.eval()

MIXTURES_INDEX = [0, 306, 51, 51, 51, 927, 65, 394, 468]
ELEMENT_INDEX = [0, 0, 4, 3, 0, 1, 4, 3, 2]

NUMBER_OF_SAMPLES = 1000
NB_OF_SAMPLING = 1000
ALIGN_TO_TARGET = False


_, val_dataset = create_train_val_datasets(
        dataset_path,
        train_size=run_config.TRAIN_SIZE,
        max_K=run_config.MAX_K,
        constant_K=run_config.CONSTANT_K,
        max_samples_train=0,
        max_samples_val=NUMBER_OF_SAMPLES,
        noise_train=True,
        noise_val=True,
        deterministic_train=False,
        deterministic_val=True,
        split_seed=run_config.SPLIT_SEED,
        split_strategy=run_config.SPLIT_STRATEGY,
        seed_train=run_config.SEED_TRAIN,
        seed_val=run_config.SEED_VAL,
        return_params=True,
)

NB_OF_STEPS = run_config.NB_OF_STEPS
dt = 1.0 / NB_OF_STEPS
generator = torch.Generator(device=device)
generator.manual_seed(0)

dataloader = DataLoader(
    val_dataset,
    batch_size=NUMBER_OF_SAMPLES,
    shuffle=False,
    num_workers=0
)


criterion = ReconstructionPIT_MSELoss(residual_weight=1.0)
nmse = ReconstructionPIT_NMSELoss()

T = 0.5 * (1 - np.cos(np.pi * np.linspace(0.0, 1.0, NB_OF_STEPS + 1)))


def _align_to_reference(prediction, reference, k):
    """Align the active source slots of one prediction to a reference."""
    cost = (
        prediction[:k, None] - reference[None, :k]
    ).square().mean(dim=(-1, -2))
    predicted_slots, reference_slots = linear_sum_assignment(
        cost.detach().cpu().numpy()
    )
    predicted_slots = predicted_slots.tolist()
    reference_slots = reference_slots.tolist()

    aligned = prediction.clone()
    aligned[reference_slots] = prediction[predicted_slots]
    # Inactive padded slots and the last residual slot are left unchanged.
    assignment = tuple(zip(predicted_slots, reference_slots))
    return aligned, assignment


def find_medoid_sampling(predictions, k):
    """Choose a medoid after putting all samplings in a common slot order."""
    print("Running medoid sampling search...")
    provisional_reference = predictions[0]
    provisionally_aligned = torch.stack([
        _align_to_reference(prediction, provisional_reference, k)[0]
        for prediction in predictions
    ])

    provisional_consensus = provisionally_aligned[:, :k].mean(dim=0)
    distances_to_consensus = (
        provisionally_aligned[:, :k] - provisional_consensus
    ).square().mean(dim=(1, 2, 3))

    # With squared Euclidean distances, the element closest to the mean also
    # minimizes its total distance to all other aligned elements.
    medoid_index = int(distances_to_consensus.argmin().item())
    return predictions[medoid_index], medoid_index


def align_sampling(predictions, k, max_iterations=20):
    """Align stochastic reconstructions using an iterative consensus.

    Parameters
    ----------
    predictions : Tensor, shape (S, L, C, F)
        The S reconstructions of the same mixture.
    k : int
        Number of active physical sources. The last slot is the residual.
    """
    print("Running iterative alignment...")
    # Start from the sampling which is globally closest to all the others.
    reference, medoid_index = find_medoid_sampling(predictions, k)
    previous_assignments = None

    for _ in range(max_iterations):
        aligned_predictions = []
        assignments = []

        for prediction in predictions:
            aligned, assignment = _align_to_reference(prediction, reference, k)
            aligned_predictions.append(aligned)
            assignments.append(assignment)

        aligned_predictions = torch.stack(aligned_predictions)
        reference = aligned_predictions.mean(dim=0)
        assignments = tuple(assignments)

        if assignments == previous_assignments:
            return aligned_predictions, reference, assignments, True, medoid_index

        previous_assignments = assignments

    return aligned_predictions, reference, assignments, False, medoid_index


def align_sampling_to_target(predictions, target, k):
    """Align every sampling directly to the known target sources."""
    print("Aligning samplings with target...")
    aligned_predictions = []
    assignments = []

    for prediction in predictions:
        aligned, assignment = _align_to_reference(prediction, target, k)
        aligned_predictions.append(aligned)
        assignments.append(assignment)

    return torch.stack(aligned_predictions), tuple(assignments)


def check_alignment_with_target(predictions, aligned_predictions, consensus, target, k):
    """Check the target-free alignment against the oracle target alignment."""
    # Consensus slot names are arbitrary. First map them once to target slots.
    print("Checking alignment with target...")
    cost = (
        consensus[:k, None] - target[None, :k]
    ).square().mean(dim=(-1, -2))
    consensus_slots, target_slots = linear_sum_assignment(
        cost.detach().cpu().numpy()
    )
    consensus_slots = consensus_slots.tolist()
    target_slots = target_slots.tolist()

    aligned_with_target_names = aligned_predictions.clone()
    aligned_with_target_names[:, target_slots] = aligned_predictions[:, consensus_slots]
    aligned_with_target_names[:, -1] = aligned_predictions[:, -1]

    oracle_predictions = torch.stack([
        _align_to_reference(prediction, target, k)[0]
        for prediction in predictions
    ])

    correctly_aligned = [
        torch.equal(aligned_with_target_names[s], oracle_predictions[s])
        for s in range(len(predictions))
    ]
    return correctly_aligned

results = []
total_loss = 0.0
total_samples = 0
generator.manual_seed(0)

elements = {}

for mixture, X_1, K, slot_mask, params in tqdm(dataloader, desc="Collecting Mixtures", leave=False):
    for i in range(mixture.shape[0]):
        if i in MIXTURES_INDEX:
            elements[i] = {
                "mixture": mixture[i],
                "X_1": X_1[i],
                "K": K[i],
                "slot_mask": slot_mask[i],
                "params": {key: params[key][i] for key in params.keys()}
            }

with torch.no_grad():
    for index in MIXTURES_INDEX:
        print(f"Evaluating mixture {index}...")
        mixture = elements[index]["mixture"].unsqueeze(0).to(device)
        X_1 = elements[index]["X_1"].unsqueeze(0).to(device)
        K = elements[index]["K"].unsqueeze(0).to(device)
        slot_mask = elements[index]["slot_mask"].unsqueeze(0).to(device)
        params = {key: elements[index]["params"][key] for key in elements[index]["params"].keys()}

        B = X_1.shape[0]
        effective_B = B * NB_OF_SAMPLING

        X_t = build_initial_state(
            mixture, X_1, K, slot_mask,
            nb_of_sampling=NB_OF_SAMPLING,
            generator=generator,
        )
        sampled_mixture = mixture.repeat_interleave(NB_OF_SAMPLING, dim=0)
        sampled_K = K.repeat_interleave(NB_OF_SAMPLING, dim=0)
        sampled_slot_mask = slot_mask.repeat_interleave(NB_OF_SAMPLING, dim=0)

        print(f"Starting sampling with {NB_OF_SAMPLING} samples...")

        for i in range(NB_OF_STEPS):
            t_val = float(T[i])
            dt = float(T[i + 1] - T[i])

            t = torch.full(
                (effective_B,),
                t_val,
                device=device,
                dtype=mixture.dtype,
            )

            v = model(X_t, t, sampled_mixture, sampled_K, sampled_slot_mask)
            X_t = X_t + dt * v  # (B * NB_OF_SAMPLING, L, C, F)
        print(f"Sampling completed for mixture {index}.")
        predictions = X_t.reshape(B, NB_OF_SAMPLING, *X_t.shape[1:])[0]
        k = int(K.item())

        if ALIGN_TO_TARGET:
            aligned_predictions, assignments = align_sampling_to_target(
                predictions, X_1[0], k
            )
            medoid_index = None
            converged = None
            correctly_aligned = [True] * NB_OF_SAMPLING
            print(
                f"Mixture {index}: aligned directly with target "
                f"({NB_OF_SAMPLING}/{NB_OF_SAMPLING})"
            )
        else:
            (
                aligned_predictions,
                consensus,
                assignments,
                converged,
                medoid_index,
            ) = align_sampling(predictions, k)

            correctly_aligned = check_alignment_with_target(
                predictions,
                aligned_predictions,
                consensus,
                X_1[0],
                k,
            )
            all_correct = all(correctly_aligned)
            print(
                f"Mixture {index}: medoid sampling={medoid_index}, "
                f"consensus converged={converged}, "
                f"alignment matches target={all_correct} "
                f"({sum(correctly_aligned)}/{NB_OF_SAMPLING})"
            )

        # Mean and standard deviation are meaningful only after slot alignment.
        avg_X_hat = aligned_predictions.mean(dim=0, keepdim=True)
        std_X_hat = aligned_predictions.std(dim=0, correction=0, keepdim=True)
        X_hat = avg_X_hat
        loss, source_loss, residual_loss = criterion(
            X_hat, X_1, slot_mask, reduction="none"
        )
        nmse_source, nmse_source_db = nmse(
            X_hat[:, :-1],
            X_1[:, :-1],
            slot_mask[:, :-1],
            reduction="none",
        )

        results.append({
            "mixture_index": index,
            "K": k,
            "align_to_target": ALIGN_TO_TARGET,
            "mixture": mixture[0].cpu(),
            "target": X_1[0].cpu(),
            "prediction_mean": avg_X_hat[0].cpu(),
            "prediction_std": std_X_hat[0].cpu(),
            "aligned_predictions": aligned_predictions.cpu(),
            "assignments": assignments,
            "medoid_sampling_index": medoid_index,
            "consensus_converged": converged,
            "alignment_matches_target": correctly_aligned,
            "loss": float(loss.item()),
            "source_loss": float(source_loss.item()),
            "residual_loss": float(residual_loss.item()),
            "nmse_source": nmse_source[0].cpu(),
            "nmse_source_db": nmse_source_db[0].cpu(),
            "params": params,
        })

    
results_path = Path(
    f"results_{run_name.name}_nb_of_samplings_{NB_OF_SAMPLING}_aligned_to_target_{ALIGN_TO_TARGET}.pkl"
)
with results_path.open("wb") as f:
    pickle.dump(results, f)
print(f"Saved {len(results)} mixtures to {results_path}")
