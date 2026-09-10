#!/usr/bin/env python3
"""Prepare iterative-consensus metrics for protocol step 5."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset

from prepare_protocol_results import (
    REPO_ROOT,
    integrate_samples,
    load_model,
    make_validation_dataset,
)
from src.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/run_20260903_170728"))
    parser.add_argument("--checkpoint", default="best_checkpoint.pth")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--mixtures", type=int, default=1_000)
    parser.add_argument("--samplings", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def align_to_reference(prediction, reference):
    cost = np.mean(
        (prediction[:, None] - reference[None]) ** 2,
        axis=(-1, -2),
    )
    predicted_slots, reference_slots = linear_sum_assignment(cost)
    aligned = np.empty_like(prediction)
    aligned[reference_slots] = prediction[predicted_slots]
    return aligned, tuple(zip(predicted_slots.tolist(), reference_slots.tolist()))


def iterative_consensus(predictions, max_iterations=20):
    provisional = np.stack(
        [align_to_reference(prediction, predictions[0])[0] for prediction in predictions]
    )
    provisional_mean = provisional.mean(axis=0)
    medoid_index = int(
        np.mean((provisional - provisional_mean[None]) ** 2, axis=(1, 2, 3)).argmin()
    )
    reference = predictions[medoid_index]
    previous_assignments = None

    for iteration in range(1, max_iterations + 1):
        aligned_and_assignments = [
            align_to_reference(prediction, reference) for prediction in predictions
        ]
        aligned = np.stack([item[0] for item in aligned_and_assignments])
        assignments = tuple(item[1] for item in aligned_and_assignments)
        reference = aligned.mean(axis=0)
        if assignments == previous_assignments:
            return aligned, reference, True, iteration, medoid_index
        previous_assignments = assignments
    return aligned, reference, False, max_iterations, medoid_index


def cardinality_aware_nmse(prediction, target):
    predicted_k, channels, frequencies = prediction.shape
    true_k = len(target)
    size = max(predicted_k, true_k)
    padded_prediction = np.zeros((size, channels, frequencies), dtype=prediction.dtype)
    padded_target = np.zeros((size, channels, frequencies), dtype=target.dtype)
    padded_prediction[:predicted_k] = prediction
    padded_target[:true_k] = target
    cost = np.mean(
        (padded_prediction[:, None] - padded_target[None]) ** 2,
        axis=(-1, -2),
    )
    predicted_slots, target_slots = linear_sum_assignment(cost)
    error_energy = np.sum(
        (padded_prediction[predicted_slots] - padded_target[target_slots]) ** 2
    )
    return float(error_energy / (np.sum(target**2) + 1e-12))


def main():
    args = parse_args()
    run_dir = args.run.resolve()
    device = torch.device(args.device) if args.device else get_device()
    model, run_config, accepts_mask, checkpoint_path, epoch = load_model(
        run_dir, args.checkpoint, device
    )
    steps = args.steps or run_config.NB_OF_STEPS
    dataset_path = (args.dataset or Path(run_config.dataset_path)).resolve()
    output_dir = (
        args.output_dir
        or REPO_ROOT / "experiments" / "results" / f"{run_dir.name}_epoch_{epoch}"
    ).resolve()

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path} (epoch {epoch})")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    if args.dry_run:
        print("Dry run completed after loading the model.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "wrong_k_iterative_consensus.csv"
    metadata_path = output_dir / "metadata_step_5.json"
    if (result_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Step-5 output already exists in {output_dir}")

    candidate_count = max(args.mixtures * 2, args.mixtures + 100)
    full_dataset = make_validation_dataset(
        run_config, dataset_path, candidate_count, return_params=False
    )
    eligible_indices = [
        index
        for index, (_, k) in enumerate(full_dataset.fixed_mixtures)
        if args.min_k <= int(k) <= args.max_k
    ][: args.mixtures]
    if len(eligible_indices) < args.mixtures:
        raise RuntimeError(
            f"Only {len(eligible_indices)} eligible mixtures found; "
            f"increase candidate_count for K in [{args.min_k}, {args.max_k}]"
        )
    dataset = Subset(full_dataset, eligible_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    fields = [
        "mixture_id", "true_K", "given_K", "delta_K", "u_pred", "u_true",
        "nmse", "consensus_converged", "consensus_iterations", "medoid_sampling",
    ]
    with result_path.open("w", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fields)
        writer.writeheader()
        mixture_offset = 0

        with torch.inference_mode():
            for batch_index, (mixture, target, true_k, original_mask) in enumerate(
                loader, start=1
            ):
                mixture = mixture.to(device)
                target_device = target.to(device)
                true_k_device = true_k.to(device)
                batch_size = len(mixture)
                batch_rows = []

                for delta in (-1, 0, 1):
                    given_k = true_k_device + delta
                    slot_mask = torch.zeros_like(original_mask, device=device)
                    source_slots = torch.arange(
                        slot_mask.shape[1] - 1, device=device
                    )[None]
                    slot_mask[:, :-1] = source_slots < given_k[:, None]
                    slot_mask[:, -1] = True
                    generator = torch.Generator(device=device).manual_seed(
                        args.seed + batch_index - 1
                    )
                    predictions = integrate_samples(
                        model,
                        mixture,
                        target_device,
                        given_k,
                        slot_mask,
                        args.samplings,
                        steps,
                        generator,
                        accepts_mask,
                    ).cpu().numpy()

                    for local_index in range(batch_size):
                        k = int(true_k[local_index])
                        inferred_k = k + delta
                        predicted_sources = predictions[local_index, :, :inferred_k]
                        aligned, consensus, converged, iterations, medoid = (
                            iterative_consensus(predicted_sources)
                        )
                        deviations = aligned - consensus[None]
                        dispersion_energy = float(
                            np.mean(np.sum(deviations**2, axis=(1, 2, 3)))
                        )
                        prediction_energy = float(
                            np.mean(np.sum(aligned**2, axis=(1, 2, 3)))
                        )
                        target_sources = target[local_index, :k].numpy()
                        target_energy = float(np.sum(target_sources**2))
                        batch_rows.append(
                            {
                                "mixture_id": mixture_offset + local_index,
                                "true_K": k,
                                "given_K": inferred_k,
                                "delta_K": delta,
                                "u_pred": np.sqrt(
                                    dispersion_energy / (prediction_energy + 1e-12)
                                ),
                                "u_true": np.sqrt(
                                    dispersion_energy / (target_energy + 1e-12)
                                ),
                                "nmse": cardinality_aware_nmse(
                                    consensus, target_sources
                                ),
                                "consensus_converged": converged,
                                "consensus_iterations": iterations,
                                "medoid_sampling": medoid,
                            }
                        )

                writer.writerows(batch_rows)
                result_file.flush()
                mixture_offset += batch_size
                print(
                    f"Step 5: batch {batch_index}/{len(loader)} "
                    f"({mixture_offset}/{len(dataset)} mixtures)",
                    flush=True,
                )

    metadata = {
        "run": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "dataset": str(dataset_path),
        "mixtures": len(dataset),
        "samplings_per_condition": args.samplings,
        "steps": steps,
        "seed": args.seed,
        "true_k_range": [args.min_k, args.max_k],
        "conditions": [-1, 0, 1],
        "alignment": "iterative consensus initialized by a medoid",
        "nmse": "global cardinality-aware NMSE with zero padding",
        "result_file": result_path.name,
    }
    with metadata_path.open("w") as file:
        json.dump(metadata, file, indent=2)
    print("Step 5 completed.", flush=True)


if __name__ == "__main__":
    main()
