#!/usr/bin/env python3
"""Prepare raw metrics for protocol steps 1 to 4."""

import argparse
import csv
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import create_train_val_datasets
from src.masking import mask_slots
from src.training import build_initial_state
from src.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/run_20260903_170728"))
    parser.add_argument("--checkpoint", default="best_checkpoint.pth")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--mixtures", type=int, default=10_000)
    parser.add_argument("--samplings", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of distinct mixtures per batch; effective batch is this times samplings.",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--recompute-overlap-only",
        action="store_true",
        help="Recompute the effective-support overlap in existing CSV files without inference.",
    )
    return parser.parse_args()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_model(run_dir, checkpoint_name, device):
    checkpoint_dir = run_dir / "checkpoints"
    model_module = load_module(
        f"evaluation_model_{run_dir.name}", checkpoint_dir / "model_structure.py"
    )
    run_config = load_module(
        f"evaluation_config_{run_dir.name}", checkpoint_dir / "used_config.py"
    )
    model = model_module.FlowSeparator(
        max_k=run_config.MAX_K,
        n_blocks=run_config.N_BLOCKS,
    ).to(device)
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    accepts_mask = "slot_mask" in inspect.signature(model.forward).parameters
    human_epoch = int(checkpoint["epoch"]) + 1
    return model, run_config, accepts_mask, checkpoint_path, human_epoch


def make_validation_dataset(run_config, dataset_path, mixtures, return_params):
    _, dataset = create_train_val_datasets(
        dataset_path=dataset_path,
        train_size=run_config.TRAIN_SIZE,
        max_K=run_config.MAX_K,
        constant_K=run_config.CONSTANT_K,
        max_samples_train=0,
        max_samples_val=mixtures,
        noise_train=True,
        noise_val=True,
        deterministic_train=False,
        deterministic_val=True,
        split_seed=run_config.SPLIT_SEED,
        split_strategy=run_config.SPLIT_STRATEGY,
        seed_train=run_config.SEED_TRAIN,
        seed_val=run_config.SEED_VAL,
        return_params=return_params,
    )
    return dataset


def integrate_samples(
    model,
    mixture,
    target,
    source_count,
    slot_mask,
    samplings,
    steps,
    generator,
    accepts_mask,
):
    batch_size = len(mixture)
    effective_batch = batch_size * samplings
    state = build_initial_state(
        mixture,
        target,
        source_count,
        slot_mask,
        nb_of_sampling=samplings,
        generator=generator,
    )
    sampled_mixture = mixture.repeat_interleave(samplings, dim=0)
    sampled_k = source_count.repeat_interleave(samplings)
    sampled_mask = slot_mask.repeat_interleave(samplings, dim=0)
    time_grid = 0.5 * (
        1
        - torch.cos(
            torch.linspace(0.0, torch.pi, steps + 1, device=mixture.device)
        )
    )

    for step in range(steps):
        t = torch.full(
            (effective_batch,),
            time_grid[step].item(),
            device=mixture.device,
            dtype=mixture.dtype,
        )
        if accepts_mask:
            velocity = model(state, t, sampled_mixture, sampled_k, sampled_mask)
        else:
            velocity = model(state, t, sampled_mixture, sampled_k)
        state = mask_slots(
            state + (time_grid[step + 1] - time_grid[step]) * velocity,
            sampled_mask,
        )

    return state.reshape(batch_size, samplings, *state.shape[1:])


def align_samples_to_target(predictions, target_sources):
    source_count = len(target_sources)
    aligned = np.empty(
        (len(predictions), *target_sources.shape), dtype=predictions.dtype
    )
    for sample_index, prediction in enumerate(predictions):
        predicted_sources = prediction[:source_count]
        cost = np.mean(
            (predicted_sources[:, None] - target_sources[None]) ** 2,
            axis=(-1, -2),
        )
        predicted_slots, target_slots = linear_sum_assignment(cost)
        aligned[sample_index, target_slots] = predicted_sources[predicted_slots]
    return aligned


def effective_frequency_bounds(target_sources, relative_threshold=1e-2):
    energy = np.sum(target_sources.astype(np.float64) ** 2, axis=1)
    starts = []
    stops = []
    for source_energy in energy:
        max_energy = float(source_energy.max())
        active = (
            np.flatnonzero(source_energy >= relative_threshold * max_energy)
            if max_energy > 0.0
            else np.array([], dtype=int)
        )
        if len(active):
            starts.append(int(active[0]))
            stops.append(int(active[-1]) + 1)
        else:
            starts.append(-1)
            stops.append(-1)
    return starts, stops


def interval_union_length(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)
    current_start, current_stop = intervals[0]
    total = 0
    for start, stop in intervals[1:]:
        if start <= current_stop:
            current_stop = max(current_stop, stop)
        else:
            total += current_stop - current_start
            current_start, current_stop = start, stop
    return total + current_stop - current_start


def mean_spectral_support_coverage(starts, stops):
    intervals = [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if start >= 0 and stop > start
    ]
    if len(intervals) < 2:
        return 0.0

    coverages = []
    for index, (start, stop) in enumerate(intervals):
        intersections = [
            (max(start, other_start), min(stop, other_stop))
            for other_index, (other_start, other_stop) in enumerate(intervals)
            if index != other_index
            and max(start, other_start) < min(stop, other_stop)
        ]
        coverages.append(interval_union_length(intersections) / (stop - start))
    return float(np.mean(coverages))


def mean_pairwise_spectral_coverage(starts, stops):
    intervals = [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if start >= 0 and stop > start
    ]
    source_count = len(intervals)
    if source_count < 2:
        return 0.0

    pairwise_coverages = []
    for index, (start, stop) in enumerate(intervals):
        width = stop - start
        for other_index, (other_start, other_stop) in enumerate(intervals):
            if index == other_index:
                continue
            intersection = max(
                0,
                min(stop, other_stop) - max(start, other_start),
            )
            pairwise_coverages.append(intersection / width)
    return float(np.mean(pairwise_coverages))


def rewrite_overlap_columns(path, overlaps, pairwise_overlaps):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with path.open(newline="") as source, temporary_path.open("w", newline="") as target:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames)
        if "frequency_overlap_pairwise" not in fieldnames:
            fieldnames.append("frequency_overlap_pairwise")
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            mixture_id = int(row["mixture_id"])
            row["frequency_overlap"] = overlaps[mixture_id]
            row["frequency_overlap_pairwise"] = pairwise_overlaps[mixture_id]
            writer.writerow(row)
    os.replace(temporary_path, path)


def recompute_existing_overlaps(args):
    run_dir = args.run.resolve()
    run_config = load_module(
        f"overlap_config_{run_dir.name}",
        run_dir / "checkpoints" / "used_config.py",
    )
    dataset_path = (args.dataset or Path(run_config.dataset_path)).resolve()
    if args.output_dir is None:
        candidates = sorted(
            (REPO_ROOT / "experiments" / "results").glob(f"{run_dir.name}_epoch_*")
        )
        if len(candidates) != 1:
            raise RuntimeError("Pass --output-dir when the result directory is ambiguous")
        output_dir = candidates[0]
    else:
        output_dir = args.output_dir.resolve()

    metadata_path = output_dir / "metadata_steps_1_to_4.json"
    with metadata_path.open() as file:
        metadata = json.load(file)
    mixture_count = int(metadata["mixtures"])
    dataset = make_validation_dataset(
        run_config, dataset_path, mixture_count, return_params=False
    )

    overlaps = {}
    pairwise_overlaps = {}
    for mixture_id in range(mixture_count):
        _, target, source_count, _ = dataset[mixture_id]
        k = int(source_count)
        starts, stops = effective_frequency_bounds(target[:k])
        overlaps[mixture_id] = mean_spectral_support_coverage(starts, stops)
        pairwise_overlaps[mixture_id] = mean_pairwise_spectral_coverage(
            starts, stops
        )
        if (mixture_id + 1) % 500 == 0:
            print(
                f"Overlap: {mixture_id + 1}/{mixture_count} mixtures",
                flush=True,
            )

    for filename in ("nmse_by_sampling_and_source.csv", "uncertainty_by_source.csv"):
        rewrite_overlap_columns(
            output_dir / filename,
            overlaps,
            pairwise_overlaps,
        )
    metadata["frequency_support"] = (
        "first/last bins with energy >= 1% of the source maximum"
    )
    metadata["frequency_support_relative_threshold"] = 1e-2
    metadata["frequency_overlap_pairwise"] = (
        "mean ordered-pair coverage: mean_i mean_j!=i |I_i intersect I_j| / |I_i|"
    )
    with metadata_path.open("w") as file:
        json.dump(metadata, file, indent=2)
    print(f"Corrected overlap in {output_dir}", flush=True)


def ensure_output_files(output_dir, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "nmse": output_dir / "nmse_by_sampling_and_source.csv",
        "uncertainty": output_dir / "uncertainty_by_source.csv",
        "metadata": output_dir / "metadata_steps_1_to_4.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {names}; use --overwrite")
    return paths


def main():
    args = parse_args()
    if args.recompute_overlap_only:
        recompute_existing_overlaps(args)
        return

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

    paths = ensure_output_files(output_dir, args.overwrite)
    dataset = make_validation_dataset(
        run_config, dataset_path, args.mixtures, return_params=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    nmse_fields = [
        "mixture_id", "sampling", "K", "source", "snr", "snr_min",
        "snr_max", "snr_mean", "snr_gap", "frequency_overlap",
        "frequency_overlap_pairwise", "nmse",
    ]
    uncertainty_fields = [
        "mixture_id", "K", "source", "snr", "snr_min", "snr_max",
        "snr_mean", "snr_gap", "frequency_overlap",
        "frequency_overlap_pairwise", "u_pred", "u_true",
    ]

    with (
        paths["nmse"].open("w", newline="") as nmse_file,
        paths["uncertainty"].open("w", newline="") as uncertainty_file,
    ):
        nmse_writer = csv.DictWriter(nmse_file, fieldnames=nmse_fields)
        uncertainty_writer = csv.DictWriter(
            uncertainty_file, fieldnames=uncertainty_fields
        )
        nmse_writer.writeheader()
        uncertainty_writer.writeheader()
        mixture_offset = 0

        with torch.inference_mode():
            for batch_index, (mixture, target, source_count, slot_mask, params) in enumerate(
                loader, start=1
            ):
                mixture = mixture.to(device)
                target_device = target.to(device)
                source_count_device = source_count.to(device)
                slot_mask_device = slot_mask.to(device)
                predictions = integrate_samples(
                    model,
                    mixture,
                    target_device,
                    source_count_device,
                    slot_mask_device,
                    args.samplings,
                    steps,
                    generator,
                    accepts_mask,
                ).cpu().numpy()
                targets = target.numpy()
                source_counts = source_count.numpy()
                snr_batch = params["snr"].numpy()

                for local_index, (prediction, target_item, k_value, snrs) in enumerate(
                    zip(predictions, targets, source_counts, snr_batch)
                ):
                    mixture_id = mixture_offset + local_index
                    k = int(k_value)
                    target_sources = target_item[:k]
                    source_snrs = np.asarray(snrs[:k], dtype=float)
                    starts, stops = effective_frequency_bounds(target_sources)
                    overlap = mean_spectral_support_coverage(starts, stops)
                    pairwise_overlap = mean_pairwise_spectral_coverage(
                        starts, stops
                    )
                    snr_stats = {
                        "snr_min": float(source_snrs.min()),
                        "snr_max": float(source_snrs.max()),
                        "snr_mean": float(source_snrs.mean()),
                        "snr_gap": float(source_snrs.max() - source_snrs.min()),
                    }
                    aligned = align_samples_to_target(prediction, target_sources)
                    error_energy = np.sum(
                        (aligned - target_sources[None]) ** 2, axis=(-1, -2)
                    )
                    target_energy = np.sum(target_sources**2, axis=(-1, -2))
                    nmse = error_energy / (target_energy[None] + 1e-12)

                    consensus = aligned.mean(axis=0)
                    dispersion_rms = np.sqrt(
                        np.mean((aligned - consensus[None]) ** 2, axis=(0, 2, 3))
                    )
                    prediction_rms = np.sqrt(np.mean(aligned**2, axis=(0, 2, 3)))
                    target_rms = np.sqrt(np.mean(target_sources**2, axis=(1, 2)))
                    u_pred = dispersion_rms / (prediction_rms + 1e-12)
                    u_true = dispersion_rms / (target_rms + 1e-12)

                    common = {
                        "mixture_id": mixture_id,
                        "K": k,
                        "frequency_overlap": overlap,
                        "frequency_overlap_pairwise": pairwise_overlap,
                        **snr_stats,
                    }
                    for sampling in range(args.samplings):
                        for source in range(k):
                            nmse_writer.writerow(
                                {
                                    **common,
                                    "sampling": sampling,
                                    "source": source,
                                    "snr": source_snrs[source],
                                    "nmse": float(nmse[sampling, source]),
                                }
                            )
                    for source in range(k):
                        uncertainty_writer.writerow(
                            {
                                **common,
                                "source": source,
                                "snr": source_snrs[source],
                                "u_pred": float(u_pred[source]),
                                "u_true": float(u_true[source]),
                            }
                        )

                mixture_offset += len(targets)
                nmse_file.flush()
                uncertainty_file.flush()
                print(
                    f"Steps 1-4: batch {batch_index}/{len(loader)} "
                    f"({mixture_offset}/{len(dataset)} mixtures)",
                    flush=True,
                )

    metadata = {
        "run": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "dataset": str(dataset_path),
        "mixtures": len(dataset),
        "samplings": args.samplings,
        "steps": steps,
        "seed": args.seed,
        "nmse_file": paths["nmse"].name,
        "uncertainty_file": paths["uncertainty"].name,
        "snr_plot_default": "snr_gap",
        "frequency_support": "first/last bins with energy >= 1% of the source maximum",
        "frequency_support_relative_threshold": 1e-2,
        "frequency_overlap_pairwise": (
            "mean ordered-pair coverage: mean_i mean_j!=i "
            "|I_i intersect I_j| / |I_i|"
        ),
    }
    with paths["metadata"].open("w") as file:
        json.dump(metadata, file, indent=2)
    print("Steps 1-4 completed.", flush=True)


if __name__ == "__main__":
    main()
