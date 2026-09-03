#!/usr/bin/env python3
"""Run sampling inference and estimate the useful number of samples."""

import argparse
import csv
import importlib.util
import inspect
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import medium_dataset_path
from src.dataset import create_train_val_datasets
from src.masking import mask_slots
from src.training import build_initial_state
from src.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/run_20260828_190432"),
        help="Run directory containing checkpoints/ (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(medium_dataset_path),
        help="HDF5 dataset used for the benchmark",
    )
    parser.add_argument("--mixtures", type=int, default=512)
    parser.add_argument(
        "--max-sampling",
        type=int,
        default=20,
        help="Generate this many initial states per mixture (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of distinct mixtures per inference batch (default: 128)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Random subset orderings used for every ensemble size (default: 50)",
    )
    parser.add_argument(
        "--tolerance-db",
        type=float,
        default=0.1,
        help="Maximum acceptable gap to the max-sampling result (default: 0.1 dB)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="For example cuda:0 or cpu")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("sampling_benchmark"),
        help="Prefix for the generated CSV and PNG files",
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


def load_model(run_dir, device):
    checkpoint_dir = run_dir / "checkpoints"
    model_module = load_module(
        f"sampling_benchmark_model_{run_dir.name}",
        checkpoint_dir / "model_structure.py",
    )
    run_config = load_module(
        f"sampling_benchmark_config_{run_dir.name}",
        checkpoint_dir / "used_config.py",
    )
    model = model_module.FlowSeparator(
        max_k=run_config.MAX_K,
        n_blocks=run_config.N_BLOCKS,
    ).to(device)
    checkpoint = torch.load(
        checkpoint_dir / "best_checkpoint.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    accepts_mask = "slot_mask" in inspect.signature(model.forward).parameters
    return model, run_config, accepts_mask


def create_validation_loader(args, run_config):
    _, validation_dataset = create_train_val_datasets(
        dataset_path=args.dataset,
        train_size=run_config.TRAIN_SIZE,
        max_K=run_config.MAX_K,
        constant_K=run_config.CONSTANT_K,
        max_samples_train=0,
        max_samples_val=args.mixtures,
        noise_train=True,
        noise_val=True,
        deterministic_train=False,
        deterministic_val=True,
        split_seed=run_config.SPLIT_SEED,
        split_strategy=run_config.SPLIT_STRATEGY,
        seed_train=run_config.SEED_TRAIN,
        seed_val=run_config.SEED_VAL,
        return_params=False,
    )
    return DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )


def align_sources(predictions, target, k):
    """Align (sampling, slots, channels, frequencies) to target sources."""
    target_sources = target[:k]
    aligned = np.empty(
        (len(predictions), *target_sources.shape), dtype=predictions.dtype
    )
    for sampling_index, prediction in enumerate(predictions):
        predicted_sources = prediction[:k]
        cost = np.mean(
            (predicted_sources[:, None] - target_sources[None]) ** 2,
            axis=(-1, -2),
        )
        predicted_slots, target_slots = linear_sum_assignment(cost)
        aligned[sampling_index, target_slots] = predicted_sources[predicted_slots]
    return aligned


def accumulate_ensemble_scores(score_sums, predictions, targets, source_counts, rng):
    trials, max_sampling = score_sums.shape
    denominators = np.arange(1, max_sampling + 1, dtype=np.float32)[
        None, :, None, None, None
    ]
    total_sources = 0

    for prediction, target, k_value in zip(predictions, targets, source_counts):
        k = int(k_value)
        aligned = align_sources(prediction, target, k)
        orders = np.stack([rng.permutation(max_sampling) for _ in range(trials)])
        # A prefix of a random permutation is a random subset of the given size.
        ensemble_means = np.cumsum(aligned[orders], axis=1) / denominators
        target_sources = target[:k]
        error_energy = np.sum(
            (ensemble_means - target_sources[None, None]) ** 2,
            axis=(-1, -2),
        )
        target_energy = np.sum(target_sources**2, axis=(-1, -2))
        nmse_db = 10 * np.log10(
            error_energy / (target_energy[None, None] + 1e-12) + 1e-12
        )
        score_sums += nmse_db.sum(axis=2)
        total_sources += k
    return total_sources


def run_benchmark(model, loader, run_config, accepts_mask, args, device):
    max_sampling = args.max_sampling
    score_sums = np.zeros((args.trials, max_sampling), dtype=np.float64)
    total_sources = 0
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    time_grid = 0.5 * (
        1
        - torch.cos(
            torch.linspace(0.0, torch.pi, run_config.NB_OF_STEPS + 1, device=device)
        )
    )

    with torch.inference_mode():
        for batch_index, (mixture, X_1, K, slot_mask) in enumerate(loader, start=1):
            mixture = mixture.to(device)
            X_1 = X_1.to(device)
            K = K.to(device)
            slot_mask = slot_mask.to(device)
            batch_size = len(X_1)
            effective_batch_size = batch_size * max_sampling

            X_t = build_initial_state(
                mixture,
                X_1,
                K,
                slot_mask,
                nb_of_sampling=max_sampling,
                generator=generator,
            )
            sampled_mixture = mixture.repeat_interleave(max_sampling, dim=0)
            sampled_K = K.repeat_interleave(max_sampling, dim=0)
            sampled_slot_mask = slot_mask.repeat_interleave(max_sampling, dim=0)

            for step in range(run_config.NB_OF_STEPS):
                t = torch.full(
                    (effective_batch_size,),
                    time_grid[step].item(),
                    device=device,
                    dtype=mixture.dtype,
                )
                if accepts_mask:
                    velocity = model(
                        X_t, t, sampled_mixture, sampled_K, sampled_slot_mask
                    )
                else:
                    velocity = model(X_t, t, sampled_mixture, sampled_K)
                dt = time_grid[step + 1] - time_grid[step]
                X_t = mask_slots(X_t + dt * velocity, sampled_slot_mask)

            predictions = X_t.reshape(
                batch_size, max_sampling, *X_t.shape[1:]
            ).cpu().numpy()
            total_sources += accumulate_ensemble_scores(
                score_sums,
                predictions,
                X_1.cpu().numpy(),
                K.cpu().numpy(),
                rng,
            )
            processed = min(batch_index * args.batch_size, args.mixtures)
            print(
                f"Inference batch {batch_index}/{len(loader)} "
                f"({processed}/{args.mixtures} mixtures)"
            )
    return score_sums / total_sources


def summarize(scores, tolerance_db):
    gaps = scores - scores[:, -1, None]
    gaps[:, -1] = 0.0
    mean_score = scores.mean(axis=0)
    mean_gap = gaps.mean(axis=0)
    gap_p95 = np.quantile(gaps, 0.95, axis=0)
    next_gain = np.r_[mean_score[:-1] - mean_score[1:], np.nan]

    optimal = len(mean_score)
    for index in range(len(mean_score)):
        # Requiring the full tail avoids selecting an accidental local dip.
        if np.all(gap_p95[index:] <= tolerance_db):
            optimal = index + 1
            break

    rows = [
        {
            "nb_sampling": index + 1,
            "source_nmse_db": mean_score[index],
            "gap_to_max_sampling_db": mean_gap[index],
            "gap_p95_db": gap_p95[index],
            "gain_from_next_sample_db": next_gain[index],
        }
        for index in range(len(mean_score))
    ]
    return rows, optimal


def print_results(rows, optimal, args):
    print(
        f"\n{'N':>4} {'NMSE (dB)':>12} {'gap/max':>12} "
        f"{'gap p95':>12} {'gain N+1':>12}"
    )
    for row in rows:
        gain = row["gain_from_next_sample_db"]
        gain_text = "-" if np.isnan(gain) else f"{gain:.4f}"
        print(
            f"{row['nb_sampling']:4d} {row['source_nmse_db']:12.4f} "
            f"{row['gap_to_max_sampling_db']:12.4f} "
            f"{row['gap_p95_db']:12.4f} {gain_text:>12}"
        )
    print(
        f"\nRecommended NB_OF_SAMPLING = {optimal} "
        f"(95th-percentile gap <= {args.tolerance_db:g} dB relative to "
        f"NB_OF_SAMPLING={args.max_sampling})"
    )


def save_results(rows, optimal, args):
    csv_path = args.output_prefix.with_suffix(".csv")
    plot_path = args.output_prefix.with_suffix(".png")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = np.asarray([row["nb_sampling"] for row in rows])
    nmse = np.asarray([row["source_nmse_db"] for row in rows])
    gap_p95 = np.asarray([row["gap_p95_db"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x, nmse, marker="o")
    axes[0].axvline(
        optimal, color="tab:red", linestyle="--", label=f"optimal={optimal}"
    )
    axes[0].set(xlabel="NB_OF_SAMPLING", ylabel="Mean source NMSE (dB)")
    axes[0].legend()
    axes[1].plot(x, gap_p95, marker="o")
    axes[1].axhline(
        args.tolerance_db, color="tab:red", linestyle="--", label="tolerance"
    )
    axes[1].axvline(optimal, color="tab:red", linestyle="--")
    axes[1].set(
        xlabel="NB_OF_SAMPLING",
        ylabel="95th-percentile gap to maximum (dB)",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.set_xticks(x)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved {csv_path} and {plot_path}")


def validate_args(args):
    for name in ("mixtures", "max_sampling", "batch_size", "trials"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_sampling < 2:
        raise ValueError("--max-sampling must be at least 2")
    if args.tolerance_db < 0:
        raise ValueError("--tolerance-db cannot be negative")


def main():
    args = parse_args()
    validate_args(args)
    run_dir = args.run if args.run.is_absolute() else REPO_ROOT / args.run
    device = torch.device(args.device or get_device())
    torch.manual_seed(args.seed)
    print(f"Device: {device}")
    print(
        f"Benchmark: {args.mixtures} mixtures x {args.max_sampling} samplings, "
        f"batch size {args.batch_size} mixtures "
        f"(effective batch size {args.batch_size * args.max_sampling})"
    )

    model, run_config, accepts_mask = load_model(run_dir, device)
    loader = create_validation_loader(args, run_config)
    scores = run_benchmark(model, loader, run_config, accepts_mask, args, device)
    rows, optimal = summarize(scores, args.tolerance_db)
    print_results(rows, optimal, args)
    save_results(rows, optimal, args)


if __name__ == "__main__":
    main()
