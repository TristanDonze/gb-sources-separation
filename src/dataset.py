import h5py
import logging
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger("Dataset")

FREQUENCY_SUPPORT_REL_THRESHOLD = 1e-2


class GalacticBinariesDataset(Dataset):
    def __init__(
        self,
        waveforms,
        params_dict,
        indices: np.ndarray | None = None,
        max_K: int = 10,
        constant_K: bool | int = False,
        noise: bool = True,
        max_samples: int | None = None,
        return_params: bool = False,
        returns_snr: bool = False,
        deterministic: bool = False,
        seed: int | None = None,
        k_sampling_probs=None,
    ):
        self.waveforms = waveforms
        self.attr_names = list(params_dict.keys())
        for key, value in params_dict.items():
            setattr(self, key, value)

        self.indices = self._normalize_indices(indices, len(waveforms))
        self.nb_selected_waveforms = len(self.indices)
        self.total_waveforms = self.nb_selected_waveforms
        self.length = (
            self.nb_selected_waveforms
            if max_samples is None
            else min(max_samples, self.nb_selected_waveforms)
        )

        self.max_K = max_K
        self.constant_K = constant_K
        self.noise = noise
        self.max_samples = max_samples
        self.return_params = return_params
        self.returns_snr = returns_snr
        self.deterministic = deterministic
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.k_sampling_probs = self._validate_k_sampling_probs(
            k_sampling_probs, max_K
        )

        if constant_K is True:
            raise ValueError("constant_K cannot be True. It must be either False or an integer value.")

        self.num_source_slots = (
            self.max_K if self.constant_K is False else int(self.constant_K)
        )

        if self.max_K > self.nb_selected_waveforms:
            raise ValueError(
                f"max_K={self.max_K} cannot be greater than "
                f"selected waveforms={self.nb_selected_waveforms}"
            )

        if self.returns_snr is True and self.return_params is True:
            raise ValueError("returns_snr and return_params cannot both be True. Choose one.")

        if self.deterministic:
            logger.info("Building fixed mixtures for deterministic sampling...")
            self.fixed_mixtures = self._build_fixed_mixtures()
            logger.info("Fixed mixtures built successfully.")
        else:
            self.fixed_mixtures = None

    @staticmethod
    def _normalize_indices(indices, total_waveforms):
        if indices is None:
            return np.arange(total_waveforms, dtype=np.int64)

        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("indices must be a 1D array")
        if len(indices) == 0:
            raise ValueError("indices cannot be empty")
        if np.any(indices < 0) or np.any(indices >= total_waveforms):
            raise ValueError(f"indices must be between 0 and {total_waveforms - 1}")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("indices must be unique")

        return np.sort(indices)

    @staticmethod
    def _validate_k_sampling_probs(k_sampling_probs, max_K):
        if k_sampling_probs is None:
            return None

        probabilities = np.asarray(k_sampling_probs, dtype=np.float64)
        if probabilities.shape != (max_K,):
            raise ValueError(
                "k_sampling_probs must contain exactly max_K probabilities "
                f"(expected {max_K}, got {probabilities.size})"
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("k_sampling_probs must contain only finite values")
        if np.any(probabilities < 0):
            raise ValueError("k_sampling_probs cannot contain negative values")
        if not np.isclose(probabilities.sum(), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError(
                "k_sampling_probs must sum to 1 "
                f"(got {probabilities.sum():.8f})"
            )
        return probabilities

    def __len__(self):
        return self.length

    def _sample_k(self, idx):
        if self.constant_K is not False:
            return int(self.constant_K)
        if self.k_sampling_probs is not None:
            return int(
                self.rng.choice(
                    np.arange(1, self.max_K + 1),
                    p=self.k_sampling_probs,
                )
            )
        return int(self.rng.integers(1, self.max_K + 1))

    def _sample_indices_for_k(self, k, idx):
        return self.rng.choice(self.nb_selected_waveforms, size=k, replace=False)

    def _sample_mixture(self, idx):
        k = self._sample_k(idx)
        sampled_indices = self._sample_indices_for_k(k, idx)
        return sampled_indices, k

    def _build_fixed_mixtures(self):
        fixed_mixtures = []
        for idx in range(self.length):
            sampled_indices, target = self._sample_mixture(idx)
            fixed_mixtures.append((sampled_indices, target))
        return fixed_mixtures
    
    def _transform_sources(self, sources, sampled_indices, target, idx):
        return sources

    def _get_noise_rng(self, idx):
        if self.deterministic:
            seed = 0 if self.seed is None else self.seed
            seed_seq = np.random.SeedSequence([seed, int(idx), 12345])
            return np.random.default_rng(seed_seq)

        return self.rng

    def _add_noise(self, summed_waveforms, idx):
        noise_rng = self._get_noise_rng(idx)
        noise = noise_rng.normal(0, 1, size=summed_waveforms.shape)
        return summed_waveforms + noise, noise

    def _build_padded_params(self, sampled_indices, k):
        params = {}
        for attr in self.attr_names:
            values = np.asarray(getattr(self, attr)[sampled_indices])
            padded_shape = (self.num_source_slots,) + values.shape[1:]
            padded = np.zeros(padded_shape, dtype=values.dtype)
            padded[:k] = values
            params[attr] = padded

        params["source_mask"] = np.arange(self.num_source_slots) < k
        params.update(self._build_frequency_support_params(sampled_indices, k))
        return params

    def _build_frequency_support_params(self, sampled_indices, k):
        source_waveforms = np.asarray(self.waveforms[sampled_indices])
        source_waveforms = source_waveforms.astype(np.float32, copy=False)

        if source_waveforms.ndim == 2:
            freq_energy = source_waveforms ** 2
        else:
            channel_axes = tuple(range(1, source_waveforms.ndim - 1))
            freq_energy = np.sum(source_waveforms ** 2, axis=channel_axes)

        starts = np.full(self.num_source_slots, -1, dtype=np.int16)
        stops = np.full(self.num_source_slots, -1, dtype=np.int16)
        peaks = np.full(self.num_source_slots, -1, dtype=np.int16)
        widths = np.zeros(self.num_source_slots, dtype=np.int16)

        for source_idx in range(k):
            energy = freq_energy[source_idx]
            max_energy = float(np.max(energy))
            if max_energy <= 0.0:
                continue

            threshold = max_energy * FREQUENCY_SUPPORT_REL_THRESHOLD
            active_bins = np.flatnonzero(energy >= threshold)
            if len(active_bins) == 0:
                peak = int(np.argmax(energy))
                active_bins = np.array([peak], dtype=np.int64)

            starts[source_idx] = int(active_bins[0])
            stops[source_idx] = int(active_bins[-1]) + 1
            peaks[source_idx] = int(np.argmax(energy))
            widths[source_idx] = int(stops[source_idx] - starts[source_idx])

        return {
            "freq_bin_start": starts,
            "freq_bin_stop": stops,
            "freq_bin_peak": peaks,
            "freq_support_width": widths,
        }

    def __getitem__(self, idx):
        if self.deterministic:
            sampled_indices, K = self.fixed_mixtures[idx]
        else:
            sampled_indices, K = self._sample_mixture(idx)

        waveform_indices = self.indices[sampled_indices]

        sources = np.asarray(self.waveforms[waveform_indices], dtype=np.float32)
        sources = self._transform_sources(
            sources,
            waveform_indices,
            K,
            idx,
        )

        clean_mixture = sources.sum(axis=0)

        if self.noise:
            mixture, residual = self._add_noise(clean_mixture, idx)
        else:
            mixture = clean_mixture
            residual = np.zeros_like(mixture)
        
        mixture = mixture.astype(np.float32, copy=False)
        residual = residual.astype(np.float32, copy=False)

        X_1 = np.zeros(
            (self.num_source_slots + 1,) + sources.shape[1:],
            dtype=np.float32,
        )
        X_1[:K] = sources
        X_1[-1] = residual

        slot_mask = np.zeros(self.num_source_slots + 1, dtype=np.bool_)
        slot_mask[:K] = True
        slot_mask[-1] = True

        if self.returns_snr:
            snrs = np.zeros(self.num_source_slots, dtype=np.float32)
            snrs[:K] = np.asarray(self.snr[waveform_indices], dtype=np.float32)
            return mixture, X_1, K, slot_mask, snrs

        if self.return_params:
            params = self._build_padded_params(waveform_indices, K)
            return mixture, X_1, K, slot_mask, params

        return mixture, X_1, K, slot_mask


class TrainDataset(GalacticBinariesDataset):
    def __init__(
        self,
        waveforms,
        params_dict,
        max_K: int = 10,
        constant_K: bool | int = False,
        noise: bool = True,
        # random_global_scale: bool = False,
        # scale_min: float = 0.5,
        # scale_max: float = 2.0,
        max_samples: int | None = None,
        indices: np.ndarray | None = None,
        return_params: bool = False,
        returns_snr: bool = False,
        deterministic: bool = False,
        seed: int | None = None,
        k_sampling_probs=None,
    ):
        super().__init__(
            waveforms=waveforms,
            params_dict=params_dict,
            max_K=max_K,
            constant_K=constant_K,
            noise=noise,
            max_samples=max_samples,
            indices=indices,
            return_params=return_params,
            returns_snr=returns_snr,
            deterministic=deterministic,
            seed=seed,
            k_sampling_probs=k_sampling_probs,
        )

        # self.random_global_scale = random_global_scale
        # self.scale_min = scale_min
        # self.scale_max = scale_max

    # def _transform_sources(self, sources, sampled_indices, target, idx):
    #     if self.random_global_scale:
    #         log_scale = self.rng.uniform(
    #             np.log(self.scale_min),
    #             np.log(self.scale_max),
    #         )
    #         scale = np.exp(log_scale)
    #         sources = sources * scale

    #     return sources


class ValidationDataset(GalacticBinariesDataset):
    def __init__(
        self,
        waveforms,
        params_dict,
        max_K: int = 10,
        constant_K: bool | int = False,
        noise: bool = True,
        max_samples: int | None = None,
        indices: np.ndarray | None = None,
        return_params: bool = False,
        returns_snr: bool = False,
        deterministic: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            waveforms=waveforms,
            params_dict=params_dict,
            max_K=max_K,
            constant_K=constant_K,
            noise=noise,
            max_samples=max_samples,
            indices=indices,
            return_params=return_params,
            returns_snr=returns_snr,
            deterministic=deterministic,
            seed=seed,
        )


# class EnergyMatchingValidationDataset(GalacticBinariesDataset):
#     def __init__(
#         self,
#         waveforms,
#         params_dict,
#         max_K: int = 10,
#         constant_K: bool | int = False,
#         noise: bool = True,
#         max_samples: int | None = None,
#         indices: np.ndarray | None = None,
#         return_params: bool = False,
#         deterministic: bool = True,
#         seed: int | None = None,
#         target_energy: float = 22000.0,
#     ):
#         self.target_energy = target_energy
#         super().__init__(
#             waveforms=waveforms,
#             params_dict=params_dict,
#             max_K=max_K,
#             constant_K=constant_K,
#             noise=noise,
#             max_samples=max_samples,
#             indices=indices,
#             return_params=return_params,
#             deterministic=deterministic,
#             seed=seed,
#         )

#     def _transform_sources(self, sources, sampled_indices, target, idx):
#         clean_mixture = sources.sum(axis=0)
#         energy = np.sum(clean_mixture ** 2)

#         if energy > 0:
#             scaling_factor = np.sqrt(self.target_energy / (energy + 1e-12))
#             sources = sources * scaling_factor

#         return sources


def split_dataset_indices(total_waveforms, train_size=0.8, split_seed=42):
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be between 0 and 1")

    split_index = int(round(train_size * total_waveforms))
    indices = np.arange(total_waveforms, dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(indices)
    return indices[:split_index], indices[split_index:]


def snr_based_split_dataset_indices(
    snr_values,
    train_size=0.8,
    split_seed=42,
    bin_width=1.0,
):
    snr_values = np.asarray(snr_values, dtype=np.float32)

    rng = np.random.default_rng(split_seed)
    bin_start = np.floor(float(np.min(snr_values)))
    bin_ids = np.floor((snr_values - bin_start) / bin_width).astype(np.int64)

    train_chunks = []
    val_chunks = []
    for bin_id in np.unique(bin_ids):
        bin_indices = np.flatnonzero(bin_ids == bin_id).astype(np.int64)
        rng.shuffle(bin_indices)

        split_index = int(round(train_size * len(bin_indices)))
        train_chunks.append(bin_indices[:split_index])
        val_chunks.append(bin_indices[split_index:])

    train_indices = np.concatenate(train_chunks) if train_chunks else np.empty(0, dtype=np.int64)
    val_indices = np.concatenate(val_chunks) if val_chunks else np.empty(0, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    
    logger.info(f"SNR-based split built with {len(train_chunks)} bins of width {bin_width:.3g}: {len(train_indices)} train / {len(val_indices)} val")
    return train_indices, val_indices


def create_train_val_datasets(
    dataset_path : str,
    train_size : float = 0.8,
    max_K : int = 10,
    constant_K : bool | int = False,
    max_samples_train : int | None = None,
    max_samples_val : int | None = None,
    noise_train : bool = True,
    noise_val : bool = True,
    # random_global_scale_train : bool = False,
    deterministic_train : bool = False,
    deterministic_val : bool = True,
    # target_energy_val : float = 22000.0,
    return_params : bool = False,
    returns_snr : bool = False,
    split_seed : int = 42,
    split_strategy : str = "snr",
    snr_bin_width : float = 1.0,
    seed_train : int = 42,
    seed_val : int = 0,
    k_sampling_probs_train=None,
):
    with h5py.File(dataset_path, "r") as f:
        logger.info(f"Loading waveforms from {dataset_path}...")
        total_waveforms = f["waveforms"].shape[0]
        logger.info(f"Total number of waveforms in the dataset: {total_waveforms}")
        logger.info("Starting to load waveforms in memory...")
        waveforms = f["waveforms"][:]
        logger.info(f"Waveforms loaded with shape {waveforms.shape}")

        params_dict = {}
        logger.info(f"Loading parameters in memory...")
        for key, value in f["params"].items():
            params_dict[key] = value[:]
        logger.info("Parameters loaded successfully")

    if split_strategy == "random":
        train_indices, val_indices = split_dataset_indices(
            total_waveforms=total_waveforms,
            train_size=train_size,
            split_seed=split_seed,
        )
    elif split_strategy == "snr":
        train_indices, val_indices = snr_based_split_dataset_indices(
            snr_values=params_dict["snr"],
            train_size=train_size,
            split_seed=split_seed,
            bin_width=snr_bin_width,
        )
    else:
        raise ValueError("split_strategy must be either 'random' or 'snr'")
    

    train_dataset = TrainDataset(
        waveforms=waveforms,
        params_dict=params_dict,
        max_K=max_K,
        constant_K=constant_K,
        max_samples=max_samples_train,
        indices=train_indices,
        noise=noise_train,
        # random_global_scale=random_global_scale_train,
        deterministic=deterministic_train,
        return_params=return_params,
        returns_snr=returns_snr,
        seed=seed_train,
        k_sampling_probs=k_sampling_probs_train,
    ) if len(train_indices) > 0 else None

    val_dataset = ValidationDataset(
        waveforms=waveforms,
        params_dict=params_dict,
        max_K=max_K,
        constant_K=constant_K,
        max_samples=max_samples_val,
        indices=val_indices,
        noise=noise_val,
        deterministic=deterministic_val,
        return_params=return_params,
        returns_snr=returns_snr,
        seed=seed_val,
    )

    # val_energy_matched_dataset = EnergyMatchingValidationDataset(
    #     waveforms=waveforms,
    #     params_dict=params_dict,
    #     max_K=max_K,
    #     constant_K=constant_K,
    #     max_samples=max_samples_val,
    #     indices=val_indices,
    #     noise=noise_val,
    #     deterministic=deterministic_val,
    #     target_energy=target_energy_val,
    #     return_params=return_params,
    #     seed=seed_val,
    # )

    return train_dataset, val_dataset

if __name__ == "__main__":
    from config import small_dataset_path
    from torch.utils.data import DataLoader
    train_dataset, val_dataset = create_train_val_datasets(
        dataset_path=small_dataset_path,
        train_size=0.8,
        max_K=10,
        constant_K=2,
        max_samples_train=None,
        max_samples_val=10_000,
        noise_train=True,
        noise_val=True,
        # random_global_scale_train=True,
        deterministic_train=False,
        deterministic_val=True,
        # target_energy_val=22000.0,
        return_params=False,
        returns_snr=True,
        split_seed=42,
        split_strategy="snr",
        snr_bin_width=1.0,
        seed_train=42,
        seed_val=0
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for batch_idx, (mixture, X_1, K, slot_mask, snrs) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  mixture shape: {mixture.shape}")
        print(f"  X_1 shape: {X_1.shape}")
        print(f"  K shape: {K.shape}")
        print(f"  K values: {K}")
        print(f"  slot mask: {slot_mask}")
        print(f"  snrs shape: {snrs.shape}")
        print(f"  SNR values: {snrs}")

        reconstructed_mixture = X_1.sum(dim=1)
        error = (reconstructed_mixture - mixture).abs().max()

        print(f"max reconstruction error: {error}")
        for i in range(mixture.shape[0]):
            for k in range(K[i]):
                source_energy = (X_1[i, k] ** 2).sum().item()
                estimated_snr_based_on_energy = np.sqrt(source_energy)
                print(f"Batch {batch_idx}, Sample {i}, Source {k}: Energy = {source_energy:.6f}, Estimated SNR = {estimated_snr_based_on_energy:.6f}, Actual SNR = {snrs[i, k]:.6f}")
        if batch_idx >= 2:
            break
