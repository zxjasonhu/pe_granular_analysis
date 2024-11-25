import torch
from monai.data import DistributedWeightedRandomSampler

from typing import Optional, Sequence, Iterator

from torch.utils.data import WeightedRandomSampler, DistributedSampler, Dataset
import numpy as np


def get_distributed_weighted_sampler(
    dataset,
    weights,
    num_samples_per_rank=None,
    even_divisible=True,
    num_replicas=None,
    rank=None,
    seed=42,
    drop_last=False,
):
    """
    dataset: Dataset used for sampling.
    weights: a sequence of weights, not necessary summing up to one, length should exactly
        match the full dataset.
    num_samples_per_rank: number of samples to draw for every rank, sample from
        the distributed subset of dataset.
        if None, default to the length of dataset split by DistributedSampler.
    generator: PyTorch Generator used in sampling.
    even_divisible: if False, different ranks can have different data length.
        for example, input data: [1, 2, 3, 4, 5], rank 0: [1, 3, 5], rank 1: [2, 4].'
    num_replicas: number of processes participating in distributed training.
        by default, `world_size` is retrieved from the current distributed group.
    rank: rank of the current process within `num_replicas`. by default,
        `rank` is retrieved from the current distributed group.
    kwargs: additional arguments for `DistributedSampler` super class, can be `seed` and `drop_last`.
    """
    generator = torch.Generator(device="cpu")

    return DistributedWeightedRandomSampler(
        dataset=dataset,
        weights=weights,
        num_samples_per_rank=num_samples_per_rank,
        generator=generator,
        even_divisible=even_divisible,
        num_replicas=num_replicas,
        rank=rank,
        seed=seed,
        drop_last=drop_last,
    )


class CustomDistributedWeightedRandomSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        replacement=True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.weights = np.array(weights)
        self.replacement = replacement

    def __iter__(self):
        indices = np.array(list(super().__iter__()))
        weights = self.weights[indices]
        weighted_indices = list(
            WeightedRandomSampler(weights, len(weights), self.replacement)
        )
        indices = [indices[wi] for wi in weighted_indices]
        return iter(indices)


def get_custom_distributed_weighted_sampler(
    dataset,
    weights,
    replacement=True,
    num_replicas=None,
    rank=None,
    shuffle=True,
    seed=42,
    drop_last=False,
):
    return CustomDistributedWeightedRandomSampler(
        dataset=dataset,
        weights=weights,
        replacement=replacement,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
