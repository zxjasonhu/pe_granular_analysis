import numpy as np
import torch


def mixup(_input, truth, aux_truth=None):
    indices = torch.randperm(_input.size(0))
    shuffled_input = _input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.beta(0.2, 1.0)
    mixed_input = shuffled_input * lam + _input * (1 - lam)

    if aux_truth is not None:
        shuffled_aux_labels = aux_truth[indices]
        return mixed_input, shuffled_labels, shuffled_aux_labels, lam

    return mixed_input, shuffled_labels, lam
