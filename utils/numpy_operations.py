import numpy as np


def load_npz_file(npz_file, keys=None):
    """
    Load npz file with provided keys and return the data
    :param npz_file:
    :param keys: could be List or String, if None, return the first key.
    :return: if keys is List, return a List of data, if keys is String, return a np.array
    """
    if keys is not None:
        loaded_npz = np.load(npz_file, allow_pickle=True)
        if isinstance(keys, str):
            return loaded_npz[keys].astype(np.float32)
        else:
            return [loaded_npz[key].astype(np.float32) for key in keys]
    else:
        loaded_npz = np.load(npz_file, allow_pickle=True)
        key = list(loaded_npz.keys())[0]  # only load the first key if keys is None
        return loaded_npz[key].astype(np.float32)
