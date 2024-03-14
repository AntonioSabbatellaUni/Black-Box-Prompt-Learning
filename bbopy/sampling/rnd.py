from typing import List, Tuple, Union

import numpy as np
from typing_extensions import override

from bbopy.sampling.base import Sampling


def boundarize(
        sample: np.ndarray,
        bounds: np.ndarray
) -> List[Union[float]]:
    r"""Normalize the given numpy array into the given bounds.

    Args:
        sample: A `n_samples x dim` numpy array.
        bounds: A `dim x 2` numpy array with lower and upper bounds for each dimension.

    Returns:
        A `n_samples x dim` list bounded by `bounds`.
    """
    bounded_sample = (bounds[:, 0] - bounds[:, 1]) * sample + bounds[:, 1]
    return bounded_sample.tolist()


class FloatRandomSampling(Sampling):
    r""" Float number sampling strategy.

    It uses the numpy random module to generate float numbers in [0, 1]
    and then normalizes them into the given bounds.

    Attributes:
        backend: A string representing the array backend. Can be one of: "numpy", "pymoo", "torch".
        seed: An integer number representing the seed for the random number. Used for replicability. Can be None.
        **tkwargs: A dictionary with additional parameters when using the "torch" backend.
            It can specify "device" and "dtype" of the torch tensor. Ignored with other backends.
    """

    @override
    def _generate(
            self,
            bounds: List[Tuple[float, float]],
            n_samples: int,
    ) -> List[List[Union[float, int, bool]]]:
        dim = len(bounds)
        bounds = np.array(bounds)
        sample = [boundarize(np.random.rand(dim), bounds) for _ in range(n_samples)]
        return sample
