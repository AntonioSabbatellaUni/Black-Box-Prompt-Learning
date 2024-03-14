from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from pymoo.core.population import Population
from pymoo.core.problem import Problem


class Sampling(ABC):
    r"""Abstract base class used to define a sampling strategy.

    Attributes:
        backend: A string representing the array backend. Can be one of: "numpy", "pymoo", "torch".
        seed: An integer number representing the seed for the random number. Used for replicability. Can be None.
        **tkwargs: A dictionary with additional parameters when using the "torch" backend.
            It can specify "device" and "dtype" of the torch tensor. Ignored with other backends.
    """

    def __init__(self, backend: str, seed: int = None, **tkwargs: dict):
        r""" Initializes the sampling strategy configuration.

        Args:
            backend: A string representing the array backend. Can be one of: "numpy", "pymoo", "torch".
            seed: An integer number representing the seed for the random number. Used for replicability. Can be None.
            **tkwargs:  A dictionary with additional parameters when using the "torch" backend.
                It can specify "device" and "dtype" of the torch tensor. Ignored with other backends.
        """
        self.backend = backend
        self.seed = seed
        self.tkwargs = tkwargs
        if seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def __call__(
            self,
            bounds: Union[List[Tuple[float, float]], Problem] = None,
            n_samples: int = None,
            **kwargs: dict
    ) -> Union[np.ndarray, Population, torch.Tensor]:
        """Generates a number of samples contained in the given bounds.

        The samples are generated using the strategy defined in `self._generate`.

        Args:
            bounds: A `dim x 2` list with the lower and upper bound for each
                dimension if backend is `torch` or `numpy`. A pymoo Problem instance if backend is `pymoo`.
                This allows to make the class compatible with pymoo algorithms. Can be None when used in `NoSampling`.
            n_samples: An integer representing the number of samples to be generated.
                Can be None when used in `NoSampling`.
            **kwargs: Used just to allow the compatibility with pymoo algorithms.

        Returns:
            A `n_samples x dim` torch tensor, numpy array, or pymoo Population object with the generated samples.
        """
        if isinstance(bounds, Problem):
            bounds = np.stack((bounds.xl, bounds.xu), axis=-1)
        val = self._generate(bounds, n_samples)
        return self._to_backend(val)

    def _to_backend(self, val: List[List[Union[float, int, bool]]]) -> Union[np.ndarray, Population, torch.Tensor]:
        r"""Converts the given list into the considered backend.

        Args:
            val: A list to be converted

        Returns:
            A torch tensor, numpy array, or pymoo Population object with the generated samples.
        """
        if self.backend == "numpy":
            return np.array(val)
        elif self.backend == "pymoo":
            return Population.new("X", np.array(val))
        elif self.backend == "torch":
            return torch.tensor(val, **self.tkwargs)

    @abstractmethod
    def _generate(
            self,
            bounds: List[Tuple[float, ...]],
            n_samples: int
    ) -> List[List[Union[float, int, bool]]]:
        r"""Generates a `n_samples x dim` list of random values contained in `bounds`.

        Args:
            bounds: A `dim x 2` list with the lower and upper bound for each dimension.
            n_samples: An integer representing the number of samples to be generated.

        Returns:
            A `n_samples x dim` list of random values contained in `bounds`.
        """
        pass


class NoSampling(Sampling):
    r"""A no-sampling strategy.

    Simply returns the given `val`, converted into the proper backend.
    Used to initialize algorithms with a given sample of data.

    Attributes:
        val: A list used as the sample. Calling `self._generate` will always return this `val`,
            converted into the proper backend.
        backend: A string representing the array backend. Can be one of: "numpy", "pymoo", "torch".
        **tkwargs: A dictionary with additional parameters when using the "torch" backend.
            It can specify "device" and "dtype" of the torch tensor. Ignored with other backends.
    """

    def __init__(self, val: List[List[Union[float, int, bool]]], backend: str, **tkwargs: dict):
        r""" Initializes the no-sampling strategy configuration and sets the sample `val` to be used.

        Args:
            val: A list used as the sample. Calling `self._generate` will always return this `val`,
                converted into the proper backend.
            backend: A string representing the array backend. Can be one of: "numpy", "pymoo", "torch".
            **tkwargs: A dictionary with additional parameters when using the "torch" backend.
                It can specify "device" and "dtype" of the torch tensor. Ignored with other backends.
        """
        super().__init__(backend, **tkwargs)
        self.val = val

    def _generate(
            self,
            bounds: Optional[List[Tuple[float, float]]] = None,
            n_samples: Optional[int] = None
    ) -> List[List[Union[float, int, bool]]]:
        r"""Converts the given `val` to the proper backend.

        All the parameters are ignored. They are just used to assure the compatibility with the algorithms.

        Args:
            bounds: Ignored.
            n_samples: Ignored.

        Returns:
            A torch tensor, numpy array, or pymoo Population object with the generated samples.
        """
        return self.val
