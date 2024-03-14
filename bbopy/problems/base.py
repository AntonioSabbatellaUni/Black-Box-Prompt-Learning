from abc import ABC, abstractmethod
from types import ModuleType
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
import pymoo.gradient.toolbox as anp
import torch


class Problem(ABC):
    r"""Abstract base class for a problem.

    Attributes:
        dim (int): The number of variables.
        n_obj (int): The number of objectives.
        n_constr (int): The number of constraints.
        bounds (List[Tuple[float, float]]): The lower and upper bounds of each variable.
        maximize (bool): A boolean indicator. False if it is a minimization problem,
            True if it is a maximization problem.
    """

    def __init__(
            self,
            backend: str,
            dim: Optional[int] = None,
            n_obj: Optional[int] = None,
            n_constr: Optional[int] = None,
            bounds: Optional[List[Tuple[float, float]]] = None,
            maximize: Optional[bool] = False,
    ) -> None:
        r"""Initializes problem settings.

        Args:
            dim: The number of variables.
            backend: The backend to use for array operations (numpy, pymoo or torch).
            n_obj: The number of objectives.
            n_constr: The number of constraints.
            bounds: The lower and upper bounds of each variable.
            maximize: A boolean indicator. False if it is a minimization problem,
                True if it is a maximization problem.
        """
        self.maximize = maximize
        if dim is not None:
            self.dim = dim
        if n_obj is not None:
            self.n_obj = n_obj
        if n_constr is not None:
            self.n_constr = n_constr
        if bounds is not None:
            self.bounds = bounds

        self._init_backend(backend)

    def _init_backend(self, backend: str) -> None:
        r"""Initializes the backend and its kwargs for array's operations.

        Numpy and Torch interfaces have some difference. This function initializes the kwargs
        (e.g., "axis" in numpy is equivalent to "dim" in torch).

        Args:
            backend: The backend to use for array operations (numpy, pymoo or torch).
        """
        if (backend == "pymoo") or (backend == "numpy"):
            self._backend = anp
            self._backend_kwargs = {"axis": -1}
        elif backend == "torch":
            self._backend = torch
            self._backend_kwargs = {"dim": -1}
        else:
            raise NotImplementedError(f"The {backend} backend is not implemented. "
                                      f"Try one of {['pymoo', 'numpy', 'torch']}")

    def _get_backend(self) -> Tuple[ModuleType, Dict]:
        r"""Utility function to get the current backend and its kwargs.

        Returns:
            A tuple containing the current backend and its kwargs.
        """
        return self._backend, self._backend_kwargs

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        r"""Evaluate the problem on the given points.

        Args:
            x: The input points to evaluate. It can be a `n_point x dim` numpy ndarray or
                torch tensor based on the backend.
        Returns:
            The objective values for the given points. It can be a `n_point x n_obj` numpy ndarray or
            torch tensor based on the backend.
        """
        val = self._evaluate(x)
        return -val if self.maximize else val

    @abstractmethod
    def _evaluate(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        r"""Utility function to evaluate the problem on the given points.

        This always assume a minimization problem.

        Args:
            x: The input points to evaluate. It can be a `n_point x dim` numpy ndarray or
                torch tensor based on the backend.
        Returns:
            The objective values for the given points. It can be a `n_point x n_obj` numpy ndarray or
            torch tensor based on the backend.
        """
        pass


class SingleObjectiveProblem(Problem, ABC):
    r"""Abstract base class for a single-objective problem.

    Attributes:
        dim (int): The number of variables.
        n_obj (int): The number of objectives.
        n_constr (int): The number of constraints.
        bounds (List[Tuple[float, float]]): The lower and upper bounds of each variable.
        maximize (bool): A boolean indicator. False if it is a minimization problem,
            True if it is a maximization problem.
    """
    n_obj: int = 1
    _optimal_value: float = None
    _optimizers: Optional[List[Tuple[float, ...]]] = None

    def __init__(
            self,
            backend: str,
            dim: Optional[int] = None,
            n_constr: Optional[int] = None,
            bounds: Optional[List[Tuple[float, float]]] = None,
            maximize: Optional[bool] = False,
    ) -> None:
        r"""Initializes problem settings.

        Args:
            dim: The number of variables.
            backend: The backend to use for array operations (numpy, pymoo or torch).
            n_constr: The number of constraints.
            bounds: The lower and upper bounds of each variable.
            maximize: A boolean indicator. False if it is a minimization problem,
                True if it is a maximization problem.
        """
        super().__init__(backend=backend, dim=dim, n_constr=n_constr, bounds=bounds, maximize=maximize)

    @property
    def optimal_value(self):
        r"""The optimal value of the problem."""
        if self._optimal_value is None:
            raise NotImplementedError()
        return -self._optimal_value if self.maximize else self._optimal_value


class MultiObjectiveProblem(Problem, ABC):
    r"""Abstract base class for a multi-objective problem.

    Attributes:
        dim (int): The number of variables.
        n_obj (int): The number of objectives.
        n_constr (int): The number of constraints.
        bounds (List[Tuple[float, float]]): The lower and upper bounds of each variable.
        maximize (bool): A boolean indicator. False if it is a minimization problem,
            True if it is a maximization problem.
    """
    _ref_point: List[float]
    _max_hv: Optional[float] = None

    def __init__(
            self,
            backend: str,
            dim: Optional[int] = None,
            n_obj: Optional[int] = None,
            n_constr: Optional[int] = None,
            bounds: Optional[List[Tuple[float, float]]] = None,
            maximize: Optional[bool] = False,
    ) -> None:
        r"""Initializes problem settings.

        Args:
            dim: The number of variables.
            backend: The backend to use for array operations (numpy, pymoo or torch).
            n_obj: The number of objectives.
            n_constr: The number of constraints.
            bounds: The lower and upper bounds of each variable.
            maximize: A boolean indicator. False if it is a minimization problem,
                True if it is a maximization problem.
        """
        super().__init__(backend=backend, dim=dim, n_obj=n_obj, n_constr=n_constr, bounds=bounds, maximize=maximize)

    @property
    def max_hv(self):
        r"""The hypervolume of the optimum of the problem."""
        if self._max_hv is None:
            raise NotImplementedError()
        return self._max_hv

    def pareto_front(self, n: int) -> Union[torch.Tensor, np.ndarray]:
        r"""Gets the pareto front of the problem.

        Args:
            n: The number of points that approximate the pareto front.

        Returns:
            A `n x dim` numpy ndarray or torch tensor representing the pareto front.
        """
        raise NotImplementedError
