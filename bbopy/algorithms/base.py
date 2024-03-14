from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

from bbopy.problems.base import Problem


class Algorithm(ABC):
    r"""Abstract base class to implement an optimization algorithm.

    It implements an ask and tell interface to use the algorithm.
    Before the optimization process, it is necessary to call the `setup` function to
    adapt the algorithm to the problem.

    Attributes:
        name (str): Name of the algorithm. Just for logging purpose.
    """
    name: str

    @abstractmethod
    def setup(self, problem: Problem) -> None:
        r"""Initializes algorithm settings based on the problem.

        Args:
            problem (Problem): The problem to be optimized.
        """
        pass

    @abstractmethod
    def ask(self) -> Union[torch.Tensor, np.ndarray]:
        r"""Gets the new points to evaluate.

        Returns:
            An iterable with the new points to evaluate.
        """
        pass

    @abstractmethod
    def tell(self, y: Union[np.ndarray, torch.Tensor]) -> None:
        r"""Updates the algorithm with the new objective evaluations.

        Args:
            y: The new objective evaluations.
        """
        pass

    @property
    @abstractmethod
    def train_x(self):
        r"""A torch tensor or a numpy ndarray with the training points."""
        pass

    @property
    @abstractmethod
    def train_y(self):
        r"""A torch tensor or a numpy ndarray with the objective evaluations."""
        pass

    def best_seen(self, maximize: bool = False) -> float:
        r"""Gets the best seen over the evaluations (minimum if maximize is False, maximum otherwise).

        Args:
            maximize (bool): Whether to get the maximum or minimum among the objective evaluations.

        Returns:
            The optimum among the objective evaluations.
        """
        return self.train_y.max() if maximize else self.train_y.min()


class PymooAlgorithm(Algorithm, ABC):
    r"""Abstract base class for Pymoo algorithms.

    It is used as a unified interface for all Pymoo algorithms.

    Attributes:
        name (str): The name of the algorithm.
    """

    def __init__(self):
        r"""Initializes attributes."""
        self._settings = None
        self._algo = None
        self._curr_pop = None
        self._n_eval = 0

    def setup(
            self,
            problem: Problem,
    ) -> None:
        r"""Calls the setup method of the Pymoo algorithm.

        Creates a dummy Pymoo problem, based on the real problem to be optimized, to set up the Pymoo algorithm.

        Args:
            problem (Problem): The problem to be optimized.
        """
        self._setup(problem)
        self._settings = PymooProblem(n_var=problem.dim,
                                      n_obj=problem.n_obj,
                                      n_constr=problem.n_constr,
                                      xl=np.array(problem.bounds)[:, 0],
                                      xu=np.array(problem.bounds)[:, 1])
        self._algo.setup(self._settings, termination=NoTermination())

    def _setup(
            self,
            problem: Problem
    ) -> None:
        """Configures additional setup.

        Args:
            problem (Problem): The problem to be optimized.
        """
        pass

    def ask(self) -> np.array:
        """Calls the ask interface of the new algorithm and get the new population as a numpy ndarray.

        Returns:
            A numpy ndarray representing the new population.
        """
        self._curr_pop = self._algo.ask()
        return self._curr_pop.get("X")

    def tell(self, y: List[List[float]]) -> None:
        """Add the objective observations to the current population and update the algorithm.

        Args:
            y (list): The objective observations to be added to the current population.
        """
        self._n_eval += len(y)
        static = StaticProblem(self._settings, F=y)
        Evaluator().eval(static, self._curr_pop)
        self._algo.tell(infills=self._curr_pop)

    @property
    def train_x(self):
        r"""The current population points as a numpy ndarray."""
        return self._curr_pop.get("X")

    @property
    def train_y(self):
        r"""The current population objectives as a numpy ndarray."""
        return self._curr_pop.get("F")

    @property
    def n_eval(self):
        r"""The number of function evaluations so far."""
        return self._n_eval


class BoTorchAlgorithm(Algorithm, ABC):
    r"""Abstract base class for BoTorch algorithms.

    It is used as a unified interface for all BoTorch algorithms.

    Attributes:
        name (str): The name of the algorithm.
        n_init (int): The number of initial points to start up the algorithm.
        bounds (List[Tuple[float, float]]): The lower and upper bounds of each variable.
            They are automatically setted when calling the `setup` function.
    """

    def __init__(self, n_init: int):
        r"""Initializes attributes.

        THe basic attributes of all botorch algorithms are the train points `_train_x`
        and their objective values `_train_y`. Train points are contained in `bounds`.

        Args:
            n_init (int): The number of initial points to start up the algorithm.
        """
        self.n_init = n_init
        self.bounds = None

        self._train_x = None
        self._train_y = None
        self._bounds = None

    def setup(self, problem: Problem) -> None:
        r"""Initializes the bounds based on the problem and do additional setups
        based on the specific algorithm.

        The bounds are stored in two formats, as a `dim x 2` list (`bounds`) and
        as a `2 x dim` torch tensor (`_bounds`). The former is used for compatibility with the problem definition.
        The latter is the required format of the BoTorch `optimize_acqf` (the default acquisition function optimizer).

        Args:
            problem (Problem): The problem to be optimized.
        """
        self.bounds = problem.bounds
        bounds = torch.tensor(self.bounds)
        self._bounds = torch.stack([bounds[:, 0], bounds[:, 1]])
        self._setup()

    def _setup(
            self,
    ) -> None:
        """Configures additional setup."""
        pass

    @property
    def train_x(self):
        r"""The training points so far."""
        return self._train_x

    @property
    def train_y(self):
        r"""The objectives' values of the training points so far."""
        return self._train_y

    @property
    def n_eval(self):
        """The number of function evaluations so far."""
        return len(self.train_y)
