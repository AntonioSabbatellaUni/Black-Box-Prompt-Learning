from typing import Callable, Type, Dict, Any, Optional

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import AcquisitionFunction, UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from bbopy.algorithms.base import BoTorchAlgorithm
from bbopy.sampling import Sampling, FloatRandomSampling


class BayesianOptimization(BoTorchAlgorithm):
    """A generic Bayesian Optimization algorithm.

    It is characterized mainly by a surrogate model and an acquisition function.
    It supports all GPyTorch and BoTorch models and acquisition functions.

    Attributes:
        name (str): The name of the algorithm.
        n_init (int): The number of initial points to start up the algorithm.
        bounds (List[Tuple[float, float]]): The lower and upper bounds of each variable.
            They are automatically setted when calling the `setup` function.
        model (ExactGP): The surrogate model used to approximate the objective function.
            It can be any GPyTorch or BoTorch model. If not specified, the default is `SingleTaskGP`.
        acquisition (AcquisitionFunction): The acquisition function used to generate candidate points.
            It can be any BoTorch acquisition function. If not specified, the default is `UpperConfidenceBound`.
    """
    name: str = "Bayesian Optimization"

    def __init__(
            self,
            n_init: int,
            surrogate: Optional[Type[ExactGP]] = None,
            acquisition: Optional[Type[AcquisitionFunction]] = None,
            acquisition_optimizer: Optional[Callable] = None,
            surrogate_kwargs: Optional[Dict[str, Any]] = None,
            acquisition_kwargs: Optional[Dict[str, Any]] = None,
            acquisition_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            sampling: Sampling = FloatRandomSampling(backend="torch", dtype=torch.double)
    ):
        r"""Initializes surrogate and acquisition settings.

        Args:
            n_init (int): The number of initial points to start up the algorithm.
            surrogate (ExactGP): The surrogate model class to be used.
            acquisition (AcquisitionFunction): The acquisition function to be used.
            acquisition_optimizer (Callable): The acquisition function optimizer.
            surrogate_kwargs (dict): Additional arguments for the surrogate model.
            acquisition_kwargs (dict): Additional arguments for the acquisition function.
            acquisition_optimizer_kwargs (dict): Additional arguments for the acquisition function optimizer.
            sampling (Sampling): The strategy used to sample starting points (Make sure to use the torch backend).
        """
        super().__init__(n_init)

        self.model = None
        self.acquisition = None

        self._acqf = acquisition
        self._acqf_kwargs = acquisition_kwargs
        self._acqf_optimizer = acquisition_optimizer
        self._acqf_optimizer_kwargs = acquisition_optimizer_kwargs
        self._sampling = sampling
        self._surrogate = surrogate
        self._surrogate_kwargs = surrogate_kwargs

    def _setup(self) -> None:
        r"""Sets the default surrogate model and acquisition function, if they are not specified."""
        self._acqf = self._acqf or UpperConfidenceBound
        self._acqf_kwargs = self._acqf_kwargs or {"beta": 3}
        self._acqf_optimizer = self._acqf_optimizer or optimize_acqf
        self._acqf_optimizer_kwargs = self._acqf_optimizer_kwargs or {"q": 1, "num_restarts": 10, "raw_samples": 50}

        self._surrogate = self._surrogate or SingleTaskGP
        self._surrogate_kwargs = self._surrogate_kwargs or {}

        self._bounds = self._bounds.to(**self._sampling.tkwargs)

    def ask(self) -> torch.Tensor:
        r"""Optimize the acquisition function and get the candidate point(s) to evaluate.

        If this is the first call to the ask function, then initialize the starting points
        using the specified sampling strategy.

        Returns:
            A tensor containing the candidate point(s) to evaluate.
        """
        if self._train_x is None:
            self._train_x = self._sampling(self.bounds, self.n_init)
            return self._train_x
        candidates, acq_value = self._acqf_optimizer(acq_function=self.acquisition,
                                                     bounds=self._bounds,
                                                     **self._acqf_optimizer_kwargs)
        new_x = candidates.detach()
        self._train_x = torch.cat([self._train_x, new_x])
        return new_x

    def tell(self, y) -> None:
        r"""Update the surrogate model with the new objective evaluations
        and re-initialize the acquisition function.

        Here the surrogate model is fitted.

        Args:
            y (torch.Tensor): The objective evaluations to add to the surrogate model.
        """
        if self._train_y is None:
            self._train_y = y
        else:
            self._train_y = torch.cat([self._train_y, y])
        self.model = self._surrogate(self._train_x, self._train_y, **self._surrogate_kwargs)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.acquisition = self._acqf(self.model, **self._acqf_kwargs)
