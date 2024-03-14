import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Type, Union

import gpytorch
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from torch.quasirandom import SobolEngine

from bbopy.algorithms import BoTorchAlgorithm
from bbopy.problems.base import Problem
from bbopy.sampling import Sampling, FloatRandomSampling


class TURBO(BoTorchAlgorithm):
    name: str = "TURBO"

    def __init__(
            self,
            n_init: int,
            acquisition: Union[Type[qExpectedImprovement], Type[MaxPosteriorSampling]],
            acquisition_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            batch_size: Optional[int] = 1,
            sampling: Sampling = FloatRandomSampling(backend="torch", dtype=torch.double)
    ) -> None:
        super().__init__(n_init)
        self._dim = None
        self.state = None
        self.n_candidates = None
        self._sampling = sampling
        self._surrogate = SingleTaskGP
        self._acquisition = acquisition
        self._acquisition_optimizer_kwargs = acquisition_optimizer_kwargs
        if acquisition_optimizer_kwargs is None:
            _, self._acquisition_optimizer_kwargs = self._default_acqf_optimizer()
        self._acquisition_optimizer_kwargs.pop("q", None)
        self.batch_size = batch_size

    def setup(
            self,
            problem: Problem,
    ) -> None:
        self._dim = problem.dim
        self.n_candidates = min(5000, max(2000, 200 * problem.dim))
        self.state = TurboState(dim=problem.dim, batch_size=self.batch_size)
        self.bounds = problem.bounds
        bounds = torch.tensor(problem.bounds)
        self._bounds = torch.stack([bounds[:, 0], bounds[:, 1]])

    def ask(self):
        if self._train_x is None:
            self._train_x = self._sampling(self.bounds, self.n_init)
            return self._train_x
        with gpytorch.settings.max_cholesky_size(float("inf")):
            candidates = self._generate_batch()
        new_x = candidates.detach()
        new_x = unnormalize(new_x, self._bounds)
        self._train_x = torch.cat([self._train_x, new_x])
        return new_x

    def tell(self, y):
        if self._train_y is None:
            self._train_y = y
        else:
            self._train_y = torch.cat([self._train_y, y])
            self._update_state(Y_next=y)
        self._update_model()

    def _update_model(self) -> None:
        train_X = normalize(self.train_x, self._bounds)
        train_Y = (self.train_y - self.train_y.mean()) / self.train_y.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=self._dim, lengthscale_constraint=Interval(0.005, 4.0)
            )
        )
        self.model = SingleTaskGP(
            train_X, train_Y, covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        with gpytorch.settings.max_cholesky_size(float("inf")):
            fit_gpytorch_mll(mll)

    def _generate_batch(self):
        assert torch.all(torch.isfinite(self.train_y))
        train_X = normalize(self.train_x, self._bounds)
        # Scale the TR to be proportional to the lengthscales
        x_center = train_X[self.train_y.argmax(), :].clone()
        weights = self.model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.state.length / 2.0, 0.0, 1.0)
        if self._acquisition == MaxPosteriorSampling:
            return self._generate_batch_ts(train_X, tr_lb, tr_ub, x_center)
        if self._acquisition == qExpectedImprovement:
            return self._generate_batch_qei(tr_lb, tr_ub)
        raise NotImplementedError()

    def _generate_batch_ts(self, train_X, tr_lb, tr_ub, x_center):
        dtype = train_X.dtype
        device = train_X.device

        dim = train_X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(self.n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(self.n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(self.n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            candidate = thompson_sampling(X_cand, num_samples=self.batch_size)
        return candidate

    def _generate_batch_qei(self, tr_lb, tr_ub):
        ei = qExpectedImprovement(self.model, self.train_y.max())
        candidate, _ = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=self.batch_size,
            **self._acquisition_optimizer_kwargs
        )
        return candidate

    def _update_state(self, Y_next):
        if max(Y_next) > self.state.best_value + 1e-3 * math.fabs(self.state.best_value):
            self.state.success_counter += 1
            self.state.failure_counter = 0
        else:
            self.state.success_counter = 0
            self.state.failure_counter += 1

        if self.state.success_counter == self.state.success_tolerance:  # Expand trust region
            self.state.length = min(2.0 * self.state.length, self.state.length_max)
            self.state.success_counter = 0
        elif self.state.failure_counter == self.state.failure_tolerance:  # Shrink trust region
            self.state.length /= 2.0
            self.state.failure_counter = 0

        self.state.best_value = max(self.state.best_value, max(Y_next).item())
        if self.state.length < self.state.length_min:
            self.state.restart_triggered = True


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )
