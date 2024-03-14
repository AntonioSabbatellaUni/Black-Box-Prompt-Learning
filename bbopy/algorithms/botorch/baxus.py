import math
from dataclasses import dataclass
from typing import Union, Type, Optional, Dict, Any, Tuple

import gpytorch
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.exceptions import ModelFittingError
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from bbopy.algorithms import BoTorchAlgorithm
from bbopy.problems.base import Problem
from bbopy.sampling import Sampling, FloatRandomSampling


class BAxUS(BoTorchAlgorithm):
    name: str = "BAxUS"

    def __init__(
            self,
            n_init: int,
            evaluation_budget: int,
            acquisition: Union[Type[ExpectedImprovement], Type[MaxPosteriorSampling]],
            acquisition_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            sampling: Sampling = FloatRandomSampling(backend="torch", dtype=torch.double),
    ) -> None:
        super().__init__(n_init)
        self._target_bounds = None
        self.evaluation_budget = evaluation_budget
        self.S = None
        self._dim = None
        self.state = None
        self.n_candidates = None
        self._sampling = sampling
        self._surrogate = SingleTaskGP
        self._acquisition = acquisition
        self._acquisition_optimizer_kwargs = acquisition_optimizer_kwargs
        if acquisition_optimizer_kwargs is None:
            _, self._acquisition_optimizer_kwargs = self._default_acqf_optimizer()

        self._train_x_target = None

    def setup(
            self,
            problem: Problem,
    ) -> None:
        self._dim = problem.dim
        self.n_candidates = min(5000, max(2000, 200 * problem.dim))
        self.state = BaxusState(dim=problem.dim, eval_budget=self.evaluation_budget - self.n_init)
        self.S = embedding_matrix(input_dim=self.state.dim, target_dim=self.state.d_init)
        self.bounds = problem.bounds
        bounds = torch.tensor(problem.bounds)
        self._bounds = torch.stack([bounds[:, 0], bounds[:, 1]])
        self._target_bounds = torch.tensor([[-1, 1]] * self.state.d_init)

    def ask(self):
        if self._train_x is None:
            self._train_x_target = self._sampling(self._target_bounds, self.n_init)
            self.S = self.S.to(device=self._train_x_target.device, dtype=self._train_x_target.dtype)
            self._train_x = self._train_x_target @ self.S
            self._train_x = self._bounds[0] + (self._bounds[1] - self._bounds[0]) * (self._train_x + 1) / 2
            return self._train_x
        candidates = self._generate_candidate()
        new_x_target = candidates.detach()
        new_x = new_x_target @ self.S
        new_x = self._bounds[0] + (self._bounds[1] - self._bounds[0]) * (new_x + 1) / 2
        self._train_x_target = torch.cat([self._train_x_target, new_x_target])
        self._train_x = torch.cat([self._train_x, new_x])
        return new_x

    def tell(self, y):
        if self._train_y is None:
            self._train_y = y
        else:
            self._train_y = torch.cat([self._train_y, y])
            self._update_state(Y_next=y)
            if self.state.restart_triggered:
                self._trigger_restart()
        self._update_model()

    def _update_model(self) -> None:
        train_Y = (self.train_y - self.train_y.mean()) / self.train_y.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.state.target_dim,
                lengthscale_constraint=Interval(0.005, 10)
            ),
            outputscale_constraint=Interval(0.05, 10),
        )
        self.model = self._surrogate(
            self._train_x_target, train_Y, covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        with gpytorch.settings.max_cholesky_size(float("inf")):
            try:
                fit_gpytorch_mll(mll)
            except ModelFittingError:
                optimizer = torch.optim.Adam([{"params": self.model.parameters()}], lr=0.1)

                for _ in range(100):
                    optimizer.zero_grad()
                    output = self.model(self._train_x_target)
                    loss = -mll(output, train_Y.flatten())
                    loss.backward()
                    optimizer.step()

    def _generate_candidate(self):
        assert torch.all(torch.isfinite(self.train_y))

        # Scale the TR to be proportional to the lengthscales
        x_center = self._train_x_target[self.train_y.argmax(), :].clone()
        weights = self.model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.state.length, -1.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.state.length, -1.0, 1.0)
        if self._acquisition == MaxPosteriorSampling:
            return self._generate_candidate_ts(self._train_x_target, tr_lb, tr_ub, x_center)
        if self._acquisition == ExpectedImprovement:
            return self._generate_candidate_qei(tr_lb, tr_ub)
        raise NotImplementedError()

    def _generate_candidate_ts(self, train_X, tr_lb, tr_ub, x_center):
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
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(self.n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = self._acquisition(model=self.model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            candidate = thompson_sampling(X_cand, num_samples=1)
        return candidate

    def _generate_candidate_qei(self, tr_lb, tr_ub):
        ei = self._acquisition(self.model, self.train_y.max())
        candidate, _ = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=1,
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
        return self.state

    def _trigger_restart(self):
        self.state.restart_triggered = False
        self.S, self._train_x_target = increase_embedding_and_observations(
            self.S, self._train_x_target, self.state.new_bins_on_split
        )
        self.state.target_dim = len(self.S)
        self.state.length = self.state.length_init
        self.state.failure_counter = 0
        self.state.success_counter = 0


def increase_embedding_and_observations(
        S: torch.Tensor, X: torch.Tensor, n_new_bins: int
) -> tuple[Tensor, Tensor]:
    assert X.size(1) == S.size(0), "Observations don't lie in row space of S"
    device = X.device
    dtype = X.dtype

    S_update = S.clone()
    X_update = X.clone()

    for row_idx in range(len(S)):
        row = S[row_idx]
        idxs_non_zero = torch.nonzero(row)
        idxs_non_zero = idxs_non_zero[torch.randperm(len(idxs_non_zero))].squeeze()

        non_zero_elements = row[idxs_non_zero].squeeze()

        n_row_bins = min(
            n_new_bins, len(idxs_non_zero)
        )  # number of new bins is always less or equal than the contributing input dims in the row minus one

        new_bins = torch.tensor_split(idxs_non_zero, n_row_bins)[
                   1:
                   ]  # the dims in the first bin won't be moved
        elements_to_move = torch.tensor_split(non_zero_elements, n_row_bins)[1:]

        new_bins_padded = torch.nn.utils.rnn.pad_sequence(
            new_bins, batch_first=True
        )  # pad the tuples of bins with zeros to apply _scatter
        els_to_move_padded = torch.nn.utils.rnn.pad_sequence(
            elements_to_move, batch_first=True
        )

        S_stack = torch.zeros(
            (n_row_bins - 1, len(row) + 1), device=device, dtype=dtype
        )  # submatrix to stack on S_update

        S_stack = S_stack.scatter_(
            1, new_bins_padded + 1, els_to_move_padded
        )  # fill with old values (add 1 to indices for padding column)

        S_update[
            row_idx, torch.hstack(new_bins)
        ] = 0  # set values that were move to zero in current row

        X_update = torch.hstack(
            (X_update, X[:, row_idx].reshape(-1, 1).repeat(1, len(new_bins)))
        )  # repeat observations for row at the end of X (column-wise)
        S_update = torch.vstack(
            (S_update, S_stack[:, 1:])
        )  # stack onto S_update except for padding column

    return S_update, X_update


def embedding_matrix(input_dim: int, target_dim: int, **tkwargs) -> torch.Tensor:
    device = tkwargs.get("device", "cpu")
    dtype = tkwargs.get("dtype", torch.float)

    if (
            target_dim >= input_dim
    ):  # return identity matrix if target size greater than input size
        return torch.eye(input_dim, device=device, dtype=dtype)

    input_dims_perm = (
            torch.randperm(input_dim, device=device) + 1
    )  # add 1 to indices for padding column in matrix

    bins = torch.tensor_split(
        input_dims_perm, target_dim
    )  # split dims into almost equally-sized bins
    bins = torch.nn.utils.rnn.pad_sequence(
        bins, batch_first=True
    )  # zero pad bins, the index 0 will be cut off later

    mtrx = torch.zeros(
        (target_dim, input_dim + 1), dtype=dtype, device=device
    )  # add one extra column for padding
    mtrx = mtrx.scatter_(
        1,
        bins,
        2 * torch.randint(2, (target_dim, input_dim), dtype=dtype, device=device) - 1,
    )  # fill mask with random +/- 1 at indices

    return mtrx[:, 1:]  # cut off index zero as this corresponds to zero padding


@dataclass
class BaxusState:
    dim: int
    eval_budget: int
    new_bins_on_split: int = 3
    d_init: int = float("nan")  # Note: post-initialized
    target_dim: int = float("nan")  # Note: post-initialized
    n_splits: int = float("nan")  # Note: post-initialized
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        n_splits = round(math.log(self.dim, self.new_bins_on_split + 1))
        self.d_init = 1 + np.argmin(
            np.abs(
                (1 + np.arange(self.new_bins_on_split))
                * (1 + self.new_bins_on_split) ** n_splits
                - self.dim
            )
        )
        self.target_dim = self.d_init
        self.n_splits = n_splits

    @property
    def split_budget(self) -> int:
        return round(
            -1
            * (self.new_bins_on_split * self.eval_budget * self.target_dim)
            / (self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1)))
        )

    @property
    def failure_tolerance(self) -> int:
        if self.target_dim == self.dim:
            return self.target_dim
        k = math.floor(math.log(self.length_min / self.length_init, 0.5))
        split_budget = self.split_budget
        return min(self.target_dim, max(1, math.floor(split_budget / k)))
