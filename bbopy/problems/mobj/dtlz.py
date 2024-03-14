import math
from abc import ABC
from typing import Optional, Union

import numpy as np
import torch

from bbopy.problems.base import MultiObjectiveProblem


class DTLZ(MultiObjectiveProblem, ABC):
    n_constr: int = 0
    _ref_val: float

    def __init__(
            self,
            backend: str,
            dim: int,
            n_obj: int,
            maximize: Optional[bool] = False,
    ) -> None:
        if dim <= n_obj:
            raise ValueError(f"dim must be > n_obj, but got {dim} and {n_obj}.")
        super().__init__(backend=backend, dim=dim, n_obj=n_obj, maximize=maximize)
        self.bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._ref_point = [self._ref_val for _ in range(self.n_obj)]
        self.k = self.dim - self.n_obj + 1


class DTLZ1(DTLZ):
    _ref_val: float = 400.0

    def _evaluate(
            self,
            x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        back, kwargs = self._get_backend()
        X_m = x[..., -self.k:]
        X_m_minus_half = X_m - 0.5
        sum_term = (
                X_m_minus_half**2 - back.cos(20 * math.pi * X_m_minus_half)
        ).sum(**kwargs)
        g_X_m = 100 * (self.k + sum_term)
        g_X_m_term = 0.5 * (1 + g_X_m)
        fs = []
        for i in range(self.n_obj):
            idx = self.n_obj - 1 - i
            f_i = g_X_m_term * x[..., :idx].prod(**kwargs)
            if i > 0:
                f_i *= 1 - x[..., idx]
            fs.append(f_i)
        return back.stack(fs, **kwargs)
