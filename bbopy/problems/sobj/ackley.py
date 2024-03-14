import math
from typing import Optional, List, Tuple

from bbopy.problems.base import SingleObjectiveProblem


class Ackley(SingleObjectiveProblem):
    r"""The Ackley synthetic problem.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) -
                exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)

    f has one minimizer for its global minimum at `x^* = (0, 0, ..., 0)` with
    `f(x^*) = 0`.
    """
    n_constr: int = 0
    _optimal_value: float = 0.0

    def __init__(
            self,
            backend: str,
            dim: int,
            bounds: Optional[List[Tuple[float, float]]] = None,
            maximize: Optional[bool] = False,
            a: Optional[float] = 20,
            b: Optional[float] = 0.2,
            c: Optional[float] = 2 * math.pi,
    ) -> None:
        super().__init__(backend=backend, dim=dim, bounds=bounds, maximize=maximize)
        if bounds is None:
            self.bounds = [(-32.768, 32.768)] * self.dim
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        self.a = a
        self.b = b
        self.c = c

    def _evaluate(self, x):
        back, kwargs = self._get_backend()
        a, b, c = self.a, self.b, self.c
        part1 = -a * back.exp(-b * back.sqrt((1 / self.dim) * back.sum(x * x, **kwargs)))
        part2 = -back.exp((1 / self.dim) * back.sum(back.cos(c * x), **kwargs))
        return (part1 + part2 + a + math.e).reshape(-1, 1)
