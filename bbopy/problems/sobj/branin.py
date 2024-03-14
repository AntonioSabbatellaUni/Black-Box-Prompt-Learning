import math
from typing import Optional, List, Tuple

from bbopy.problems.base import SingleObjectiveProblem


class Branin(SingleObjectiveProblem):
    r"""The Branin synthetic problem.

    2-dimensional function (usually evaluated on `[-5.0, 10.0] x [0.0, 15.0]`):

        f(x) = (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`.
    f has 3 minimizers for its global minimum at `x^*_1 = (-pi, 12.275)`,
    `x^*_2 = (pi, 2.275)`, `x^*_3 = (9.42478, 2.475)` with `f(x^*_i) = 0.397887`.
    """
    dim: int = 2
    n_constr: int = 0
    _optimal_value: float = 0.397887
    _optimizers: List[Tuple[float, float]] = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]

    def __init__(
            self,
            backend: str,
            bounds: Optional[List[Tuple[float, float]]] = None,
            maximize: Optional[bool] = False
    ) -> None:
        super().__init__(backend=backend, bounds=bounds, maximize=maximize)
        if bounds is None:
            self.bounds = [(-5.0, 10.0), (0.0, 15.0)]

    def _evaluate(self, x):
        back, _ = self._get_backend()
        part1 = (x[..., 1]
                 - 5.1 / (4 * math.pi ** 2) * x[..., 0] ** 2
                 + 5 / math.pi * x[..., 0]
                 - 6)
        part2 = 10 * (1 - 1 / (8 * math.pi)) * back.cos(x[..., 0])
        return (part1 ** 2 + part2 + 10).reshape(-1, 1)
