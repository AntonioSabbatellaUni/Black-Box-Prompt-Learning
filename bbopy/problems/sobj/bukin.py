from typing import Optional, List, Tuple

from bbopy.problems.base import SingleObjectiveProblem


class Bukin(SingleObjectiveProblem):
    r"""The Ackley synthetic problem.

    2-dimensional function (usually evaluated on `[-15.0, -5.0] x [-3.0, 3.0]`):

        f(x) = 100 sqrt(abs(x_2 - 0.01 x_1^2)) + 0.01 abs(x_1 + 10)

    f has one minimizer for its global minimum at `x^* = (-10.0, 1.0)` with
    `f(x^*) = 0`.
    """
    dim = 2
    n_constr = 0
    _optimal_value = 0.0
    _optimizers = [(-10.0, 1.0)]

    def __init__(
            self,
            backend: str,
            bounds: Optional[List[Tuple[float, float]]] = None,
            maximize: Optional[bool] = False
    ) -> None:
        super().__init__(backend=backend, bounds=bounds, maximize=maximize)
        if bounds is None:
            self.bounds = [(-15.0, -5.0), (-3.0, 3.0)]

    def _evaluate(self, x):
        back, _ = self._get_backend()
        part1 = 100.0 * back.sqrt(back.abs(x[..., 1] - 0.01 * x[..., 0] ** 2))
        part2 = 0.01 * back.abs(x[..., 0] + 10.0)
        return (part1 + part2).reshape(-1, 1)
