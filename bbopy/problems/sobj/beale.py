from typing import Optional, List, Tuple

from bbopy.problems.base import SingleObjectiveProblem


class Beale(SingleObjectiveProblem):
    r"""The Beale synthetic problem.

    2-dimensional function (usually evaluated on `[-4.5, 4.5]^2`):

        f(x) = (1.5 - x_1 + x_1 x_2)^2 + (2.25 - x_1 + x_1 x_2^2)^2 +
                (2.625 - x_1 + x_1 x_2^3)^2

    f has one minimizer for its global minimum at `x^* = (3.0, 0.5)` with
    `f(x^*) = 0`.
    """
    dim: int = 2
    _optimal_value: float = 0.0
    _optimizers: List[Tuple[float, float]] = [(3.0, 0.5)]

    def __init__(
            self,
            backend: str,
            bounds: Optional[List[Tuple[float, float]]] = None,
            maximize: Optional[bool] = False
    ) -> None:
        super().__init__(backend=backend, bounds=bounds, maximize=maximize)
        if bounds is None:
            self.bounds = [(-4.5, 4.5), (-4.5, 4.5)]

    def _evaluate(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        part3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        return (part1 + part2 + part3).reshape(-1, 1)
