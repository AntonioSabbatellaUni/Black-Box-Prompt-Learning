from pymoo.algorithms.soo.nonconvex.cmaes import CMAES as PymooCMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA, FitnessSurvival, comp_by_cv_and_fitness
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.selection import Selection
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection

from bbopy.algorithms.base import PymooAlgorithm
from bbopy.problems.base import Problem
from bbopy.sampling import Sampling, FloatRandomSampling


class GeneticAlgorithm(PymooAlgorithm):
    r"""A simple Genetic Algorithm.

    It uses the Pymoo implementation of a Genetic Algorithm.
    """
    name: str = "Genetic Algorithm"

    def __init__(
            self,
            pop_size: int = 100,
            sampling: Sampling = FloatRandomSampling(backend="pymoo"),
            selection: Selection = TournamentSelection(func_comp=comp_by_cv_and_fitness),
            crossover: Crossover = SBX(),
            mutation: Mutation = PM(),
            survival: Survival = FitnessSurvival(),
            n_offsprings: int = None,
    ) -> None:
        r"""Initializes the genetic operators and the algorithm settings.

        Args:
            pop_size (int): The dimension of the population.
            sampling (Sampling): The strategy to sample initial points (Be sure to use the pymoo backend).
            selection (Selection): The selection operator.
            crossover (Crossover): The crossover operator.
            mutation (Mutation): The mutation operator.
            survival (Survival): The survival operator.
            n_offsprings (int): The number of new offsprings to be generated at each generation.
                If None `n_offsprings`=`pop_size`.
        """
        super().__init__()
        self._algo = GA(pop_size=pop_size,
                        sampling=sampling,
                        selection=selection,
                        crossover=crossover,
                        mutation=mutation,
                        survival=survival,
                        n_offsprings=n_offsprings)


class DifferentialEvolution(PymooAlgorithm):
    r"""The Differential Evolution algorithm.

    It uses the Pymoo implementation of a Differential Evolution.
    """
    name: str = "Differential Evolution"

    def __init__(
            self,
            pop_size: int = 100,
            sampling: Sampling = FloatRandomSampling(backend="pymoo"),
            variant: str = "DE/rand/1/bin",
            CR: float = 0.3,
            dither: str = "vector",
            jitter: bool = False,
            **kwargs
    ) -> None:
        r"""Initializes the genetic operators and the algorithm settings.

        Args:
            pop_size (int): The dimension of the population.
            sampling (Sampling): The strategy to sample initial points (Be sure to use the pymoo backend).
            variant (str): The variant of the method to be used. It is a "/" separated string which represents
                the name of the algorithm (e.g., "DE"), the selection operator (e.g., "rand"),
                the number of difference vectors (e.g., "1") and the crossover operator (e.g., "bin").
            CR (float): The crossover constant.
            dither (str): The technique used to select the weighting factor F.
            jitter (bool): Whether to apply jitter.
            **kwargs (dict): Additional arguments.
        """
        super().__init__()
        self._algo = DE(pop_size=pop_size,
                        sampling=sampling,
                        variant=variant,
                        CR=CR,
                        dither=dither,
                        jitter=jitter,
                        **kwargs)


class CMAES(PymooAlgorithm):
    r"""The Covariance Matrix Adaptation Evolutionary Strategy algorithm.

    It uses the Pymoo implementation of CMAES.
    """
    name: str = "CMAES"

    def __init__(
            self,
            sampling: Sampling = FloatRandomSampling(backend="pymoo"),
            **kwargs
    ) -> None:
        r"""Initializes the genetic operators and the algorithm settings.

        Args:
            sampling (Sampling): The strategy to sample initial points (Be sure to use the pymoo backend).
            kwargs (dict): Additional arguments passed to the Pymoo CMAES constructor.
        """
        super().__init__()
        self._algo_kwargs = kwargs
        self._sampling = sampling

    def _setup(
            self,
            problem: Problem
    ) -> None:
        r"""Sets the starting point and initializes the Pymoo CMAES algorithm.

        Args:
            problem (Problem): The problem to be optimized.
        """
        x0 = self._sampling(problem.bounds, 1)
        self._algo = PymooCMAES(
            x0=x0,
            **self._algo_kwargs,
        )


class EvolutionaryStrategy(PymooAlgorithm):
    """The Evolutionary Strategy algorithm.

    It uses the Pymoo implementation of ES.
    """
    name: str = "Evolutionary Strategy"

    def __init__(
            self,
            pop_size: int = None,
            sampling: Sampling = FloatRandomSampling(backend="pymoo"),
            survival: Survival = FitnessSurvival(),
            n_offsprings: int = 200,
            rule: float = 1.0 / 7.0,
            phi: float = 1.0,
            gamma: float = 0.85
    ) -> None:
        r"""Initializes the genetic operators and the algorithm settings.

        Args:
            pop_size (int): The number of individuals which are surviving
                from the offspring population (non-elitist).
            sampling (Sampling): The sampling method for creating the initial population.
            survival (Survival): The survival operator.
            n_offsprings (int): The number of individuals created in each iteration.
            rule (float): The rule (ratio) of individuals surviving. This automatically either
                calculated `n_offsprings` or `pop_size`.
            phi (float): Expected rate of convergence (usually 1.0).
            gamma (float): If not `None`, some individuals are created using the differentials
                with this as a length scale.
        """
        super().__init__()
        self._algo = ES(
            n_offsprings=n_offsprings,
            pop_size=pop_size,
            rule=rule,
            phi=phi,
            gamma=gamma,
            sampling=sampling,
            survival=survival)
