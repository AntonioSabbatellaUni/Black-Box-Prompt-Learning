from typing import Tuple, List
import torch
from torch import Tensor  
import cma
import numpy as np

class AcquisitionOptimizer:
    """Optimizes an acquisition function using CMA-ES.

    Attributes:
        acquisition_fn: The acquisition function to optimize.
        bounds: The bounds for the input space as (lower, upper).
        sigma0: Initial step size for CMA-ES.
        popsize: Population size for CMA-ES.

    """
    
    def __init__(
        self,
        acquisition_fn: callable,
        q: int = 1,
        bounds: List[Tuple[float, float]] = [(0, 1)],
        sigma0: float = 0.2,
        popsize: int = 50  
    ) -> None:
        """Initialize the optimzier."""
        self.acquisition_fn = acquisition_fn
        self.q = q  
        self.bounds = bounds
        self.sigma0 = sigma0
        self.popsize = popsize

    def optimize(
        self, 
    ) -> List[Tensor]:
        """Optimize the acquisition function.

        Returns:
            best_x: The q best points found by optimizing the acquisition function.  

        """
        
        # Get initial random points
        x0 = np.random.rand(self.q, len(self.bounds[0]))* np.array(self.bounds[1])
        
        # Create CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(
            x0=x0, 
            sigma0=self.sigma0,
            inopts={
                "bounds": self.bounds, 
                "popsize": self.popsize
            }
        )
            
        # Optimization loop
        with torch.no_grad():
            while not es.stop():
                xs = es.ask()
                X = torch.tensor(xs, dtype=torch.float32)
                
                # Evaluate acquisition function
                Y = -self.acquisition_fn(X.unsqueeze(-2))
                
                # Convert result to numpy array
                y = Y.view(-1).numpy()
                
                # Return result
                es.tell(xs, y)
                
        # Get best points  
        best_x = torch.from_numpy(es.best.x)
                    
        if best_x.dim() == 1:
            best_x = best_x.unsqueeze(0)  # Convert 1D tensor to 2D tensor
        return best_x