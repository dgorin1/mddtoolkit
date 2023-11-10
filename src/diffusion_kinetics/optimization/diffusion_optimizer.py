
from diffusion_kinetics.optimization import DiffusionObjective, Dataset
from diffusion_kinetics.pipeline import PipelineConfig
from scipy.optimize import differential_evolution, NonlinearConstraint
from diffusion_kinetics.optimization.conHe_Param import conHe_Param
import numpy as np

class DiffusionOptimizer:
    def __init__(
        self, 
        dataset:Dataset, 
        config:PipelineConfig, 
        nlc:list=NonlinearConstraint(conHe_Param, lb=[0], ub=[np.inf])
    ):
        self.dataset = dataset
        self.config = config
        self.nlc = nlc
        
    def run(self, misfit_stat:str, ndom:int, seed:int=0):
        """
        Run the optimization for a given misfit statistic and number of domains.
        
        Args:
            - misfit_stat (str): The misfit statistic to use.
            - ndom (int): The number of domains to use.
        """
        bounds = self.config.generate_bounds(ndom)
        
        objective = DiffusionObjective(
            self.dataset, 
            self.config.time_add,
            self.config.temp_add,
            self.config.omit_value_indices,
            misfit_stat,
            self.config.geometry,
            self.config.punish_degas_early
        )
        
        result = differential_evolution(
            objective,
            bounds,
            disp=False,
            tol=0.0001,  # zeros seems like a good number from testing. slow, but useful.
            maxiter=self.config.iteration_repeats,
            constraints=self.nlc,
            vectorized=True,
            updating="deferred",
            seed=seed,
        )
        
        return result