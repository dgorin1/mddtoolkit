
from diffusion_kinetics.optimization import DiffusionObjective, Dataset
from diffusion_kinetics.pipeline import SingleProcessPipelineConfig
from scipy.optimize import differential_evolution, NonlinearConstraint
from diffusion_kinetics.optimization.conHe_Param import conHe_Param
import numpy as np
import torch

class DiffusionOptimizer:
    def __init__(
        self, 
        dataset:Dataset, 
    ):
        self.dataset = dataset
        
    def run(self, config:SingleProcessPipelineConfig, seed:int=0):
        """
        Run the optimization for a given misfit statistic and number of domains.
        
        Args:
            - misfit_stat (str): The misfit statistic to use.
            - ndom (int): The number of domains to use.
        """
        bounds = self._construct_bounds(config)
        nlcs = self._construct_nlcs(config.num_domains) 
        objective = DiffusionObjective(
            self.dataset, 
            config.time_add,
            config.temp_add,
            config.omit_value_indices,
            config.misfit_stat,
            config.geometry,
            config.punish_degas_early
        )

        return differential_evolution(
            objective,
            bounds,
            disp=False,
            tol=config.tol,  
            maxiter=config.max_iters,
            constraints=nlcs,
            vectorized=True,
            updating=config.updating,
            seed=seed,
            popsize=config.popsize,
            strategy=config.strategy,
            mutation=config.mutation,
            recombination=config.recombination,
            init=config.init,
        )
    
    def _construct_bounds(self, config:SingleProcessPipelineConfig):
        if (
            config.misfit_stat.lower() == "chisq"
            or config.misfit_stat.lower() == "l2_moles"
            or config.misfit_stat.lower() == "l1_moles"
        ):
            moles = True
        else:
            moles = False

        frac_bounds = (0, 1)
        mole_bound = tuple((
                sum(self.dataset.M) - 1 * torch.sqrt(sum(torch.tensor(self.dataset.delM) ** 2)),
                sum(self.dataset.M) + 1 * torch.sqrt(sum(torch.tensor(self.dataset.delM) ** 2)),
            )
        )

        if config.num_domains == 1:
            if moles == True:
                return [mole_bound, config.ea_bounds, config.lnd0aa_bounds]
            else:
                return [config.ea_bounds, config.lnd0aa_bounds]
        elif config.num_domains > 1:
            if moles == True:
                return (
                    [mole_bound, config.ea_bounds]
                    + config.num_domains * [config.lnd0aa_bounds]
                    + (config.num_domains - 1) * [frac_bounds]
                )
            else:
                return [config.ea_bounds] + config.num_domains * [config.lnd0aa_bounds] + (config.num_domains - 1) * [frac_bounds]
    
       
    def _construct_nlcs(self, ndom:int):
        if ndom > 1:
            return NonlinearConstraint(conHe_Param, lb=[0], ub=[np.inf])
        else:
            return []
        