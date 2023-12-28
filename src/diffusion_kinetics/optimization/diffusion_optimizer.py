
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
        config:SingleProcessPipelineConfig, 
    ):
        self.dataset = dataset
        self.config = config
        
    def run(self, misfit_stat:str, ndom:int, iters=10, seed:int=0):
        """
        Run the optimization for a given misfit statistic and number of domains.
        
        Args:
            - misfit_stat (str): The misfit statistic to use.
            - ndom (int): The number of domains to use.
        """
        print("Running optimization for {} with {} domains".format(misfit_stat, ndom))
        bounds = self._construct_bounds(misfit_stat, ndom)
        nlcs = self._construct_nlcs(ndom)

        objective = DiffusionObjective(
            self.dataset, 
            self.config.time_add,
            self.config.temp_add,
            self.config.omit_value_indices,
            misfit_stat,
            self.config.geometry,
            self.config.punish_degas_early
        )
        
        misfits = []
        results = []
        for i in range(iters):
            result = differential_evolution(
                objective,
                bounds,
                disp=False,
                tol=0.0001,  
                maxiter=self.config.max_iters,
                constraints=nlcs,
                vectorized=True,
                updating="deferred",
                seed=seed,
                popsize=15
            )

            misfits.append(result.fun)
            print(f"misfit: {result.fun}")
            print(f"number of iterations: {result.nit}")
            results.append(result)

            seed += 1

        index = np.argmin(misfits)
        return results[index]
        # result = differential_evolution(
        #     objective,
        #     bounds,
        #     disp=False,
        #     tol=0.0001,  # zeros seems like a good number from testing. slow, but useful.
        #     maxiter=self.config.max_iters,
        #     constraints=nlcs,
        #     vectorized=True,
        #     updating="deferred",
        #     seed=seed
        # )
        
        # return result
    
    def _construct_bounds(self, stat:str, ndom:int):
        if (
            stat.lower() == "chisq"
            or stat.lower() == "l2_moles"
            or stat.lower() == "l1_moles"
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

        if ndom == 1:
            if moles == True:
                return [mole_bound, self.config.ea_bounds, self.config.lnd0aa_bounds]
            else:
                return [self.config.ea_bounds, self.config.lnd0aa_bounds]
        elif ndom > 1:
            if moles == True:
                return (
                    [mole_bound, self.config.ea_bounds]
                    + ndom * [self.config.lnd0aa_bounds]
                    + (ndom - 1) * [frac_bounds]
                )
            else:
                return [self.config.ea_bounds] + ndom * [self.config.lnd0aa_bounds] + (ndom - 1) * [frac_bounds]
    
       
    def _construct_nlcs(self, ndom:int):
        if ndom > 1:
            return NonlinearConstraint(conHe_Param, lb=[0], ub=[np.inf])
        else:
            return []
        