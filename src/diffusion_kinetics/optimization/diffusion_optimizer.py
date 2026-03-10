from diffusion_kinetics.optimization.diffusion_objective import DiffusionObjective
from diffusion_kinetics.optimization.dataset import Dataset
from diffusion_kinetics.pipeline.pipeline_config import SingleProcessPipelineConfig
from scipy.optimize import differential_evolution, NonlinearConstraint
from diffusion_kinetics.optimization.conHe_Param import conHe_Param
import numpy as np
import torch


class DiffusionOptimizer:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def run(self, config: SingleProcessPipelineConfig, seed: int = 0):
        """
        Run the optimization for a given config.

        Args:
            config (SingleProcessPipelineConfig): Optimization configuration.
            seed (int): Random seed for the differential-evolution algorithm.
        """
        bounds = self._construct_bounds(config)
        nlcs = self._construct_nlcs(config.num_domains)
        objective = DiffusionObjective(
            self.dataset,
            config.omit_value_indices,
            config.misfit_stat,
            config.geometry,
            config.punish_degas_early,
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

    def _construct_bounds(self, config: SingleProcessPipelineConfig):
        # chisq optimises total_moles jointly with kinetics parameters
        moles = config.misfit_stat.lower() == "chisq"

        frac_bounds = (0, 1)
        delM_unc = torch.sqrt(torch.sum(torch.tensor(self.dataset.delM) ** 2))
        total_M = sum(self.dataset.M)
        mole_bound = (total_M - delM_unc, total_M + delM_unc)

        if config.num_domains == 1:
            if moles:
                return [mole_bound, config.ea_bounds, config.lnd0aa_bounds]
            else:
                return [config.ea_bounds, config.lnd0aa_bounds]
        else:
            if moles:
                return (
                    [mole_bound, config.ea_bounds]
                    + config.num_domains * [config.lnd0aa_bounds]
                    + (config.num_domains - 1) * [frac_bounds]
                )
            else:
                return (
                    [config.ea_bounds]
                    + config.num_domains * [config.lnd0aa_bounds]
                    + (config.num_domains - 1) * [frac_bounds]
                )

    def _construct_nlcs(self, ndom: int):
        if ndom > 1:
            return NonlinearConstraint(conHe_Param, lb=[0], ub=[np.inf])
        else:
            return []
