from diffusion_kinetics.optimization.diffusion_objective import DiffusionObjective
from diffusion_kinetics.optimization.dataset import Dataset
from diffusion_kinetics.pipeline.pipeline_config import SingleProcessPipelineConfig
from diffusion_kinetics.optimization.con_he_param import con_he_param
from scipy.optimize import differential_evolution, NonlinearConstraint
import numpy as np
import torch


class DiffusionOptimizer:
    """Wraps scipy's ``differential_evolution`` with MDD-specific bounds and
    constraints derived from a :class:`SingleProcessPipelineConfig`.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def run(self, config: SingleProcessPipelineConfig, seed: int = 0):
        """Run the optimization for a given configuration.

        Args:
            config (SingleProcessPipelineConfig): Optimization configuration.
            seed (int): Random seed passed to ``differential_evolution``.
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
        # chisq optimises total_moles jointly with the kinetic parameters
        uses_moles = config.misfit_stat.lower() == "chisq"

        frac_bounds = (0, 1)
        del_m_unc = torch.sqrt(torch.sum(torch.tensor(self.dataset.delM) ** 2))
        total_m = sum(self.dataset.M)
        mole_bound = (total_m - del_m_unc, total_m + del_m_unc)

        if config.num_domains == 1:
            if uses_moles:
                return [mole_bound, config.ea_bounds, config.lnd0aa_bounds]
            else:
                return [config.ea_bounds, config.lnd0aa_bounds]
        else:
            if uses_moles:
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

    def _construct_nlcs(self, n_dom: int):
        if n_dom > 1:
            return NonlinearConstraint(con_he_param, lb=[0], ub=[np.inf])
        else:
            return []
