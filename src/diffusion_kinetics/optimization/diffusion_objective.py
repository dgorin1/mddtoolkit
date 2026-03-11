from diffusion_kinetics.optimization.forward_model_kinetics import forward_model_kinetics
from diffusion_kinetics.optimization.dataset import Dataset
import torch
import numpy as np


class DiffusionObjective:
    """
    Calculates the objective function (misfit) between experimental diffusion data
    and a Multi-Domain Diffusion (MDD) forward model.

    Supported misfit statistics:
        - ``"chisq"``:        chi-squared on measured moles vs. predicted moles.
        - ``"percent_frac"``: mean absolute fractional-release residual, normalised
                              by the observed fractional release at each step.
    """

    def __init__(
        self,
        data: Dataset,
        omit_value_indices: list = [],
        stat: str = "chisq",
        geometry: str = "spherical",
        punish_degas_early: bool = True,
    ):
        """
        Args:
            data (Dataset): Experimental dataset.
            omit_value_indices (list): Indices of steps to exclude from the misfit.
            stat (str): Misfit statistic (``"chisq"`` or ``"percent_frac"``).
            geometry (str): Diffusion geometry (``"spherical"`` or ``"plane sheet"``).
            punish_degas_early (bool): If ``True``, penalises models that exhaust
                gas before the final heating step.
        """
        self.dataset = data
        self.stat = stat
        self.geometry = geometry
        self.punish_degas_early = punish_degas_early

        self.tsec = data._thr * 3600
        self._tc = data._tc

        self.omit_value_indices = torch.isin(
            torch.arange(len(data)), torch.tensor(omit_value_indices)
        ).to(torch.int)

        # For chisq: also exclude steps where uncertainty is zero
        indices_zero_uncert = np.where(data.uncert == 0)[0].tolist()
        omit_value_indices_chisq = omit_value_indices + indices_zero_uncert
        self.omit_value_indices_chisq = torch.isin(
            torch.arange(len(data)), torch.tensor(omit_value_indices_chisq)
        ).to(torch.int)

        data.uncert[data.uncert == 0] = torch.min(data.uncert[data.uncert != 0]) * 0.1
        self.exp_moles = torch.tensor(data.M)

        self.fi = torch.tensor(data.Fi)
        self.true_frac_fi = torch.concat(
            (self.fi[:1], self.fi[1:] - self.fi[:-1]), dim=0
        )

    def __call__(self, X):
        return self.objective(X)

    def objective(self, X):
        if X.size == 0:
            return []

        # chisq prepends total_moles, giving an odd-length parameter vector
        total_moles = None
        if len(X) % 2 != 0:
            total_moles = X[0]
            X = X[1:]

        fi_mdd, punishment_flag = forward_model_kinetics(
            X,
            self.tsec,
            self._tc,
            geometry=self.geometry,
            added_steps=0,
        )

        if self.punish_degas_early:
            punishment_flag = punishment_flag * 10 + 1
        else:
            punishment_flag = 1

        # Convert cumulative → differential fractional release
        true_frac_mdd = torch.concat(
            (fi_mdd[:1], fi_mdd[1:] - fi_mdd[:-1]), dim=0
        )

        if len(true_frac_mdd.shape) < 2:
            true_frac_mdd = torch.unsqueeze(true_frac_mdd, 1)

        if self.stat.lower() == "chisq":
            if total_moles is None:
                raise ValueError(
                    "chisq requires an odd-length parameter vector with total_moles prepended"
                )
            multiplier = 1 - torch.tile(
                self.omit_value_indices_chisq.unsqueeze(1),
                [1, true_frac_mdd.shape[1]],
            )
            moles_mdd = true_frac_mdd * total_moles
            misfit = torch.sum(
                multiplier
                * ((self.exp_moles.unsqueeze(1) - moles_mdd) ** 2)
                / (self.dataset.uncert.unsqueeze(1) ** 2),
                axis=0,
            )

        elif self.stat.lower() == "percent_frac":
            multiplier = 1 - torch.tile(
                self.omit_value_indices.unsqueeze(1), [1, true_frac_mdd.shape[1]]
            )
            true_frac_fi = torch.tile(
                self.true_frac_fi.unsqueeze(1), [1, true_frac_mdd.shape[1]]
            )
            true_frac_fi[true_frac_fi == 0] = torch.min(true_frac_fi[true_frac_fi != 0]) * 0.1
            misfit = torch.sum(
                multiplier * torch.abs(true_frac_fi - true_frac_mdd) / true_frac_fi,
                axis=0,
            )

        return misfit * punishment_flag
