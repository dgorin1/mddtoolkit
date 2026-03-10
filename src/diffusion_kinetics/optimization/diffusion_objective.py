from diffusion_kinetics.optimization.forward_model_kinetics import forwardModelKinetics
from diffusion_kinetics.optimization.dataset import Dataset
import torch
import numpy as np


class DiffusionObjective:
    """
    Calculates the objective function (misfit) between experimental diffusion data
    and a Multi-Domain Diffusion (MDD) forward model.

    Supported misfit statistics:
        - "chisq":        chi-squared on measured moles vs. predicted moles.
        - "percent_frac": mean absolute fractional-release residual, normalised by
                          the observed fractional release at each step.
    """

    def __init__(
        self,
        data: Dataset,
        omitValueIndices: list = [],
        stat: str = "chisq",
        geometry: str = "spherical",
        punish_degas_early: bool = True,
    ):
        """
        Args:
            data (Dataset): Experimental dataset.
            omitValueIndices (list): Indices of steps to exclude from the misfit.
            stat (str): Misfit statistic ("chisq" or "percent_frac").
            geometry (str): Diffusion geometry ("spherical" or "plane sheet").
            punish_degas_early (bool): If True, penalises models that exhaust gas
                before the final heating step.
        """
        self.dataset = data
        self.stat = stat
        self.geometry = geometry
        self.punish_degas_early = punish_degas_early

        self.tsec = data._thr * 3600
        self._TC = data._TC

        self.omitValueIndices = torch.isin(
            torch.arange(len(data)), torch.tensor(omitValueIndices)
        ).to(torch.int)

        # For chisq: also exclude steps where uncertainty is zero
        indices_zero_uncert = np.where(data.uncert == 0)[0].tolist()
        omitValueIndices_chisq = omitValueIndices + indices_zero_uncert
        self.omitValueIndices_chisq = torch.isin(
            torch.arange(len(data)), torch.tensor(omitValueIndices_chisq)
        ).to(torch.int)

        data.uncert[data.uncert == 0] = torch.min(data.uncert[data.uncert != 0]) * 0.1
        self.exp_moles = torch.tensor(data.M)

        self.Fi = torch.tensor(data.Fi)
        self.trueFracFi = torch.concat(
            (self.Fi[:1], self.Fi[1:] - self.Fi[:-1]), dim=0
        )

    def __call__(self, X):
        return self.objective(X)

    def objective(self, X):
        if X.size == 0:
            return []

        # chisq prepends total_moles, giving an odd-length parameter vector
        if len(X) % 2 != 0:
            total_moles = X[0]
            X = X[1:]

        Fi_MDD, punishmentFlag = forwardModelKinetics(
            X,
            self.tsec,
            self._TC,
            geometry=self.geometry,
            added_steps=0,
        )

        if self.punish_degas_early:
            punishmentFlag = punishmentFlag * 10 + 1
        else:
            punishmentFlag = 1

        # Convert cumulative → differential fractional release
        trueFracMDD = torch.concat(
            (Fi_MDD[:1], Fi_MDD[1:] - Fi_MDD[:-1]), dim=0
        )

        if len(trueFracMDD.shape) < 2:
            trueFracMDD = torch.unsqueeze(trueFracMDD, 1)

        if self.stat.lower() == "chisq":
            multiplier = 1 - torch.tile(
                self.omitValueIndices_chisq.unsqueeze(1),
                [1, trueFracMDD.shape[1]],
            )
            moles_MDD = trueFracMDD * total_moles
            misfit = torch.sum(
                multiplier
                * ((self.exp_moles.unsqueeze(1) - moles_MDD) ** 2)
                / (self.dataset.uncert.unsqueeze(1) ** 2),
                axis=0,
            )

        elif self.stat.lower() == "percent_frac":
            multiplier = 1 - torch.tile(
                self.omitValueIndices.unsqueeze(1), [1, trueFracMDD.shape[1]]
            )
            trueFracFi = torch.tile(
                self.trueFracFi.unsqueeze(1), [1, trueFracMDD.shape[1]]
            )
            trueFracFi[trueFracFi == 0] = torch.min(trueFracFi[trueFracFi != 0]) * 0.1
            misfit = torch.sum(
                multiplier * torch.abs(trueFracFi - trueFracMDD) / trueFracFi,
                axis=0,
            )

        return misfit * punishmentFlag
