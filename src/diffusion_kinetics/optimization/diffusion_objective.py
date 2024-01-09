from diffusion_kinetics.optimization.forward_model_kinetics import forwardModelKinetics
from diffusion_kinetics.optimization.forward_model_kinetics import calc_lnd0aa
from diffusion_kinetics.optimization.dataset import Dataset
import math as math
import torch as torch
import numpy as np


class DiffusionObjective:
    def __init__(
        self,
        data: Dataset,
        time_add: list,
        temp_add: list,
        omitValueIndices=[],
        stat: str = "chisq",
        geometry: str = "spherical",
        punish_degas_early:bool = True
    ):
        """
        This function forward models a set of MDD parameters and returns a residual based on the specified misfit statistic.
        
        Args:
            data (Dataset): the dataset to be used for the objective function.
            time_add (list): the times of the extra heating steps to be added to the dataset.
            temp_add array (list): the temperatures of the extra heating steps to be added to the dataset.
            omitValueIndices (list, optional): the indices of the values to be omitted from the objective function. Defaults to [].
            stat (str, optional): the statistic to be used for the objective function. Defaults to "chisq".
            geometry (str, optional): the geometry of the sample. Defaults to "spherical".
            punish_degas_early(bool, optional): Tells the model whether to punish proposed fits that degas before the modeled experiment is complete
        """

        self.dataset = data

        self.time_add = time_add
        self.temp_add = temp_add

        # Add total moles information for priors
        self.total_moles = torch.sum(torch.tensor(self.dataset.M))
        self.total_moles_del = torch.sqrt(
            torch.sum(torch.tensor(self.dataset.delM) ** 2)
        )

        
        self.stat = stat
        time = self.dataset._thr * 3600
        if time_add.numel() > 0:
            self.tsec = torch.cat([time_add, time])
            self._TC = torch.cat([temp_add, self.dataset._TC])
            self.extra_steps = True
        else:
            self.tsec = time
            self._TC = self.dataset._TC
            self.extra_steps = False

        self.lnd0aa = torch.tensor(self.dataset["ln(D/a^2)"])
        self.lnd0aa[-1] = 0

        indices = np.where(np.isinf(self.lnd0aa))
        self.lnd0aa[self.lnd0aa == -np.inf] = 0
        self.lnd0aa[self.lnd0aa == -np.inf] = 0
        self.lnd0aa[torch.isnan(self.lnd0aa)] = 0
        self.omitValueIndices = torch.isin(   torch.arange(len(self.dataset)), torch.tensor(omitValueIndices)   ).to(torch.int)
        omitValueIndices_lnd0aa = omitValueIndices + (indices[0].tolist())
        self.omitValueIndices_lnd0aa = torch.isin(
            torch.arange(len(self.dataset)), torch.tensor(omitValueIndices_lnd0aa)
        ).to(torch.int)

        # Add locations where uncertainty (and measurement value) is zero to the list of values to ignore
        indices_chisq = np.where(data.uncert == 0)
        omitValueIndices_chisq = omitValueIndices + (indices_chisq[0].tolist())
        self.omitValueIndices_chisq = torch.isin(
            torch.arange(len(self.dataset)), torch.tensor(omitValueIndices_chisq)
        ).to(torch.int)

        self.plateau = torch.sum(
            ((torch.tensor(self.dataset.M) - torch.zeros(len(self.dataset.M))) ** 2)
            / (data.uncert**2)
        )
        self.Fi = torch.tensor(data.Fi)
        self.trueFracFi = self.Fi[1:] - self.Fi[0:-1]
        self.trueFracFi = torch.concat((torch.unsqueeze(self.Fi[0], dim=-0), self.trueFracFi), dim=-1)
        self.geometry = geometry
        self.Daa_uncertainty = torch.tensor(self.dataset["Daa uncertainty"])
        self.Daa_uncertainty[self.Daa_uncertainty == -np.inf] = 0
        self.Daa_uncertainty[self.Daa_uncertainty == -np.inf] = 0
        self.Daa_uncertainty[torch.isnan(self.Daa_uncertainty)] = 0
        data.uncert[data.uncert == 0] = torch.min(data.uncert[data.uncert != 0]) * 0.1
        self.exp_moles = torch.tensor(data.M)
        self.added_steps = len(time_add)
        self.punish_degas_early = punish_degas_early

    def __call__(self, X):
        return self.objective(X)

    def grad(self, X):
        return self.grad(X)

    def objective(self, X):
        
        # This function calculates the fraction of gas released from each domain
        # in an MDD model during the heating schedule used in the diffusion
        # experiment. Then the fractions released from each domain are combined in
        # proportion to one another as specified by the MDD model, and the
        # diffusivity of each step is calculated. A residual is calculated as prescribed by the user.
        # If the constraint function removes all possible models for this step, return empty list

        if X.size == 0:
            return([])


        # Determine whether or not moles are being calculated and save to variable if yes
        if len(X) % 2 != 0:
            total_moles = X[0]
            X = X[1:]

        # Forward model the results so that we can calculate the misfit.

        # If the mineral is diffusive enough that we're correcting for laboratory storage and irradiation:
        # if self.extra_steps == True:
        Fi_MDD, punishmentFlag = forwardModelKinetics(
            X,
            self.tsec,
            self._TC,
            geometry=self.geometry,
            added_steps=self.added_steps,
        )

        # Create a punishment flag if the user specified. If the experiment degassed before the end of the temperature steps,
        # then we add an extra value to the misfit calculated at each step. We do this by multiplying the misfit value 
        # at each step that degassed too early by 10. This punishes the model and "teaches" it that we don't want the experiment
        # to degas too early. This is not recommended for experiments where the sample was fused or melted in final steps.
        if self.punish_degas_early == True:
            punishmentFlag = punishmentFlag * 10 + 1
        else:
            punishmentFlag = 1

        
        # Objective function calculates Fi in cumulative form. Switch into non-cumulative space for calculations.               
        trueFracMDD = Fi_MDD[1:] - Fi_MDD[0:-1]
        trueFracMDD = torch.concat(
            (torch.unsqueeze(Fi_MDD[0], dim=0), trueFracMDD), dim=0
        )

        # If only one history was tested in a 1D shape, we need to put it into a column shape in 2-d so that 
        # it is the correct dimensions for the calculations below. If only one history was tested, 
        # trueFracMDD will have just 1 dimension.
        if len(trueFracMDD.shape) < 2:
            trueFracMDD = torch.unsqueeze(trueFracMDD, 1)

        # Assign to a variable since we need to modify the shape of this variable depending on the size of X
        trueFracFi = self.trueFracFi
      
      
        if (
            self.stat.lower() == "l1_frac"
            or self.stat.lower() == "l2_frac"
            or self.stat.lower() == "percent_frac"
            or self.stat == "lnd0aa"
            or self.stat == "lnd0aa_chisq"
        ):
            
            trueFracFi = torch.tile(
                trueFracFi.unsqueeze(1), [1, trueFracMDD.shape[1]]
            )

            # If you're using percent_frac, we'll reassign all the values that are 0 to 
            # 10% of the minimum size in seen in the experiment to avoid "divide by zero" errors.
            # We ignore zero steps when calculating the misfit anyway.
            if self.stat == "percent_frac":
                trueFracFi[trueFracFi == 0] = (
                    torch.min(trueFracFi[trueFracFi != 0]) * 0.1
                )
        # If using l1_frac_cum, make that variable the correct shape.
        elif self.stat.lower() == "l1_frac_cum":
            Fi = torch.tile(self.Fi.unsqueeze(1), [1, Fi_MDD.shape[1]])
        
        # If nothing above is true, then you're using a calc that involves moles and you'll need to calculate
        # the predicted moles for each step.
        else:
            moles_MDD = trueFracMDD * total_moles

        # Create the multiplier mask which will show values of 1 for values we want to include in the misfit, and 
        # zero for those we don't. 

        # This one is specific for lnd0aa (DREW TO ADD BETTER COMMENT)
        if self.stat.lower() == "lnd0aa" or self.stat.lower() == "lnd0aa_chisq":
            multiplier = 1 - torch.tile(
                self.omitValueIndices_lnd0aa.unsqueeze(1),
                [1, trueFracMDD.shape[1]],
            )

        
        # This one is specific for chi_sq (DREW TO ADD BETTER COMMENT)
        elif self.stat.lower() == "chisq":
            multiplier = 1 - torch.tile(
                self.omitValueIndices_chisq.unsqueeze(1),
                [1, trueFracMDD.shape[1]],
            )

        # This is the last multiplier contianing user-specified indices.
        else:
            multiplier = 1 - torch.tile(
                self.omitValueIndices.unsqueeze(1), [1, trueFracMDD.shape[1]]
            )

        
        # This is a giant if statement to decide which misfit statistic you're using. 
        # It calculates the misfit as appropriate.
        if self.stat.lower() == "chisq":
            misfit = torch.sum(
                multiplier
                * ((self.exp_moles.unsqueeze(1) - moles_MDD) ** 2)
                / (self.dataset.uncert.unsqueeze(1) ** 2),
                axis=0,
            )
        elif self.stat.lower() == "l1_moles":
            misfit = misfit = torch.sum(
                multiplier * (torch.abs(self.exp_moles.unsqueeze(1) - moles_MDD)),
                axis=0,
            )
        elif self.stat.lower() == "l2_moles":
            misfit = torch.sum(
                (multiplier * ((self.exp_moles.unsqueeze(1) - moles_MDD) ** 2)),
                axis=0,
            )
        elif self.stat.lower() == "l1_frac":
            misfit = torch.sum(
                multiplier * (torch.abs(trueFracFi - trueFracMDD)), axis=0
            )
        elif self.stat.lower() == "l1_frac_cum":
            misfit = torch.sum(multiplier * (torch.abs(Fi - Fi_MDD)), axis=0)
        elif self.stat.lower() == "l2_frac":
            misfit = torch.sum(
                (multiplier * (trueFracFi - trueFracMDD) ** 2), axis=0
            )
        elif self.stat.lower() == "percent_frac":
            misfit = torch.sum(
                multiplier * (torch.abs(trueFracFi - trueFracMDD)) / trueFracFi,
                axis=0,
            )
        elif self.stat.lower() == "lnd0aa":
            lnd0aa_MDD = calc_lnd0aa(
                Fi_MDD, self.tsec, self.geometry, self.extra_steps, self.added_steps
            )
            lnd0aa_MDD[lnd0aa_MDD == -np.inf] = 0
            lnd0aa_MDD[lnd0aa_MDD == np.inf] = 0
            lnd0aa_MDD[torch.isnan(lnd0aa_MDD)] = 0

            misfit = torch.sum(
                multiplier * ((lnd0aa_MDD - self.lnd0aa.unsqueeze(1)) ** 2),
                axis=0,
            )
        
        elif self.stat.lower() == "lnd0aa_chisq":
            lnd0aa_MDD = calc_lnd0aa(
                Fi_MDD, self.tsec, self.geometry, self.extra_steps, self.added_steps
            )
            lnd0aa_MDD[lnd0aa_MDD == -np.inf] = 0
            lnd0aa_MDD[lnd0aa_MDD == np.inf] = 0
            lnd0aa_MDD[torch.isnan(lnd0aa_MDD)] = 0
            if len(lnd0aa_MDD.shape) < 2: 
                lnd0aa_MDD = torch.unsqueeze(lnd0aa_MDD ,1)
            
            misfit = multiplier * ((torch.exp(lnd0aa_MDD) - torch.exp(self.lnd0aa.unsqueeze(1)))** 2/ self.Daa_uncertainty.unsqueeze(1))
            nan_rows = (torch.isnan(misfit).any(dim=1)) | (torch.isinf(misfit).any(dim=1))
            misfit = torch.sum(misfit[~nan_rows], axis=0)

        return misfit * punishmentFlag
    

   