
from forwardModelKinetics import forwardModelKineticsDiffEV
from forwardModelKinetics import forward_model_kinetics_no_extra_heating
from forwardModelKinetics import calc_lnd0aa
from dataset import Dataset
import math as math 
import pickle
import jax
import torch as torch
import numpy as np
from jax import numpy as jnp


class DiffusionObjective():
    def __init__(self, method:str,data:Dataset, time_add: jnp.array, temp_add: jnp.array,pickle_path="../lookup_table.pkl",omitValueIndices = [], stat: str = "chisq", geometry:str = "spherical"):
        self.method = method
        if self.method == "ipopt":
            self.dataset = data
            self.lookup_table = pickle.load(open(pickle_path,'rb'))
            self.time_add = time_add
            self.temp_add = temp_add

            # Add total moles information for priors
            self.total_moles = torch.sum(torch.tensor(self.dataset.M))
            self.total_moles_del = torch.sqrt(torch.sum(torch.tensor(self.dataset.delM)**2))

            #self.omitValueIndices = jnp.array(omitValueIndices)
                  
            time = self.dataset._thr*3600
            self.tsec = jnp.concatenate([time_add,time])
            self._TC = jnp.concatenate([temp_add,self.dataset._TC])
            #(~np.isin(jnp.arange(len(self.dataset)), jnp.array(omitValueIndices))).astype(int)
            self.omitValueIndices = (~np.isin(jnp.arange(len(self.dataset)), jnp.array(omitValueIndices))).astype(int)
            #(~jnp.isin(jnp.arange(len(self.dataset)), jnp.array(omitValueIndices)).astype(int)
            self.grad = jax.grad(self.objective)
        
        elif self.method == "MCMC" or self.method == "diffEV":
            self.dataset = data
            self.lookup_table = pickle.load(open(pickle_path,'rb'))
            self.time_add = time_add
            self.temp_add = temp_add

            # Add total moles information for priors
            self.total_moles = torch.sum(torch.tensor(self.dataset.M))
            self.total_moles_del = torch.sqrt(torch.sum(torch.tensor(self.dataset.delM)**2))

            #self.omitValueIndices = jnp.array(omitValueIndices)
            self.stat = stat
            time = self.dataset._thr*3600
            if time_add.numel() >0:
                self.tsec = torch.cat([time_add,time])
                self._TC = torch.cat([temp_add,self.dataset._TC])
                self.extra_steps = True
            else:
                self.tsec = time
                self._TC = self.dataset._TC
                self.extra_steps = False
            
            
            self.lnd0aa = torch.tensor(self.dataset["ln(D/a^2)"])
            self.lnd0aa[-1] = 0

            indices = np.where(np.isinf(self.lnd0aa))
            self.lnd0aa[self.lnd0aa==-np.inf] = 0
            self.lnd0aa[self.lnd0aa==-np.inf] = 0
            self.lnd0aa[torch.isnan(self.lnd0aa)] = 0

            self.omitValueIndices = torch.isin(torch.arange(len(self.dataset)), torch.tensor(omitValueIndices)).to(torch.int)
            omitValueIndices_lnd0aa = omitValueIndices+(indices[0].tolist())
            self.omitValueIndices_lnd0aa = torch.isin(torch.arange(len(self.dataset)), torch.tensor(omitValueIndices_lnd0aa)).to(torch.int)

            # Add locations where uncertainty (and measurement value) is zero to the list of values to ignore
            indices_chisq = np.where(data.uncert == 0)
            omitValueIndices_chisq = omitValueIndices+(indices_chisq[0].tolist())
            self.omitValueIndices_chisq = torch.isin(torch.arange(len(self.dataset)), torch.tensor(omitValueIndices_chisq)).to(torch.int)


            self.plateau = torch.sum(((torch.tensor(self.dataset.M)-torch.zeros(len(self.dataset.M)))**2)/(data.uncert**2))
            self.Fi = torch.tensor(data.Fi)
            
            self.geometry = geometry
            self.Daa_uncertainty = torch.tensor(self.dataset["Daa uncertainty"])
            self.Daa_uncertainty[self.Daa_uncertainty==-np.inf] = 0
            self.Daa_uncertainty[self.Daa_uncertainty==-np.inf] = 0
            self.Daa_uncertainty[torch.isnan(self.Daa_uncertainty)] = 0
            data.uncert[data.uncert == 0] = torch.min(data.uncert[data.uncert !=0])*0.1







    
    
    def __call__(self, X):

        return self.objective(X)
    
    
    def grad(self, X):
        return self.grad(X)

    
    def objective(self, X): #__call__ #evaluate

        data = self.dataset
        # This function calculates the fraction of gas released from each domain
        # in an MDD model during the heating schedule used in the diffusion
        # experiment. Then the fractions released from each domain are combined in
        # proportion to one another as specified by the MDD model, and the
        # diffusivity of each step is calculated. A residual is calculated as the
        # sum of absolute differences between the observed and modeled release
        # fractions over all steps.

        # JOSH, DO I GET TO ASSUME THAT X IS GOING TO COME IN AS A JAX NP ARRAY?
        # X = jnp.array(X)
 
        if len(X) % 2 != 0:
            total_moles = X[0]
            X = X[1:]
        
        if len(X) <= 3:
            ndom = 1
        else:
            ndom = (len(X))//2

        # Grab the other parameters from the input
        temp = X[1:]

        


        if self.method == "diffEV":
            # Forward model the results so that we can calculate the misfit.

            if self.extra_steps == True:
                Fi_MDD,punishmentFlag = forwardModelKineticsDiffEV(X,self.lookup_table,self.tsec,self._TC,geometry = self.geometry)
            else:
                Fi_MDD, punishmentFlag = forward_model_kinetics_no_extra_heating(X,self.lookup_table,self.tsec,self._TC,geometry = self.geometry)
                
            punishmentFlag = punishmentFlag *10 + 1

            exp_moles = torch.tensor(data.M)
            if len(X.shape) > 1:

                if X.shape[1] == 0: #If we get passed an empty vector, which seems to happen when all generated samples do not meet constraints
                    return([])

                        # Calculate the fraction released for each heating step in the modeled experiment
                elif X.shape[1] ==1: 
                    trueFracMDD = Fi_MDD[1:]-Fi_MDD[0:-1]
                    trueFracMDD = torch.concat((torch.unsqueeze(Fi_MDD[0],dim=-0),trueFracMDD),dim=-1)

                    if self.stat == "l1_frac" or self.stat == "l2_frac" or self.stat == "percent_frac" or self.stat == "lnd0aa" or self.stat == "lnd0aa_chisq":
                        trueFracFi = self.Fi[1:] - self.Fi[0:-1]
                        trueFracFi = torch.concat((torch.unsqueeze(self.Fi[0],dim=-0),trueFracFi),dim=-1)
                        if self.stat == "percent_frac":
                            trueFracFi[trueFracFi==0] = torch.min(trueFracFi[trueFracFi!=0])*0.1

                    elif self.stat.lower() == "l1_frac_cum":
                        Fi = torch.tile(self.Fi.unsqueeze(1),[1,Fi_MDD.shape[0]])
                    else:
    
                        moles_MDD = trueFracMDD * total_moles

                    # Scale by chosen number of moles



                    if self.stat.lower() == "chisq":
                        misfit = torch.sum((1-self.omitValueIndices_chisq)*((exp_moles-moles_MDD)**2)/(data.uncert**2))
                    elif self.stat.lower() == "l1_moles":
                        misfit = torch.sum((1-self.omitValueIndices)*(torch.abs(exp_moles-moles_MDD)))
                    elif self.stat.lower() == "l2_moles":
                        misfit = torch.sum((1-self.omitValueIndices)*((exp_moles-moles_MDD)**2))
                    elif self.stat.lower() == "l1_frac":
                        misfit = torch.sum((1-self.omitValueIndices)*(torch.abs(trueFracFi-trueFracMDD)))
                    elif self.stat.lower() == "l2_frac":
                        misfit = torch.sum((1-self.omitValueIndices)*(trueFracFi-trueFracMDD)**2)
                    elif self.stat.lower() == "l1_frac_cum":
                        misfit = torch.sum((1-self.omitValueIndices)*torch.abs(Fi-Fi_MDD))
                    elif self.stat.lower() == "percent_frac":

                        temp1 = torch.abs(trueFracMDD-trueFracFi)
                        misfit = torch.sum((1-self.omitValueIndices)*(torch.abs(trueFracFi-trueFracMDD))/trueFracFi)
                    elif self.stat.lower() == "lnd0aa":
                        lnd0aa_MDD = calc_lnd0aa(Fi_MDD,self.tsec,self.geometry,self.extra_steps)

                        lnd0aa_MDD[lnd0aa_MDD==-np.inf] = 0
                        lnd0aa_MDD[lnd0aa_MDD==np.inf] = 0
                        lnd0aa_MDD[torch.isnan(lnd0aa_MDD)] = 0

                        misfit = torch.sum((1-self.omitValueIndices_lnd0aa)*((lnd0aa_MDD-self.lnd0aa)**2))
                    elif self.stat.lower() == "lnd0aa_chisq":
                        lnd0aa_MDD = calc_lnd0aa(Fi_MDD,self.tsec,self.geometry,self.extra_steps)
                        lnd0aa_MDD[lnd0aa_MDD==-np.inf] = 0
                        lnd0aa_MDD[lnd0aa_MDD==np.inf] = 0
                        lnd0aa_MDD[torch.isnan(lnd0aa_MDD)] = 0
                        misfit = (1-self.omitValueIndices_lnd0aa)*(((torch.exp(lnd0aa_MDD)-torch.exp(self.lnd0aa))**2)/self.Daa_uncertainty)
                        nan_rows = (torch.isnan(misfit)) | (misfit == np.inf) | (misfit == -np.inf)
                        misfit = torch.sum(misfit[~nan_rows],axis=0)



                        
                                            



                else:
                    trueFracMDD = Fi_MDD[1:]-Fi_MDD[0:-1]
                    trueFracMDD = torch.concat((torch.unsqueeze(Fi_MDD[0],dim=0),trueFracMDD),dim=0)

                    if self.stat.lower() == "l1_frac" or self.stat.lower() == "l2_frac" or self.stat.lower() == "percent_frac" or self.stat == "lnd0aa" or self.stat == "lnd0aa_chisq":
                        trueFracFi = self.Fi[1:] - self.Fi[0:-1]
                        trueFracFi = torch.concat((torch.unsqueeze(self.Fi[0],dim=-0),trueFracFi),dim=-1)
                        trueFracFi = torch.tile(trueFracFi.unsqueeze(1),[1,trueFracMDD.shape[1]])
                        if self.stat == "percent_frac":
                            trueFracFi[trueFracFi==0] = torch.min(trueFracFi[trueFracFi!=0])*0.1
                    
                        


                    elif self.stat.lower() == "l1_frac_cum":
                        Fi = torch.tile(self.Fi.unsqueeze(1),[1,Fi_MDD.shape[1]])
                    else:
                        moles_MDD = trueFracMDD * total_moles


                    if self.stat.lower()== "lnd0aa" or self.stat.lower() == "lnd0aa_chisq":
                        multiplier = 1- torch.tile(self.omitValueIndices_lnd0aa.unsqueeze(1),[1,trueFracMDD.shape[1]])
                    elif self.stat.lower() == "chisq":
                        multiplier = 1-torch.tile(self.omitValueIndices_chisq.unsqueeze(1),[1, trueFracMDD.shape[1]])
                    else:
                        multiplier = 1- torch.tile(self.omitValueIndices.unsqueeze(1),[1,trueFracMDD.shape[1]])
                        
                    
                    
                    if self.stat.lower() == "chisq":
                        misfit = torch.sum(multiplier*((exp_moles.unsqueeze(1)-moles_MDD)**2)/(data.uncert.unsqueeze(1)**2),axis=0)
                    elif self.stat.lower() == "l1_moles":
                        misfit = misfit = torch.sum(multiplier*(torch.abs(exp_moles.unsqueeze(1)-moles_MDD)),axis=0)
                    elif self.stat.lower() == "l2_moles":
                        misfit = torch.sum((multiplier*((exp_moles.unsqueeze(1)-moles_MDD)**2)),axis=0)
                    elif self.stat.lower() == "l1_frac":
                        misfit = torch.sum(multiplier*(torch.abs(trueFracFi-trueFracMDD)),axis=0)
                    elif self.stat.lower() == "l1_frac_cum":
                        misfit = torch.sum(multiplier*(torch.abs(Fi-Fi_MDD)),axis=0)
                    elif self.stat.lower() == "l2_frac":
                        misfit = torch.sum((multiplier*(trueFracFi-trueFracMDD)**2),axis=0)
                    elif self.stat.lower() == "percent_frac":
            

                        misfit = torch.sum(multiplier*(torch.abs(trueFracFi-trueFracMDD))/trueFracFi,axis=0)
                    elif self.stat.lower() == "lnd0aa":

                        lnd0aa_MDD = calc_lnd0aa(Fi_MDD,self.tsec,self.geometry,self.extra_steps)
                        lnd0aa_MDD[lnd0aa_MDD==-np.inf] = 0
                        lnd0aa_MDD[lnd0aa_MDD==np.inf] = 0
                        lnd0aa_MDD[torch.isnan(lnd0aa_MDD)] = 0

                        misfit = torch.sum(multiplier*((lnd0aa_MDD-self.lnd0aa.unsqueeze(1))**2),axis=0)
 
                    elif self.stat.lower() == "lnd0aa_chisq":
                        lnd0aa_MDD = calc_lnd0aa(Fi_MDD,self.tsec,self.geometry,self.extra_steps)
                        lnd0aa_MDD[lnd0aa_MDD==-np.inf] = 0
                        lnd0aa_MDD[lnd0aa_MDD==np.inf] = 0
                        lnd0aa_MDD[torch.isnan(lnd0aa_MDD)] = 0

                        misfit = multiplier*((torch.exp(lnd0aa_MDD)-torch.exp(self.lnd0aa.unsqueeze(1)))**2/self.Daa_uncertainty.unsqueeze(1))
                        nan_rows = torch.isnan(misfit).any(dim=1)
                        misfit = torch.sum(misfit[~nan_rows],axis=0)



                if torch.sum(misfit<0) >0:
                    breakpoint()

                return misfit*punishmentFlag
                

            trueFracMDD = Fi_MDD[1:]-Fi_MDD[0:-1]
            trueFracMDD = torch.concat((torch.unsqueeze(Fi_MDD[0],dim=-0),trueFracMDD),dim=-1)
    
            if self.stat == "l1_frac" or self.stat == "l2_frac" or self.stat == "percent_frac" or self.stat == "lnd0aa" or self.stat == "lnd0aa_chisq":
                trueFracFi = self.Fi[1:] - self.Fi[0:-1]
                trueFracFi = torch.concat((torch.unsqueeze(self.Fi[0],dim=-0),trueFracFi),dim=-1)
                if self.stat == "percent_frac":
                    trueFracFi[trueFracFi==0] = torch.min(trueFracFi[trueFracFi!=0])*0.1

            elif self.stat.lower() == "l1_frac_cum":
                Fi = torch.tile(self.Fi.unsqueeze(1),[1,Fi_MDD.shape[0]])
            else:
                moles_MDD = trueFracMDD * total_moles



            if self.stat.lower() == "chisq":             
                misfit = torch.sum(((1-self.omitValueIndices_chisq)*(exp_moles-moles_MDD)**2)/(data.uncert**2))
            elif self.stat.lower() == "l1_moles":
                misfit = torch.sum((1-self.omitValueIndices)*torch.abs(exp_moles-moles_MDD))
            elif self.stat.lower() == "l2_moles":
                misfit = torch.sum((1-self.omitValueIndices)*((exp_moles-moles_MDD)**2))
            elif self.stat.lower() == "l1_frac":
                misfit = torch.sum((1-self.omitValueIndices)*torch.abs(trueFracFi-trueFracMDD))
            elif self.stat.lower() == "l2_frac":
                misfit = torch.sum((1-self.omitValueIndices)*(trueFracFi-trueFracMDD)**2)
            elif self.stat.lower() == "l1_frac_cum":
               misfit = torch.sum((1-self.omitValueIndices)*torch.abs((Fi-Fi_MDD)))
            elif self.stat.lower() == "percent_frac":

                misfit = torch.sum((1-self.omitValueIndices)*(torch.abs(trueFracFi-trueFracMDD))/trueFracFi)
            elif self.stat.lower() == "lnd0aa":
                lnd0aa_MDD = calc_lnd0aa(Fi_MDD,self.tsec,self.geometry,self.extra_steps)
                lnd0aa_MDD[lnd0aa_MDD==-np.inf] = 0
                lnd0aa_MDD[lnd0aa_MDD==np.inf] = 0
                lnd0aa_MDD[torch.isnan(lnd0aa_MDD)] = 0
                misfit = torch.sum((1-self.omitValueIndices_lnd0aa)*((lnd0aa_MDD-self.lnd0aa)**2))
            elif self.stat.lower() == "lnd0aa_chisq":
                lnd0aa_MDD = calc_lnd0aa(Fi_MDD,self.tsec,self.geometry,self.extra_steps)
                lnd0aa_MDD[lnd0aa_MDD==-np.inf] = 0
                lnd0aa_MDD[lnd0aa_MDD==-np.inf] = 0
                lnd0aa_MDD[torch.isnan(lnd0aa_MDD)] = 0
                misfit = (1-self.omitValueIndices_lnd0aa)*(((torch.exp(lnd0aa_MDD)-torch.exp(self.lnd0aa))**2)/self.Daa_uncertainty)
                nan_rows = (torch.isnan(misfit)) | (misfit == np.inf) | (misfit == -np.inf)
                misfit = torch.sum(misfit[~nan_rows],axis=0)

 
            
            if torch.sum(misfit<0) >0:
                breakpoint()
            

            return misfit*punishmentFlag
