import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from utils.get_plot_name import get_plot_name
from diffusion_kinetics.optimization import (
    forwardModelKinetics, 
    calc_lnd0aa
)
import os

def plot_results(
    params,
    dataset,
    objective,
    plot_path:str,
    reference_law=[],
    quiet=True,
):
    """Plot the results of the optimization.

    Args:
        - params (torch.tensor): The parameters from the optimization.
        - dataset (Dataset): The dataset for the optimization.
        - objective (DiffusionObjective): The objective function for the optimization.
        - output_path (str): The path to save the plot to.
        - reference_law (list, optional): The reference law for the optimization. Defaults to [].
        - quiet (bool, optional): Whether to show the plot. Defaults to False.
    """
    # Params is a vector X of the input parameters
    # dataset is the dataset class with your data
    # objective is the objective you used
    # reference_law is an array with values [Ea, lnd0aa]
    R = 0.008314
    params = torch.tensor(params)
    if len(params) % 2 != 0:
        tot_moles = params[0]
        params = params[1:]
        moles_calc = True
    else:
        moles_calc = False

    # Infer the number of domains from input
    if len(params) <= 3:
        ndom = 1
    else:
        ndom = (len(params)) // 2

    # Reconstruct the time-added and temp-added inputs
    time = torch.tensor(dataset.thr * 3600)
    TC = torch.tensor(dataset.TC)
    
    # If adding extra input steps to ignore..
    if objective.extra_steps == True:
        tsec = torch.cat([torch.tensor(np.array(objective.time_add)), time])
        TC = torch.cat([torch.tensor(np.array(objective.temp_add)), TC])
    else:
        tsec = objective.tsec

    # Calculate the cumulative fractions from the MDD model
    Fi_MDD, punishmentFlag = forwardModelKinetics(params,
                                     tsec, 
                                     TC, 
                                     geometry = objective.geometry,
                                     added_steps=objective.added_steps)

    # Calculate the lndaa from the mdd model
    lnd0aa_MDD = calc_lnd0aa(
                        Fi_MDD[0:-1], objective.tsec[0:-1], objective.geometry, objective.extra_steps, objective.added_steps
                 )
    
    data = (Fi_MDD.ravel(),lnd0aa_MDD.ravel())

    T_plot = 10000 / (dataset["TC"] + 273.15)


    n_plots = 4



    # Calculate weights proportional to the gas fractions if numdom > 1
    if ndom > 1:
        fracs = params[ndom + 1 :]
        fracs = torch.concat(
            (fracs, 1 - torch.sum(fracs, axis=0, keepdim=True)), axis=-1
        )
        if ndom <= 6:
            frac_weights = fracs *(ndom+5) 
        else:
            frac_weights = fracs*(ndom+10)
        frac_weights[frac_weights < 0.4] = 0.4
    else:
        fracs = 1
        frac_weights = [2]


    
    fig, axes = plt.subplots(ncols=2, nrows=2, layout="constrained", figsize=(10, 10))


    # This is going to cause and error and Drew needs to fix
    errors_for_plot = np.array(
        pd.concat(
            [
                dataset["ln(D/a^2)-del"],
                dataset["ln(D/a^2)+del"],
            ],
            axis=1,
        ).T
    )



    for i in range(ndom):
        # Calculate a line representing each domain
        D = params[i+1]-params[0]/R*(1/(TC[objective.added_steps:-1]+273.15))
        # Plot each line
        axes[0, 0].plot(
            np.linspace(min(10000/(TC[objective.added_steps:-1]+273.15)), max(10000/(TC[objective.added_steps:-1]+273.15)), 1000),
            np.linspace(max(D), min(D), 1000),
            "--",
            linewidth=frac_weights[i],
            zorder=0,
            color = (0.6,0,0),
            alpha = 0.5
        )
    

    # Plot the MDD Model lndaa values
    axes[0, 0].plot(
        T_plot[0:-1],
        pd.Series(data[1].tolist())
        .replace(-np.inf, np.inf)
        .fillna(max(data[1]).item()),
         "o", 
         markersize=5, 
         color='black', 
         linewidth=1, 
         mec='black',
         zorder = 2
    )

    # Plot the experimental lndaa values

    axes[0, 0].errorbar(
        T_plot,
        dataset["ln(D/a^2)"].replace(-np.inf, 0),
        yerr=errors_for_plot,
        fmt = 'o', 
        markersize=12, 
        color= (0.69,0.69,0.69),
        linewidth=1,
         mec='black', 
        #  alpha = 0.8
        zorder = 1
    )
    

    # Label axes
    axes[0, 0].set_ylabel("ln(D/a$^2$)")
    axes[0, 0].set_xlabel("10000/T (K)")
    axes[0,0].set_ylim(-30,0)
    axes[0, 0].set_box_aspect(1)


    
    # Make a plot of the gas fractions 

    # Set Fi_MDD to a variable
    Fi_MDD = np.array(data[0])
   
    # Put Fi_MDD in non-cumulative space
    temp = Fi_MDD[1:] - Fi_MDD[0:-1]
    Fi_MDD = np.insert(temp, 0, Fi_MDD[0])

    # Get gas fractions from actual experiment and put in non-cumulative space
    Fi = np.array(dataset.Fi)
    temp = Fi[1:] - Fi[0:-1]
    Fi = np.insert(temp, 0, Fi[0])

    # Plot T_plot vs the gas fraction observed at each step
    axes[1, 0].errorbar(
        range(0, len(T_plot)),
        Fi,
        fmt='-o', 
        markersize=12, 
        mfc= (0.69,0.69,0.69), 
        mec='black', 
        # alpha = 0.8,
        zorder = 5,
        linewidth = 1,
        color = 'k'
    )

    # Plot T_plot vs the modeled gas fraction observed at each step
    axes[1, 0].plot(range(0, len(T_plot)), 
                    Fi_MDD, 
                    "-o", 
                    markersize=5.25, 
                    color='black', 
                    linewidth=1, 
                    mec='black',
                    zorder = 10
                    )




    axes[1, 0].set_xlabel("step number")
    axes[1, 0].set_ylabel("Fractional Release (%)")
    # axes[1].axis('square')
    axes[1, 0].set_box_aspect(1)

    # If moles were calculated, make the same plot but with moles
    if moles_calc == True:


        # Plot the moles measured in experiment at each step
        axes[1, 1].errorbar(
            range(0, len(T_plot)),
            dataset["M"],
            yerr=dataset["delM"],
            fmt='-o', 
            markersize=12, 
            mfc= (0.69,0.69,0.69), 
            mec='black', 
            alpha = 0.8,
            zorder = 5,
            linewidth = 1,
            color = 'k'

        )
        axes[1, 1].plot(
            range(0, len(T_plot)), 
            tot_moles * Fi_MDD, 
            "-o", 
            markersize=5.25, 
            color='black', 
            linewidth=1, 
            mec='black',
            zorder = 10
        )
        axes[1, 1].set_xlabel("step number")
        axes[1, 1].set_ylabel("Atoms Released at Each Step")
        axes[1, 1].set_box_aspect(1)
        # axes[2].axis('square')

    if n_plots == 4:
        # Calculate reference law results

        # Slope
        m = params[0]/83.14 #Activation energy (kJ/mol) / gas constant
  
        resid_exp = dataset["ln(D/a^2)"][0:-1] - (-m.item() * T_plot[0:-1] + params[1].item())

        resid_model = np.array(lnd0aa_MDD.ravel()) - (-m.item() *T_plot[0:-1] + params[1].item())

        axes[0, 1].plot(data[0][0:-1] * 100, 
                        resid_exp, 'o', markersize=12, 
                        color= (0.69,0.69,0.69), 
                        mec='black', 
                        alpha = 0.8
                        )
        

        axes[0, 1].plot(data[0][0:-1] * 100, resid_model, 
                        "-o", markersize=5, 
                        color='black',  
                        linewidth=1, 
                        mec='black'
                        )
        
        axes[0, 1].set_xlabel("Cumulative 3He Release (%)")
        axes[0, 1].set_ylabel("Residual ln(1/s)")
        axes[0, 1].set_box_aspect(1)

    plt.tight_layout
    plt.savefig(plot_path)
    
    if quiet == False:
        plt.show()
