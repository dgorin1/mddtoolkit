import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
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
    quiet=True,
):
    """Plot the results of the optimization.

    Args:
        - params (torch.tensor): The parameters from the optimization.
        - dataset (Dataset): The dataset for the optimization.
        - objective (DiffusionObjective): The objective function for the optimization.
        - output_path (str): The path to save the plot to.
        - quiet (bool, optional): Whether to show the plot. Defaults to False.
    """
    # Params is a vector X of the input parameters
    # dataset is the dataset class with your data
    # objective is the objective you used

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
                        Fi_MDD, objective.tsec, objective.geometry, objective.extra_steps, objective.added_steps
                 )

    # Ensure that values aren't infinity in Fi for plotting purposes
    mask = torch.isinf(Fi_MDD)
    Fi_MDD[mask] = float("nan")
    Fi_MDD = np.array(Fi_MDD.ravel())

    # Ensure that values aren't infinity in lnd0aa_MDD for plotting purposes
    mask = torch.isinf(lnd0aa_MDD)
    lnd0aa_MDD[mask] = float("nan")
    lnd0aa_MDD = np.array(lnd0aa_MDD.ravel())
    T_plot = 10000 / (dataset["TC"] + 273.15)

    # Ensure that values from the experimental data aren't infinity also
    dataset["ln(D/a^2)"].replace([np.inf,-np.inf],np.nan, inplace = True)
    dataset["ln(D/a^2)-del"].replace([np.inf, -np.inf], np.nan, inplace = True)
    dataset["ln(D/a^2)+del"].replace([np.inf, -np.inf], np.nan, inplace = True)
    dataset["Fi"].replace([np.inf, -np.inf], np.nan, inplace = True)


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


    
    # fig, axes = plt.subplots(ncols=3, nrows=1, layout="constrained", figsize=(10, 10))


    # Plot figure with subplots of different sizes
    fig = plt.figure(1)
    # set up subplot grid
    gridspec.GridSpec(2,4)
    


    ax = plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2)
    ax.set_box_aspect(1)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.15)
    
    for i in range(ndom):
        # Calculate a line representing each domain for visualization in the plot
        D = params[i+1]-params[0]/R*(1/(TC[objective.added_steps:-1]+273.15))
        # Plot each line
        plt.plot(
            np.linspace(min(10000/(TC[objective.added_steps:-1]+273.15)), max(10000/(TC[objective.added_steps:-1]+273.15)), 1000),
            np.linspace(max(D), min(D), 1000),
            "--",
            linewidth=frac_weights[i],
            zorder=0,
            color = (0.6,0,0),
            alpha = 0.5
        )
    
    #Perform type conversions and grab appropriate indices for plotting
    included = np.array(((1-objective.omitValueIndices) == 1).nonzero().squeeze())
    omitted = np.array((objective.omitValueIndices == 1).nonzero().squeeze())

    # Put into a form that's easy to reference for the indices included in fit and also remove infs
    errors_for_plot_included = np.array(
        pd.concat(
            [
                dataset["ln(D/a^2)-del"][included],
                dataset["ln(D/a^2)+del"][included],
            ],
            axis=1,
        ).T
    )
    # Put into a form that's easy to reference for the indices NOT included in fit and also remove infs
    errors_for_plot_not_omitted = np.array(
        pd.concat(
            [
                dataset["ln(D/a^2)-del"][omitted],
                dataset["ln(D/a^2)+del"][omitted],
            ],
            axis=1,
        ).T
    )

    # Plot the MDD Model lndaa values that were included
    plt.plot(
        T_plot[included],
        pd.Series(lnd0aa_MDD[included].tolist()),
         "o", 
         markersize=5, 
         color='black', 
         linewidth=1, 
         mec='black',
         zorder = 2
    )

    # Plot the MDD Model lndaa values that were omitted
    plt.plot(
        T_plot[omitted],
        pd.Series(lnd0aa_MDD[omitted].tolist()),
         "o", 
         markersize=5, 
         color='black', 
         linewidth=1, 
         mec='black',
         zorder = 2,
         alpha = 0.4
    )

    # Plot the experimental lndaa values that were included
    plt.errorbar(
        T_plot[included],
        dataset["ln(D/a^2)"].loc[included],
        yerr=errors_for_plot_included,
        fmt = 'o', 
        markersize=12, 
        color= (0.69,0.69,0.69),
        linewidth=1,
         mec='black', 
        zorder = 1
    )

    # Plot the experimental lndaa values that were omitted
    plt.errorbar(
        T_plot[omitted],
        dataset["ln(D/a^2)"].loc[omitted],
        yerr=errors_for_plot_not_omitted,
        fmt = 'o', 
        markersize=12, 
        color= (0.69,0.69,0.69),
        linewidth=1,
         mec='black', 
        zorder = 1,
        alpha = 0.4
    )

    # Label axes
    plt.ylabel("ln(D/a$^2$)",fontsize = 15.5)
    plt.xlabel("10000/T (K)",fontsize = 15.5)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    
    
    # plt].set_ylim(-30,0)


    
    # Make a plot of the gas fractions 


   
    # Put Fi_MDD in non-cumulative space
    temp = Fi_MDD[1:] - Fi_MDD[0:-1]
    Fi_MDD = np.insert(temp, 0, Fi_MDD[0])

    # Get gas fractions from actual experiment and put in non-cumulative space
    Fi = np.array(dataset["Fi"])
    temp = Fi[1:] - Fi[0:-1]
    Fi = np.insert(temp, 0, Fi[0])





    #     # Plot T_plot vs the gas fraction observed at each step for values that were included
    # axes[2].errorbar(
    #     range(0, len(T_plot)),
    #     Fi,
    #     fmt='-o', 
    #     markersize=12, 
    #     mfc= (0.69,0.69,0.69), 
    #     mec='black', 
    #     zorder = 5,
    #     linewidth = 1,
    #     color = 'k'
    # )

    ax = plt.subplot2grid((2,4), (1,2), colspan=2, rowspan=1)
    ax.set_box_aspect(1)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.15)


    # Plot T_plot vs the gas fraction observed at each step for values that were included
    plt.errorbar(
        included+1,
        Fi[included],
        fmt='o', 
        markersize=12, 
        mfc= (0.69,0.69,0.69), 
        mec='black', 
        zorder = 5,
        linewidth = 1,
        color = 'k'
    )

    # Plot T_plot vs the gas fraction observed at each step for values that were omitted
    plt.errorbar(
        omitted+1,
        Fi[omitted],
        fmt='o', 
        markersize=12, 
        mfc= (0.69,0.69,0.69), 
        mec='black', 
        zorder = 5,
        linewidth = 1,
        color = 'k',
        alpha = 0.3
    )

    # Plot T_plot vs the modeled gas fraction observed at each step that was included
    plt.plot(included+1, 
                    Fi_MDD[included], 
                    "o", 
                    markersize=5.25, 
                    color='black', 
                    linewidth=1, 
                    mec='black',
                    zorder = 10
                    )


    # Plot T_plot vs the modeled gas fraction observed at each step that was omitted
    plt.plot(omitted+1, 
                    Fi_MDD[omitted], 
                    "o", 
                    markersize=5.25, 
                    color='black', 
                    linewidth=1, 
                    mec='black',
                    zorder = 10,
                    alpha = 0.55
                    )

    for i in range(len(Fi_MDD)-1):
            if i in omitted or i+1 in omitted:
                alpha_val = .45
            else:
                alpha_val = 1.
            plt.plot(range(i+1,i+3),
                     Fi_MDD[i:i+2],
                     'k-',
                     alpha = alpha_val,
                     zorder = 10
                     )
            plt.plot(range(i+1,i+3),
                     Fi[i:i+2],
                     '--',
                     color = (0.69,0.69,0.69),
                     alpha = alpha_val,
                     zorder = 1
                     )

    plt.xlabel("step number",fontsize = 12)
    plt.ylabel("Fractional Release (%)", fontsize = 12)
   


    # Slope
    m = params[0]/83.14 #Activation energy (kJ/mol) / gas constant
    resid_exp = dataset["ln(D/a^2)"] - (-m.item() * T_plot + params[1].item())
    resid_model = np.array(lnd0aa_MDD.ravel()) - (-m.item() *T_plot + params[1].item())

    ax = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=1)
    ax.set_box_aspect(1)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.15)
    
    #Plot the values of residuals that were included in the fit calculated against the experimental results
    cum_Fi_MDD = np.cumsum(Fi_MDD)
    cum_Fi_exp = np.cumsum(Fi)
    plt.plot(cum_Fi_exp[included] * 100, 
                    resid_exp[included], 'o', markersize=12, 
                    color= (0.69,0.69,0.69), 
                    mec='black', 
                    alpha = 0.8
                    )
    
    #Plot the values of residuals that were excluded in the fit calculated against the experimental results
    plt.plot(cum_Fi_exp[omitted] * 100, 
                    resid_exp[omitted], 'o', markersize=12, 
                    color= (0.69,0.69,0.69), 
                    mec='black', 
                    alpha = 0.3
                    )
    
    plt.plot(cum_Fi_MDD[included] * 100, 
             resid_model[included], 
                    "o", markersize=5, 
                    color='black',  
                    linewidth=1, 
                    mec='black'
                    )
    plt.plot(cum_Fi_MDD[omitted] * 100, 
             resid_model[omitted], 
                "o", markersize=5, 
                color='black',  
                linewidth=1, 
                mec='black',
                alpha = 0.3
                )

    for i in range(len(cum_Fi_MDD)-1):
            if i in omitted or i+1 in omitted:
                alpha_val = .45
            else:
                alpha_val = 1.
            plt.plot(cum_Fi_MDD[i:i+2]*100,
                     resid_model[i:i+2],
                     'k-',
                     alpha = alpha_val,
                     zorder = 250
                     )
            plt.plot(cum_Fi_exp[i:i+2]*100,
                     resid_exp[i:i+2],
                     '--',
                     color= (0.69,0.69,0.69),
                     alpha = alpha_val,
                     zorder = 1
                     )

    breakpoint()

    # #Connect the line segments that were bisected by the omitting of values from the fit
    # for i in range(len(omitted)):
    #     #If index is not zero
    #     #If index is the last in the last in the series
    #     if omitted[i]+1 == len(resid_model):
    #         start = omitted[i-1]
    #         plt.plot(Fi_MDD[start:], resid_model[start:], 'k-', alpha=.1)  # Connect the segments

    #     elif omitted[i] == 0:
    #         plt.plot(Fi_MDD[0:2], resid_model[0:2], 'k-', alpha=.1)  # Connect the segments

    #     else:
    #         index = omitted[i]
    #         print(Fi_MDD[index-1:index+2])
    #         plt.plot(Fi_MDD[index-1:index+2], resid_model[index-1:index+2], 'k-', alpha=.1)  # Connect the segments






    plt.xlabel("Cumulative $^3$He Release (%)",fontsize = 11)
    plt.ylabel("Residual ln(1/s)",fontsize = 11)
    fig.tight_layout()
    fig.set_size_inches(w=15,h=7)
    plt.savefig(plot_path)
    
    if quiet == False:
        plt.show()
