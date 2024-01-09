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


    
    # fig, axes = plt.subplots(ncols=3, nrows=1, layout="constrained", figsize=(10, 10))


    # Plot figure with subplots of different sizes
    fig = plt.figure(1)
    # set up subplot grid
    gridspec.GridSpec(2,4)


    plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2)
    plt.tight_layout
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

    # Put into a form that's easy to reference for the indices included in fit
    errors_for_plot_included = np.array(
        pd.concat(
            [
                dataset["ln(D/a^2)-del"][included],
                dataset["ln(D/a^2)+del"][included],
            ],
            axis=1,
        ).T
    )
    # Put into a form that's easy to reference for the indices NOT included in fit
    errors_for_plot_not_omitted = np.array(
        pd.concat(
            [
                dataset["ln(D/a^2)-del"][omitted[0:-1]],
                dataset["ln(D/a^2)+del"][omitted[0:-1]],
            ],
            axis=1,
        ).T
    )

    # Plot the MDD Model lndaa values that were included
    plt.plot(
        T_plot[included],
        pd.Series(data[1][included].tolist())
        .replace(-np.inf, np.inf)
        .fillna(max(data[1]).item()),
         "o", 
         markersize=5, 
         color='black', 
         linewidth=1, 
         mec='black',
         zorder = 2
    )

    # Plot the MDD Model lndaa values that were omitted
    plt.plot(
        T_plot[omitted[0:-1]],
        pd.Series(data[1][omitted[0:-1]].tolist())
        .replace(-np.inf, np.inf)
        .fillna(max(data[1]).item()),
         "o", 
         markersize=5, 
         color='black', 
         linewidth=1, 
         mec='black',
         zorder = 2,
         alpha = 0.3
    )

    # Plot the experimental lndaa values that were included
    plt.errorbar(
        T_plot[included],
        dataset["ln(D/a^2)"].replace(-np.inf, 0).loc[included],
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
        T_plot[omitted[0:-1]],
        dataset["ln(D/a^2)"].replace(-np.inf, 0).loc[omitted[0:-1]],
        yerr=errors_for_plot_not_omitted,
        fmt = 'o', 
        markersize=12, 
        color= (0.69,0.69,0.69),
        linewidth=1,
         mec='black', 
        zorder = 1,
        alpha = 0.3
    )

    # Label axes
    plt.ylabel("ln(D/a$^2$)")
    plt.xlabel("10000/T (K)")
    #plt.box_aspect(1)
    # plt].set_ylim(-30,0)


    
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

    plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=1)
    plt.tight_layout
    # Plot T_plot vs the gas fraction observed at each step for values that were included
    plt.errorbar(
        range(1, len(T_plot[included])+1),
        Fi[included],
        fmt='-o', 
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
        fmt='-o', 
        markersize=12, 
        mfc= (0.69,0.69,0.69), 
        mec='black', 
        zorder = 5,
        linewidth = 1,
        color = 'k',
        alpha = 0.3
    )

    indices_to_change = omitted

    
    #drew is still trying to figure out how to connect segments 
    

        
   
    # axes[2].plot(T_plot[indices_to_change[-1]:], Fi[indices_to_change[-1]:], 'ro-',zorder = 1, alpha=1.0)
    
    # Plot T_plot vs the modeled gas fraction observed at each step that was included
    plt.plot(included+1, 
                    Fi_MDD[included], 
                    "-o", 
                    markersize=5.25, 
                    color='black', 
                    linewidth=1, 
                    mec='black',
                    zorder = 10
                    )


    # Plot T_plot vs the modeled gas fraction observed at each step that was omitted
    plt.plot(omitted+1, 
                    Fi_MDD[omitted], 
                    "-o", 
                    markersize=5.25, 
                    color='black', 
                    linewidth=1, 
                    mec='black',
                    zorder = 10,
                    alpha = 0.3
                    )

    for i in range(len(indices_to_change) - 1):
        #If index is not zero
        #If index is the last in the last in the series
        if indices_to_change[i]+1 == len(Fi):
            start = indices_to_change[i]-1
            end = indices_to_change[i]
            plt.plot(range(start+1,end+1), Fi[start:end+1], 'k-', alpha=.1)  # Connect the segments
            plt.plot(range(start+1,end+1), Fi_MDD[start:end+1],'k-',alpha=0.1)
        elif indices_to_change[i] == 0:
            start = indices_to_change[i]
            end = indices_to_change[i]+1
            plt.plot(range(start+1,end+1), Fi[start:end+1], 'k-', alpha=.1)  # Connect the segments
            plt.plot(range(start+1,end+1), Fi_MDD[start:end+1],'k-',alpha=0.1)
        else:
            start = indices_to_change[i]-1
            end = indices_to_change[i]+1
            plt.plot(range(start+1,end+2), Fi[start:end+1], 'k-', alpha=.1)  # Connect the segments
            plt.plot(range(start+1,end+2), Fi_MDD[start:end+1],'k-',alpha=0.1)
        print(indices_to_change[i])

    plt.xlabel("step number")
    plt.ylabel("Fractional Release (%)")
#    plt.box_aspect(1)

    # If moles were calculated, make the same plot but with moles
    # if moles_calc == True:


    #     # Plot the moles measured in experiment at each step
    #     axes[1, 1].errorbar(
    #         range(0, len(T_plot)),
    #         dataset["M"],
    #         yerr=dataset["delM"],
    #         fmt='-o', 
    #         markersize=12, 
    #         mfc= (0.69,0.69,0.69), 
    #         mec='black', 
    #         alpha = 0.8,
    #         zorder = 5,
    #         linewidth = 1,
    #         color = 'k'

    #     )
    #     axes[1, 1].plot(
    #         range(0, len(T_plot)), 
    #         tot_moles * Fi_MDD, 
    #         "-o", 
    #         markersize=5.25, 
    #         color='black', 
    #         linewidth=1, 
    #         mec='black',
    #         zorder = 10
    #     )
    #     axes[1, 1].xlabel("step number")
    #     axes[1, 1].ylabel("Atoms Released at Each Step")
    #     axes[1, 1].set_box_aspect(1)
    #     # plt.axis('square')

   
    # Calculate reference law results

    # Slope
    m = params[0]/83.14 #Activation energy (kJ/mol) / gas constant

    resid_exp = dataset["ln(D/a^2)"][0:-1] - (-m.item() * T_plot[0:-1] + params[1].item())

    resid_model = np.array(lnd0aa_MDD.ravel()) - (-m.item() *T_plot[0:-1] + params[1].item())

    plt.subplot2grid((2,4), (1,2), colspan=2, rowspan=1)
    plt.tight_layout
    plt.plot(data[0][0:-1] * 100, 
                    resid_exp, 'o', markersize=12, 
                    color= (0.69,0.69,0.69), 
                    mec='black', 
                    alpha = 0.8
                    )
    

    plt.plot(data[0][0:-1] * 100, resid_model, 
                    "-o", markersize=5, 
                    color='black',  
                    linewidth=1, 
                    mec='black'
                    )
    
    plt.xlabel("Cumulative 3He Release (%)")
    plt.ylabel("Residual ln(1/s)")
   # plt.box_aspect(1)

    fig.tight_layout()
    fig.set_size_inches(w=10,h=7)
    plt.savefig(plot_path)
    
    if quiet == False:
        plt.show()
