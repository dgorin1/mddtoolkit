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
):
    """Plot the results of the optimization.

    Args:
        - params (torch.tensor): The parameters from the optimization.
        - dataset (Dataset): The dataset for the optimization.
        - objective (DiffusionObjective): The objective function for the optimization.
        - output_path (str): The path to save the plot to.
    """
    # Params is a vector X of the input parameters
    # dataset is the dataset class with your data
    # objective is the objective you used

    R = 0.008314
    params = torch.tensor(params)

    # Remove the moles parameter if not needed for plotting
    if len(params) % 2 != 0:
        params = params[1:]

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

    # Ensure that values from the experimental data aren't infinity also
    dataset["ln(D/a^2)"].replace([np.inf,-np.inf],np.nan, inplace = True)
    dataset["ln(D/a^2)-del"].replace([np.inf, -np.inf], np.nan, inplace = True)
    dataset["ln(D/a^2)+del"].replace([np.inf, -np.inf], np.nan, inplace = True)
    dataset["Fi"].replace([np.inf, -np.inf], np.nan, inplace = True)

    # Recast T_plot T as 10000/T
    T_plot = 10000 / (dataset["TC"] + 273.15)


    # Calculate weights proportional to the gas fractions if numdom > 1 to be used in drawing line thicknesses represenging
    # the domains
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

    # Create a figure with subplots of differing sizes and create a subplot grid
    fig = plt.figure(1)
    gridspec.GridSpec(2,4)
    
    # Begin the first and major plot, the arrhenius plot
    ax1 = plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.15)
    
     # Calculate and plot a line representing each domain for visualization in the plot
    for i in range(ndom):
        D = params[i+1]-params[0]/R*(1/(TC[objective.added_steps:-1]+273.15))
        plt.plot(
            np.linspace(min(10000/(TC[objective.added_steps:-1]+273.15)), max(10000/(TC[objective.added_steps:-1]+273.15)), 1000),
            np.linspace(max(D), min(D), 1000),
            "--",
            linewidth=frac_weights[i],
            zorder=0,
            color = (0.6,0,0),
            alpha = 0.5
        )
    
    # Perform type conversions and grab appropriate indices for plotting so that excluded values can be plotted with different symbology
    included = np.array(((1-objective.omitValueIndices) == 1).nonzero().squeeze())
    omitted = np.array((objective.omitValueIndices == 1).nonzero().squeeze())

    # Avoid having a value close to inf plot and rescale the plot..
    if any(dataset["ln(D/a^2)"].isna()):
        indices = dataset["ln(D/a^2)"].isna()
        lnd0aa_MDD[indices] = np.nan
    # Put into the correct form to be plotted w/ errorbar function for values included
    errors_for_plot_included = np.array(
        pd.concat(
            [
                dataset["ln(D/a^2)-del"][included],
                dataset["ln(D/a^2)+del"][included],
            ],
            axis=1,
        ).T
    )

    if len(omitted.shape) == 0:
        errors_for_plot_not_omitted = [[dataset["ln(D/a^2)-del"].loc[omitted]], [dataset["ln(D/a^2)+del"].loc[omitted]]]
        omitted = [omitted]


    else:
    # Put into the correct form to be plotted w/ errorbar function for values excluded
        errors_for_plot_not_omitted = np.array(
            pd.concat(
                [
                    dataset[["ln(D/a^2)-del"]].loc[omitted],
                    dataset[["ln(D/a^2)+del"]].loc[omitted],
                ],
                axis=1
            ).T
        )
    # Plot the MDD Model ln(D/a^2) values that were included
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
  
    # Plot the MDD Model ln(D/a^2) values that were omitted
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

    # Plot the experimental ln(D/a^2) values that were included
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


    # Plot the experimental ln(D/a^2) values that were omitted
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

    # Label and format axes
    plt.ylabel("ln(D/a$^2$)",fontsize = 15.5)
    plt.xlabel("10000/T (K)",fontsize = 15.5)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    ax1.set_box_aspect(1)
    # plt.ylim(-30,0)


    # Create axes for plotting the gas fractions as a function of step #
    ax2 = plt.subplot2grid((2,4), (1,2), colspan=2, rowspan=1)
    ax2.set_box_aspect(1)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(1.15)

    # Put Fi_MDD in non-cumulative space
    temp = Fi_MDD[1:] - Fi_MDD[0:-1]
    Fi_MDD = np.insert(temp, 0, Fi_MDD[0])

    # Get gas fractions from laboratory experiment and put in non-cumulative space
    Fi = np.array(dataset["Fi"])
    temp = Fi[1:] - Fi[0:-1]
    Fi = np.insert(temp, 0, Fi[0])

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

    # This loop draws lines between the points and makes them transparent depending on whether 
    # or not each value was included in the optimization/fitting
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
    # Label axes
    plt.xlabel("Step Number",fontsize = 12)
    plt.ylabel("Fractional Release (%)", fontsize = 12)
   

    # Create space for the residual plot
    ax3 = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=1)
    
    # Calculate the residual using the highest-retentivity domain as a reference for both 
    # the model and experimental data
    m = params[0]/83.14 #Activation energy (kJ/mol) / gas constant
    resid_exp = dataset["ln(D/a^2)"] - (-m.item() * T_plot + params[1].item())
    resid_model = np.array(lnd0aa_MDD.ravel()) - (-m.item() *T_plot + params[1].item())

    # Plot the values of residuals that were included in the fit calculated against the experimental results
    cum_Fi_MDD = np.cumsum(Fi_MDD)
    cum_Fi_exp = np.cumsum(Fi)
    plt.plot(cum_Fi_exp[included] * 100, 
                    resid_exp[included], 'o', markersize=12, 
                    color= (0.69,0.69,0.69), 
                    mec='black', 
                    alpha = 0.8
                    )
    
    # Plot the values of residuals that were excluded in the fit calculated against the experimental results
    plt.plot(cum_Fi_exp[omitted] * 100, 
                    resid_exp[omitted], 'o', markersize=12, 
                    color= (0.69,0.69,0.69), 
                    mec='black', 
                    alpha = 0.3
                    )
    # Plot the values of residuals that were included in the fit calculated against the model results
    plt.plot(cum_Fi_MDD[included] * 100, 
             resid_model[included], 
                    "o", markersize=5, 
                    color='black',  
                    linewidth=1, 
                    mec='black'
                    )
    
    # Plot the values of residuals that were excluded in the fit calculated against the model results
    plt.plot(cum_Fi_MDD[omitted] * 100, 
             resid_model[omitted], 
                "o", markersize=5, 
                color='black',  
                linewidth=1, 
                mec='black',
                alpha = 0.3
                )
    
    # This loop draws lines between the points and makes them transparent depending on whether 
    # or not each value was included in the optimization/fitting
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

    # Format plot and label axes
    ax3.set_box_aspect(1)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax3.spines[axis].set_linewidth(1.15)
    plt.xlabel("Cumulative Gas Release (%)",fontsize = 11)
    plt.ylabel("Residual ln(1/s)",fontsize = 11)
    fig.tight_layout()
    fig.set_size_inches(w=15,h=7)

    # Save output
    plt.savefig(plot_path)
    plt.close(fig)
