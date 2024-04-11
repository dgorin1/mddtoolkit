import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
# from utils.get_plot_name import get_plot_name
from optimization.forwardModelKinetics import forwardModelKinetics

from optimization.forwardModelKinetics import calc_lnd0aa
import os

def plot_results_schematic(
    params,
    dataset,
    objective,
    params2,
    plot_path:str,
    params3 = []
    
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
    params2 = torch.tensor(params2)
    params3 = torch.tensor(params3)

    # Remove the moles parameter if not needed for plotting
    if len(params) % 2 != 0:
        params = params[1:]
        params2 = params2[1:]
        params3 = params3[1:]


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


    Fi_MDD2, punishmentFlag2 = forwardModelKinetics(params2,
                                                    tsec,
                                                    TC,
                                                    geometry = objective.geometry,
                                                    added_steps = objective.added_steps)
    
    Fi_MDD3, punishmentFlag2 = forwardModelKinetics(params3,
                                                tsec,
                                                TC,
                                                geometry = objective.geometry,
                                                added_steps = objective.added_steps)

    # Calculate the lndaa from the mdd model
    lnd0aa_MDD = calc_lnd0aa(
                        Fi_MDD, objective.tsec, objective.geometry, objective.extra_steps, objective.added_steps
                 )



    lnd0aa_MDD2 = calc_lnd0aa(
                        Fi_MDD2, objective.tsec, objective.geometry, objective.extra_steps, objective.added_steps
                 )

    lnd0aa_MDD3 = calc_lnd0aa(
                        Fi_MDD3, objective.tsec, objective.geometry, objective.extra_steps, objective.added_steps
                 )
    # Ensure that values aren't infinity in Fi for plotting purposes
    mask = torch.isinf(Fi_MDD)
    Fi_MDD[mask] = float("nan")
    Fi_MDD = np.array(Fi_MDD.ravel())


    mask = torch.isinf(Fi_MDD2)
    Fi_MDD2[mask] = float("nan")
    Fi_MDD2 = np.array(Fi_MDD2.ravel())



    mask = torch.isinf(Fi_MDD3)
    Fi_MDD3[mask] = float("nan")
    Fi_MDD3 = np.array(Fi_MDD3.ravel())

    # Ensure that values aren't infinity in lnd0aa_MDD for plotting purposes
    mask = torch.isinf(lnd0aa_MDD)
    lnd0aa_MDD[mask] = float("nan")
    lnd0aa_MDD = np.array(lnd0aa_MDD.ravel())

    mask = torch.isinf(lnd0aa_MDD2)
    lnd0aa_MDD2[mask] = float("nan")
    lnd0aa_MDD2 = np.array(lnd0aa_MDD2.ravel())

    mask = torch.isinf(lnd0aa_MDD3)
    lnd0aa_MDD3[mask] = float("nan")
    lnd0aa_MDD3 = np.array(lnd0aa_MDD3.ravel())

    # Ensure that values from the experimental data aren't infinity also
    dataset["ln(D/a^2)"].replace([np.inf,-np.inf],np.nan, inplace = True)
    dataset["ln(D/a^2)-del"].replace([np.inf, -np.inf], np.nan, inplace = True)
    dataset["ln(D/a^2)+del"].replace([np.inf, -np.inf], np.nan, inplace = True)
    dataset["Fi"].replace([np.inf, -np.inf], np.nan, inplace = True)

    # Recast T_plot T as 10000/T
    T_plot = 10000 / (dataset["TC"] + 273.15)


    # Calculate weights proportional to the gas fractions if numdom > 1 to be used in drawing line thicknesses represenging


    # Create a figure with subplots of differing sizes and create a subplot grid
    fig = plt.figure(figsize = (1,1))
    # gridspec.GridSpec(4,10)
    
    # Begin the first and major plot, the arrhenius plot
    #ax1 = plt.subplot2grid((4,11), (0,0), colspan=4 ,rowspan=4)
    ax1 = plt.subplot(1,1,1)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.15)
    
    # Perform type conversions and grab appropriate indices for plotting so that excluded values can be plotted with different symbology
    included = np.array(((1-objective.omitValueIndices) == 1).nonzero().squeeze())
    omitted = np.array((objective.omitValueIndices == 1).nonzero().squeeze())


    marker_size_measured = 15
    marker_size_modeled = marker_size_measured-7.5
    # Avoid having a value close to inf plot and rescale the plot..
    if any(dataset["ln(D/a^2)"].isna()):
        indices = dataset["ln(D/a^2)"].isna()
        lnd0aa_MDD[indices] = np.nan
        lnd0aa_MDD2[indices] = np.nan
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
    alpha1 = 1
    color_best_fit = "black"
    color_else =(4/255, 133/255, 209/255) #'red'
    color_med = (252/255,90/255,80/255)
    # Plot the MDD Model ln(D/a^2) values that were included
    plt.plot(
        T_plot[included],
        pd.Series(lnd0aa_MDD[included].tolist()),
         "o", 
         markersize=marker_size_modeled, 
         color=color_best_fit, 
         linewidth=1, 
         mec=(60/255,62/255,60/255),
         zorder = 2,
    )
    
    plt.plot(
        T_plot[included],
        pd.Series(lnd0aa_MDD2[included].tolist()),
        "o", 
        markersize=marker_size_modeled, 
        color=color_else, 
        linewidth=1, 
        mec='black',
        zorder = 2,
        alpha = alpha1
    )
    plt.plot(
        T_plot[included],
        pd.Series(lnd0aa_MDD3[included].tolist()),
        "o", 
        markersize=marker_size_modeled, 
        color=color_med, 
        linewidth=1, 
        mec='black',
        alpha = alpha1,
        zorder = 2
    )
  
    # Plot the MDD Model ln(D/a^2) values that were omitted
    plt.plot(
        T_plot[omitted],
        pd.Series(lnd0aa_MDD[omitted].tolist()),
         "o", 
         markersize=marker_size_modeled, 
         color=color_best_fit, 
         linewidth=1, 
         mec=color_best_fit,
         zorder = 2,
         alpha = 0.3
    )

        # Plot the MDD Model ln(D/a^2) values that were omitted
    plt.plot(
        T_plot[omitted],
        pd.Series(lnd0aa_MDD2[omitted].tolist()),
         "o", 
         markersize=marker_size_modeled, 
         color=color_else, 
         linewidth=1, 
         mec=color_else,
         zorder = 2,
         alpha = 0.3
    )

    plt.plot(
        T_plot[omitted],
        pd.Series(lnd0aa_MDD3[omitted].tolist()),
         "o", 
         markersize=marker_size_modeled, 
         color=color_med, 
         linewidth=1, 
         mec=color_med,
         zorder = 2,
         alpha = 0.3
    )


    # Plot the experimental ln(D/a^2) values that were included
    plt.errorbar(
        T_plot[included],
        dataset["ln(D/a^2)"].loc[included],
        yerr=errors_for_plot_included,
        fmt = 'o', 
        markersize=marker_size_measured, 
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
        markersize=marker_size_measured, 
        color= (0.69,0.69,0.69),
        linewidth=1,
         mec='black', 
        zorder = 1,
        alpha = 0.3
    )


    label_size = 17
    tick_text_size = label_size-2

    # Label and format axes
    plt.ylabel("ln(D/a$^2$)",fontsize = label_size)
    plt.xlabel("$10^4/T$ $(K^{-1})$",fontsize = label_size)
    plt.xticks(fontsize = tick_text_size)
    plt.yticks(fontsize = tick_text_size)
    ax1.set_box_aspect(1)
    plt.ylim(-23,-5)
    plt.xlim(5.,14.3)
  
    

    w1 = 6
    h1 = 6
    fig.set_size_inches(w =w1, h=h1)
    fig.tight_layout()
    # Save output
    plt.savefig(plot_path+"1")
 
    plt.close(fig)

    # Create axes for plotting the gas fractions as a function of step #
   # ax2 = plt.subplot2grid((4,11),(0,5), colspan=8, rowspan=4)
    fig = plt.figure(figsize = (1,1))
    ax2 = plt.subplot(1,1,1)
    ax2.set_box_aspect(.55)


    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(1.15)

    # Put Fi_MDD in non-cumulative space
    temp = Fi_MDD[1:] - Fi_MDD[0:-1]
    Fi_MDD = np.insert(temp, 0, Fi_MDD[0])

    temp = Fi_MDD2[1:] - Fi_MDD2[0:-1]
    Fi_MDD2 = np.insert(temp, 0, Fi_MDD2[0])

    temp = Fi_MDD3[1:] - Fi_MDD3[0:-1]
    Fi_MDD3 = np.insert(temp, 0, Fi_MDD3[0])

    # Get gas fractions from laboratory experiment and put in non-cumulative space
    Fi = np.array(dataset["Fi"])
    temp = Fi[1:] - Fi[0:-1]
    Fi = np.insert(temp, 0, Fi[0])




    marker_size_measured_fig2 = 19
    marker_size_modeled_fig2 = marker_size_measured_fig2-8
    # Plot T_plot vs the gas fraction observed at each step 1for values that were included
    plt.errorbar(
        included+1,
        Fi[included],
        fmt='o', 
        markersize=marker_size_measured_fig2, 
        mfc= (0.69,0.69,0.69), 
        mec='black', 
        zorder = 2,
        linewidth = 1,
        color = 'k'
    )

    # Plot T_plot vs the gas fraction observed at each step for values that were omitted
    plt.errorbar(
        omitted+1,
        Fi[omitted],
        fmt='o', 
        markersize=marker_size_measured_fig2, 
        mfc= (0.69,0.69,0.69), 
        mec='black', 
        zorder = 2,
        linewidth = 1,
        color = 'k',
        alpha = 0.3
    )

    # Plot T_plot vs the modeled gas fraction observed at each step that was included
    plt.plot(included+1, 
                    Fi_MDD[included], 
                    "o", 
                    markersize=marker_size_modeled_fig2, 
                    color=color_best_fit, 
                    linewidth=1, 
                    mec=color_best_fit,
                    zorder = 4
                    )

    line, = plt.plot(included+1, 
                    Fi_MDD2[included], 
                    "o", 
                    markersize=marker_size_modeled_fig2, 
                    color=color_else, 
                    linewidth=1, 
                    mec='black',
                    zorder = 6
                    )
    
    # plt.plot(included+1, 
    #                 Fi_MDD2[included], 
    #                 "o", 
    #                 markersize=marker_size_modeled, 
    #                 color=color_else, 
    #                 linewidth=1, 
    #                 mec='black',
    #                 zorder = 10
    #                 )
    
    plt.plot(included+1, 
                    Fi_MDD3[included], 
                    "o", 
                    markersize=marker_size_modeled_fig2, 
                    color=color_med, 
                    linewidth=1, 
                    mec='black',
                    zorder = 8
                    )


    # This loop draws lines between the points and makes them transparent depending on whether 
    # or not each value was included in the optimization/fitting
    for i in range(len(Fi_MDD)-1):
            if i in omitted or i+1 in omitted:
                alpha_val = .0
            else:
                alpha_val = .7
            plt.plot(range(i+1,i+3),
                     Fi_MDD[i:i+2],
                     '-',
                     color = color_best_fit,
                     alpha = alpha_val,
                     zorder = 3
                     )
            plt.plot(range(i+1,i+3),
                     Fi[i:i+2],
                     '--',
                     color = (0.69,0.69,0.69),
                     alpha = alpha_val,
                     zorder = 1
                     )
            

    for i in range(len(Fi_MDD2)-1):
        if i in omitted or i+1 in omitted:
            alpha_val = .0
        else:
            alpha_val = .7
        plt.plot(range(i+1,i+3),
                    Fi_MDD2[i:i+2],
                    '-',
                    color = "black",
                    alpha = alpha_val,
                    linewidth = 1,
                    zorder = 5
                    )
        plt.plot(range(i+1,i+3),
                    Fi_MDD3[i:i+2],
                    '-',
                    color = "black",
                    alpha = alpha_val,
                    linewidth = 1,
                    zorder = 7
                    )

    # Label axes
    plt.xlabel("Step Number",fontsize = label_size)
    plt.ylabel("Fractional Release (%)", fontsize = label_size)
    plt.xlim(0,34)
    plt.ylim(-.003,0.063)
    plt.xticks(fontsize = tick_text_size)
    plt.yticks(fontsize = tick_text_size)

    fig.set_size_inches(w =w1*2, h=h1)


    # Save output   
    fig.tight_layout() 
    plt.savefig(plot_path+"2")
    plt.close(fig)
    # Create space for the residual plot
   # ax3 = plt.subplot2grid((4,4), (0,2), colspan=2, rowspan=2)
    
    # # Calculate the residual using the highest-retentivity domain as a reference for both 
    # # the model and experimental data
    # m = params[0]/83.14 #Activation energy (kJ/mol) / gas constant
    # m2 = params[0]/83.14
    # m3 = params[0]/83.14

    # resid_exp = dataset["ln(D/a^2)"] - (-m.item() * T_plot + params[1].item())
    # resid_model = np.array(lnd0aa_MDD.ravel()) - (-m.item() *T_plot + params[1].item())
    # resid_model2 = np.array(lnd0aa_MDD2.ravel()) - (-m.item() *T_plot + params2[1].item())
    # resid_model3 = np.array(lnd0aa_MDD3.ravel()) - (-m.item() *T_plot + params3[1].item())

    # # Plot the values of residuals that were included in the fit calculated against the experimental results
    # cum_Fi_MDD = np.cumsum(Fi_MDD)
    # cum_Fi_MDD2 = np.cumsum(Fi_MDD2)
    # cum_Fi_MDD3 = np.cumsum(Fi_MDD3)
    # cum_Fi_exp = np.cumsum(Fi)


    # # plt.plot(cum_Fi_exp[included] * 100, 
    # #                 resid_exp[included], 'o', markersize=marker_size_measured, 
    # #                 color= (0.69,0.69,0.69), 
    # #                 mec='black', 
    # #                 alpha = 0.8
    # #                 )
    
    # # # Plot the values of residuals that were excluded in the fit calculated against the experimental results
    # # plt.plot(cum_Fi_exp[omitted] * 100, 
    # #                 resid_exp[omitted], 'o', markersize=marker_size_measured, 
    # #                 color= (0.69,0.69,0.69), 
    # #                 mec='black', 
    # #                 alpha = 0.3
    #                 # )
    # # Plot the values of residuals that were included in the fit calculated against the model results
    # # plt.plot(cum_Fi_MDD[included] * 100, 
    # #          resid_model[included], 
    # #                 "o", markersize=marker_size_modeled, 
    # #                 color=color_best_fit,  
    # #                 linewidth=1, 
    # #                 mec=color_best_fit
    # #                 )
    
    # # # Plot the values of residuals that were included in the fit calculated against the model results
    # # plt.plot(cum_Fi_MDD2[included] * 100, 
    # #          resid_model2[included], 
    # #                 "o", markersize=marker_size_modeled, 
    # #                 #color=color_else,  
    # #                 linewidth=1, 
    # #                 mec=color_else
    # #                 )
    
    # #     # Plot the values of residuals that were included in the fit calculated against the model results
    # # plt.plot(cum_Fi_MDD3[included] * 100, 
    # #          resid_model3[included], 
    # #                 "o", markersize=marker_size_modeled, 
    # #                 #color=color_med,  
    # #                 linewidth=1, 
    # #                 mec=color_med
    # #                 )
    
    # # # # Plot the values of residuals that were excluded in the fit calculated against the model results
    # # # plt.plot(cum_Fi_MDD[omitted] * 100, 
    # # #          resid_model[omitted], 
    # # #             "o", markersize=marker_size_modeled, 
    # # #             color='black',  
    # # #             linewidth=1, 
    # # #             mec='black',
    # # #             alpha = 0.3
    # # #             )
    
    # # # This loop draws lines between the points and makes them transparent depending on whether 
    # # # or not each value was included in the optimization/fitting
    # # for i in range(len(cum_Fi_MDD)-1):
    # #         if i in omitted or i+1 in omitted:
    # #             alpha_val = .0
    # #         else:
    # #             alpha_val = 1.
    # #         plt.plot(cum_Fi_MDD[i:i+2]*100,
    # #                  resid_model[i:i+2],
    # #                  '-',
    # #                  color = color_best_fit,
    # #                  alpha = alpha_val,
    # #                  zorder = 250
    # #                  )
    # #         plt.plot(cum_Fi_MDD2[i:i+2]*100,
    # #                  resid_model2[i:i+2],
    # #                  '-',
    # #                  color = color_else,
    # #                  alpha = alpha_val,
    # #                  linewidth = .5,
    # #                  zorder = 250
    # #                  )
    # #         plt.plot(cum_Fi_MDD3[i:i+2]*100,
    # #                  resid_model3[i:i+2],
    # #                  '-',
    # #                  color = color_med,
    # #                  alpha = alpha_val,
    # #                  linewidth = .5,
    # #                  zorder = 250
    # #                  )


    # #         plt.plot(cum_Fi_exp[i:i+2]*100,
    # #                  resid_exp[i:i+2],
    # #                  '--',
    # #                  color= (0.69,0.69,0.69),
    # #                  alpha = alpha_val,
    # #                  zorder = 1
    # #                  )

    # # Format plot and label axes
    # # ax3.set_box_aspect(1)
    # # for axis in ['top', 'bottom', 'left', 'right']:
    # #     ax3.spines[axis].set_linewidth(1.15)
    # # plt.xlabel("Cum. Gas Release (%)",fontsize = label_size)
    # # plt.ylabel("Residual ln(1/s)",fontsize = label_size)
