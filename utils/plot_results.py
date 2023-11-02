import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.get_plot_name import get_plot_name
from optimization.forwardModelKinetics import forwardModelKinetics
from optimization.forwardModelKinetics import calc_lnd0aa

def plot_results(
    params,
    dataset,
    objective,
    sample_name: str = "",
    moves_type: str = "",
    misfit_stat: str = "",
    quiet=False,
):
    """Plot the results of the optimization.

    Args:
        params (torch.tensor): The parameters from the optimization.
        dataset (Dataset): The dataset for the optimization.
        objective (DiffusionObjective): The objective function for the optimization.
        reference_law (list, optional): The reference law for the optimization. Defaults to [].
        sample_name (str, optional): The name of the sample. Defaults to "".
        moves_type (str, optional): _description_. Defaults to "".
        misfit_stat (str, optional): _description_. Defaults to "".
        quiet (bool, optional): Whether to show the plot. Defaults to False.
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
                        Fi_MDD, objective.tsec, objective.geometry, objective.extra_steps, objective.added_steps
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
    else:
        fracs = 1
        frac_weights = [2]


    fig, axes = plt.subplots(ncols=2, nrows=2, layout="constrained", figsize=(10, 10))


    # This is going to cause and error and Drew needs to fix
    errors_for_plot = np.array(
        pd.concat(
            [
                dataset["ln(D/a^2)"] - dataset["ln(D/a^2)-del"],
                dataset["ln(D/a^2)+del"] - dataset["ln(D/a^2)"],
            ],
            axis=1,
        ).T
    )



    for i in range(ndom):
        
        # Calculate a line representing each domain
        D = np.log(
            np.exp(params[i + 1])
            * np.exp((-params[0]) / (R * (TC + 273.15)))
        )

        # Plot each line
        axes[0, 0].plot(
            np.linspace(min(T_plot), max(T_plot), 1000),
            np.linspace(max(D), min(D), 1000),
            "--",
            linewidth=frac_weights[i],
            zorder=0,
            color = (0.6,0,0),
            alpha = 0.5
        )
    

    # Plot the MDD Model lndaa values
    axes[0, 0].plot(
        T_plot,
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
    axes[0, 0].set_ylabel("ln(D/a^2)")
    axes[0, 0].set_xlabel("10000/T (K)")
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
        linestyle = '--',
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
        # axes[2].subplot(n_plots,1,3)
        axes[1, 1].errorbar(
            range(0, len(T_plot)),
            dataset["M"],
            yerr=dataset["delM"],
            fmt='-o', 
            markersize=12, 
            mfc= (0.69,0.69,0.69), 
            mec='black', 
            alpha = 0.8,
            zorder = 10,
            linewidth = 1,
            linestyle = '--',
            color = 'k'

        )
        axes[1, 1].plot(
            range(0, len(T_plot)), tot_moles * Fi_MDD, "k-o", markersize=3, zorder=10
        )
        axes[1, 1].set_xlabel("step number")
        axes[1, 1].set_ylabel("Atoms Released at Each Step")
        axes[1, 1].set_box_aspect(1)
        # axes[2].axis('square')

    if n_plots == 4:
        # Calculate reference law results

        # Slope
        m = params[0]/83.14 #Activation energy (kJ/mol) / gas constant
   
        resid_exp = dataset["ln(D/a^2)"] - (-m * T_plot + params[1].item())
        resid_model = lnd0aa_MDD.ravel() - (-m*T_plot + params[1].item())
    
        axes[0, 1].plot(data[0] * 100, 
                        resid_exp, 'o', markersize=12, 
                        color= (0.69,0.69,0.69), 
                        mec='black', 
                        alpha = 0.8
                        )
        

        axes[0, 1].plot(data[0] * 100, resid_model, 
                        "-o", markersize=5, 
                        color='black', 
                        linestyle='-', 
                        linewidth=1, 
                        mec='black'
                        )
        
        axes[0, 1].set_xlabel("Cumulative 3He Release (%)")
        axes[0, 1].set_ylabel("Residual ln(1/s)")
        axes[0, 1].set_box_aspect(1)

    plt.tight_layout
    file_name = get_plot_name(
        ndom, "fit_plot", sample_name, moves_type=moves_type, misfit_stat=misfit_stat
    )

    plt.savefig(file_name)

    if quiet == False:
        plt.show()



    
