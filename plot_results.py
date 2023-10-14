import torch as torch
import numpy as np
from calc_arrhenius import calc_arrhenius
import matplotlib.pyplot as plt
from jax import numpy as jnp
import pandas as pd
from get_plot_name import get_plot_name

def plot_results(params,dataset,objective, reference_law = [],sample_name:str = "",moves_type:str = "",misfit_stat:str = "",quiet = False):
    # Params is a vector X of the input parameters
    # dataset is the dataset class with your data
    # objective is the objective you used
    # reference_law is an array with values [Ea, lnd0aa]

    R = 0.008314
    params = torch.tensor(params)
    if len(params) %2 !=0:
        tot_moles = params[0]
        params = params[1:]
        moles_calc = True
    else:
        moles_calc = False

        
        # Infer the number of domains from input
    if len(params) <= 3:
        ndom = 1
    else:
        ndom = (len(params))//2

   
    # Reconstruct the time-added and temp-added inputs
    time = torch.tensor(dataset.thr*3600)
    TC = torch.tensor(dataset.TC)
    if objective.extra_steps == True:

        tsec = torch.cat([torch.tensor(np.array(objective.time_add)),time])
        TC = torch.cat([torch.tensor(np.array(objective.temp_add)),TC])
    else:
        tsec = objective.tsec



    data = calc_arrhenius(params,objective.lookup_table,tsec,TC,objective.geometry,objective.extra_steps)


    T_plot = 10000/(dataset["TC"]+273.15)
    if len(reference_law) == 0:
        n_plots = 3
    else:
        n_plots = 4

    fracs = params[ndom+1:]
    fracs = torch.concat((fracs,1-torch.sum(fracs,axis=0,keepdim=True)),axis=-1)

    fig,axes = plt.subplots(ncols = 2, nrows = 2,layout = "constrained",figsize=(10,10))


    errors_for_plot = np.array(pd.concat([dataset["ln(D/a^2)"]-dataset["ln(D/a^2)-del"],dataset["ln(D/a^2)+del"]-dataset["ln(D/a^2)"]],axis=1).T)        

    #plt.subplot(n_plots,1,1)

    frac_weights = (fracs-torch.min(fracs)/(torch.max(fracs)-torch.min(fracs)))*3.5+1.7
    if torch.any(torch.lt(frac_weights,0)):

        frac_weights = torch.abs(frac_weights.flip(0))

    for i in range(ndom):
        D = np.log(np.exp(params[i+1])*np.exp((-params[0])/(R*(dataset["TC"]+273.15))))
        axes[0,0].plot(np.linspace(min(T_plot),max(T_plot),1000), np.linspace(max(D),min(D),1000), '--k',linewidth = frac_weights[i],zorder=0)
    
    axes[0,0].errorbar(T_plot,dataset["ln(D/a^2)"].replace(-np.inf,0),yerr= errors_for_plot,fmt = 'bo',markersize=10,zorder=5)
    
    axes[0,0].plot(T_plot,pd.Series(data[1].tolist()).replace(-np.inf,np.inf).fillna(max(data[1]).item()),'ko',markersize=7,zorder = 10)

    #Normalize Fractions for plotting weights
   
    if n_plots == 4:
        ref = np.log(np.exp(reference_law[1])*np.exp((-reference_law[0])/(R*(dataset["TC"]+273.15))))
    #     breakpoint()
    #     plt.plot(np.linspace(min(T_plot),max(T_plot),1000), np.linspace(max(ref),min(ref),1000), '--k')
    axes[0,0].set_ylabel("ln(D/a^2)")
    axes[0,0].set_xlabel("10000/T (K)")
    #axes[0].xlim([min(T_plot)-2,max(T_plot)+2])
    #axes[0].ylim([min(dataset["ln(D/a^2)"]-2),min(dataset["ln(D/a^2)"]+2)])
    axes[0,0].set_box_aspect(1)

    #plt.subplot(n_plots,1,2)
    Fi_MDD = np.array(data[0])
    temp = Fi_MDD[1:]-Fi_MDD[0:-1]
    Fi_MDD =np.insert(temp,0,Fi_MDD[0])
    Fi = np.array(dataset.Fi) 
    temp = Fi[1:]-Fi[0:-1]
    Fi = np.insert(temp,0,Fi[0])


    axes[1,0].errorbar(range(0,len(T_plot)),Fi,yerr = dataset["Fi uncertainty"],fmt ='b-o',markersize=5,zorder=5)
    axes[1,0].plot(range(0,len(T_plot)),Fi_MDD,'k-o',markersize=3,zorder=10)
    axes[1,0].set_xlabel("step number")
    axes[1,0].set_ylabel("Fractional Release (%)")
    #axes[1].axis('square')
    axes[1,0].set_box_aspect(1)

    if moles_calc == True:
        #axes[2].subplot(n_plots,1,3)
        axes[1,1].errorbar(range(0,len(T_plot)),dataset["M"],yerr = dataset["delM"], fmt ='b-o',markersize=5,zorder=5)
        axes[1,1].plot(range(0,len(T_plot)),tot_moles*Fi_MDD,'k-o',markersize=3,zorder=10)
        axes[1,1].set_xlabel("step number")
        axes[1,1].set_ylabel("Atoms Released at Each Step")
        axes[1,1].set_box_aspect(1)
        #axes[2].axis('square')


    if n_plots == 4:
        #Calculate reference law results
        
        r_r0 = 0.5*(ref-dataset["ln(D/a^2)"])
        #axes[3].subplot(n_plots,1,n_plots)
        axes[0,1].plot(data[0]*100,r_r0, 'b-o', markersize=4)
        axes[0,1].set_xlabel("Cumulative 3He Release (%)")
        axes[0,1].set_ylabel("log(r/r_0)")
        axes[0,1].set_box_aspect(1)     
    plt.tight_layout
    file_name = get_plot_name(ndom,"fit_plot",sample_name,moves_type = moves_type,misfit_stat = misfit_stat)

    plt.savefig(file_name)
    
    if quiet == False:
        plt.show()