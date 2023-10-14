

import numpy as  np
from dataset import Dataset
from diffusion_objective import DiffusionObjective
import pandas as pd
import torch as torch
import emcee
import matplotlib.pyplot as plt
import corner
from multiprocessing import Pool
import os
import cProfile
import datetime as datetime
from get_plot_name import get_plot_name


   


def generate_labels(num_domains):
    labels = ["Total Moles", "Ea"]

    for domain_number in range(1, num_domains + 1):
        labels.append("LnD0aa_" + str(domain_number))

    for domain_number in range(1, num_domains):
        labels.append("Frac_" + str(domain_number))

    return labels

def emcee_main(x, objective, num_iters = 10000,sample_name:str = "",moves:str = ""):
    os.environ["OMP_NUM_THREADS"] = "1"


    def fracs_prior(fracs):
        if np.sum(fracs) <= 1 and np.sum(fracs) == np.sum(np.abs(fracs)):
            return 0
        else:
            return -np.inf
        
    def moles_prior(moles,objective):

        return -(1/2)*((moles-objective.total_moles)**2/(objective.total_moles_del**2))
        
    def Ea_prior(Ea):
        if 30 < Ea < 150:
            return 0
        else:
            return -np.inf
    def lnd0aa_prior(lnD0aa):
        if len(lnD0aa)>1: #If we have greater than one domain
            diff = lnD0aa[0:-1]-lnD0aa[1:] # Take difference and 
            if sum(diff<0) > 0:
                return -np.inf
            elif all(-10 <= x <= 35 for x in lnD0aa):
                return 0
            else:
                return -np.inf
        else:
            return 0

    def log_prior(theta,objective):
        total_moles = theta[0]
        moles_P = moles_prior(total_moles,objective)
        theta = theta[1:] # Shorten X to remove this parameter
        Ea = theta[0]
        
        # Get the Ea prior
        Ea_P = Ea_prior(Ea)

        # Unpack the parameters
        if len(theta) <= 3:
            ndom = 1
        else:
            ndom = (len(theta))//2

        # Grab the other parameters from the input
        temp = theta[1:]
        lnD0aa = temp[0:ndom]
        # Get lnD0aa prior
        lnD0aa_P = lnd0aa_prior(lnD0aa)

        # Get fracs prior
        fracs = temp[ndom:]
        fracs_P = fracs_prior(fracs)

        return moles_P+Ea_P+lnD0aa_P+fracs_P

    def log_likelihood(theta,objective):     
        ans = (objective.__call__(theta)).numpy()

        if ans.size > 1:
            ans = np.sum(ans)
        return -(1/2)*ans

    def log_probability(theta,objective):

        lp = log_prior(theta,objective)
        if not np.isfinite(lp):
            return -np.inf

        return lp+log_likelihood(theta,objective)

    def organize_x(x,ndim):
        ndom = int(((ndim-1)/2))
        moles = x[0]
        Ea = x[1]
        lnd0aa = x[2:2+ndom]
        fracs = x[2+ndom:]
        fracs = np.append(fracs,1-np.sum(fracs))
        
        n = len(fracs)
        # Traverse through all array elements
        for i in range(n):
            
            # Last i elements are already in place
            for j in range(0, n - i - 1):
                
                # Traverse the array from 0 to n-i-1
                # Swap if the element found is greater than the next element
                if lnd0aa[j] < lnd0aa[j + 1]:
                    lnd0aa[j], lnd0aa[j + 1] = lnd0aa[j + 1], lnd0aa[j]
                    fracs[j], fracs[j + 1] = fracs[j + 1], fracs[j]
        output = np.append(moles,Ea)
        output = np.append(output,lnd0aa)
        output = np.append(output,fracs[0:-1])
        return output


    nwalkers = int(len(x)*2+0.3*(len(x)+2))
    ndim = len(x)

    # Put x values in order using a modified bubble sort if they arent already in order.
    x = organize_x(x,ndim)

   
    multiplier = np.concatenate((np.random.uniform(low=-0.001, high=0.001, size=[nwalkers, 1]), np.random.uniform(low=-0.001, high=0.001, size=[nwalkers, ndim-1])), axis=1)
    pos = np.tile(x, (nwalkers, 1)) * (1 + multiplier) 


    # pos = np.tile(x,(nwalkers,1))+ np.tile(x,(nwalkers,1))*np.random.uniform(low= -0.05,high =0.05,size=[nwalkers,ndim])
    # pos = np.row_stack((x,pos))

    if moves.lower() == "snooker":
        moves_label = moves
        moves =[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2)]

    else:
        moves = None
        moves_label = "default"

    # moles_add = np.random.normal(4e+09, 24943848.04, size=(nwalkers, 1))
    
    # pos = np.column_stack([moles_add,pos])
    filename = get_plot_name(int(((ndim-1)/2)),"MCMC_Data",sample_name = sample_name,file_type = "h5",moves_type = moves_label)

    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, 
    ndim,
    log_probability, 
    args = [objective], 
    moves = moves,
    backend = backend
    )


    max_n = num_iters

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(pos, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 1000:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        
        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.1)
        converged &= sampler.iteration > 10000
        if converged:
            break
        old_tau = tau


        print(tau)
    # sampler.run_mcmc(pos, num_iters, progress=True,backend=backend)


    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    labels = generate_labels(int(((ndim-1)/2)))
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    file_name = get_plot_name(int(((ndim-1)/2)),"chain_plot",sample_name,moves_type = moves_label)
    plt.savefig(file_name)


    #tau = sampler.get_autocorr_time()
    #print(tau)
    number_to_discard = int(np.max(tau)*2)
    flat_samples = sampler.get_chain(discard=number_to_discard, thin=15, flat=True)


    mcmc = []
    for i in range(ndim):
        mcmc.append(np.percentile(flat_samples[:, i], [50-16, 50, 50+16]))
        print(np.percentile(flat_samples[:, i], [50-16, 50, 50+16]))

    tau = sampler.get_autocorr_time(tol=0)
    #print(tau)



    n = 1000 * np.arange(1, index + 1)
    y = autocorr[:index]
    plt.figure()
    plt.plot(n, n / 100.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$");
    
    file_name = get_plot_name(int(((ndim-1)/2)), "correlation_time", sample_name=sample_name,moves_type = moves_label)
    plt.savefig(file_name)

    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))


    all_samples = np.concatenate(
        (samples, log_prob_samples[:, None]), axis=1
    )

    # labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))
    # labels += ["log prob", "log prior"]
    params_final = []
    for i in range(len(mcmc)):
        params_final.append(mcmc[i][1])

   
   
    corner.corner(flat_samples, labels=labels,truths = params_final)
    file_name = get_plot_name(int(((ndim-1)/2)), "corner_plot", sample_name=sample_name,moves_type = moves_label)
    plt.savefig(file_name)
    plt.show()
    
    return mcmc
