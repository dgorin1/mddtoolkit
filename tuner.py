from hyperopt import fmin, tpe, hp, Trials,STATUS_OK
from diffusion_objective import DiffusionObjective
from diffusion_problem import DiffusionProblem
from jax import numpy as jnp
from dataset import Dataset

import torch as torch
import pandas as pd
import os
import numpy as np
from conHe_Param import conHe_Param
from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint
from emcee_main import emcee_main
from generate_bounds import generate_bounds
import random
import time
import pickle




search_space = {
    'popsize': hp.quniform('popsize', 5, 10,1),
    'strategy': hp.choice('strategy',['best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin', 'currenttobest1bin', 'best2bin', 'rand2bin', 'rand1bin']),
    'mutation1': hp.uniform('mutation1', 0, 2),
    'mutation2': hp.uniform('mutation2', 0, 1),
    'recombination': hp.uniform('recombination',0,1),
    'init': hp.choice('init',(['latinhypercube','sobol','halton','random']))




}

def objective(hyperparameters):
    popsize = hyperparameters['popsize']
    strategy = hyperparameters['strategy']
    mutation1 = hyperparameters['mutation1']
    mutation2 = hyperparameters['mutation2']
    recombination = hyperparameters['recombination']
    init = hyperparameters['init']

    mutation = (mutation1, (2-mutation1)*mutation2+mutation1)

    # get this file's directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_input = pd.read_csv(f"{dir_path}/data/input_KM95-28-Dc-1250um.csv")
    domains_to_model = 4
    mineral_name = "quartz"
    time_add = [300*60,110073600]
    temp_add = [40,21.111111111111]
    sample_name = "KM95-28-Dc-1250um"
    moves = "default" # Define moves as "snooker" if you fear multimodality in your dataset. Can lead to poor performance if no multimodality exists

    # Create dataset class for each associate package
    dataset = Dataset("ipopt", data_input)
    datasetEv = Dataset("diffEV", data_input)

    objective_diffEV = DiffusionObjective(
        "diffEV",
        datasetEv, 
        time_add = torch.tensor(time_add), 
        temp_add = torch.tensor(temp_add), 
        pickle_path = f"{dir_path}/data/lookup_table.pkl",
        omitValueIndices= []
    )

    # Read in the nonlinear constraint
    nlc = NonlinearConstraint(conHe_Param,lb =[0,0],ub = [np.inf,np.inf])

    mole_bound = tuple((sum(dataset.M)- 1*jnp.sqrt(sum(dataset.delM **2)), sum(dataset.M) + 1*jnp.sqrt(sum(dataset.delM **2))))

    bounds = generate_bounds(domains_to_model,mole_bound,"Quartz")

    intermediate_run_results = []
    minVal = []
    for i in range(2):
        time_start = time.time()
        intermediate_run_results = []
        for i in range(10):
            diffEV = differential_evolution(
                objective_diffEV, 
                bounds, 
                constraints = nlc,
                seed = random.randint(0, 2**28),
                init = init,
                popsize = int(popsize),
                mutation = mutation,
                recombination = recombination,
                strategy = strategy,
                maxiter = 3000,
                vectorized = True,
                updating = "deferred",
            )



            intermediate_run_results.append(float(diffEV.fun))
        minVal.append(min(intermediate_run_results))

    final = np.median(minVal)

            


    return {'loss': final,'duration': time.time()-time_start,'status': STATUS_OK}

trials = Trials()  # Used to store intermediate results


best = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=1,  # Number of trials to run
    trials=trials
)
trials = Trials()
for i in range(1000):
    print(f"{(i+1)*10} iteration set")
    best_hyperparameters = fmin(objective, search_space, algo=tpe.suggest, max_evals=(i+1)*10,trials=trials)
    best_metric = -trials.best_trial['result']['loss']
    with open(f"tuner_trials_KM95-28_4dom_min.pkl", "wb") as file:
        pickle.dump(trials, file)

breakpoint()