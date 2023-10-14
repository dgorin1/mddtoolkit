import argparse
# from diffusion_optimizer.parameter_tuning import tune_parameters
from diffusion_optimizer.generate_inputs import generate_inputs

import yaml
import numpy as np
import os 
import uuid
import sys
import pandas as pd

from diffusion_optimizer.neighborhood.objective import Objective

from diffusion_optimizer.neighborhood.optimizer import Optimizer, OptimizerOptions
from diffusion_optimizer.neighborhood.dataset import Dataset

file_path = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    root_output_dir = f"{file_path}/../../../main/output"
    
    # generate output folder
    run_id = uuid.uuid1()
    full_output_dir = f"{root_output_dir}/{run_id}"
    config = yaml.safe_load(open(sys.argv[sys.argv.index("--config_path") + 1]))
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", required=True)    
    argparser.add_argument("--input_csv_path", default=f"{file_path}/../../../main/input/{config['nameOfInputCSVFile']}")
    argparser.add_argument("--output_path", default=full_output_dir)
    argparser.add_argument("-generate_inputs", action='store_true')
    argparser.add_argument("-parameter_tuning", action='store_true')
    return (argparser.parse_args(), config)

def list2range(list):
    return range(list[0], list[1])

def list2tuple(list):
    return (list[0], list[1])
    
def main():
    (args, config) = parse_args()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    # convert arrays to tuples and limits in config
    limitDict = config["limits"]
    names = list(limitDict.keys())
    limits = [list2tuple(limit) for limit in limitDict.values()]
    
    # generate inputs
    if args.generate_inputs:
        generate_inputs(args.input_csv_path, f"{args.output_path}/{config['nameOfExperimentalResultsFile']}")
    # run parameter tuning
    if args.parameter_tuning:
        tune_parameters(args.config_path, f"{args.output_path}/{config['nameOfExperimentalResultsFile']}", args.output_path)
    
    # else:
    #     dataset = Dataset(pd.read_csv(f"{args.output_path}/{config['nameOfExperimentalResultsFile']}"))
    #     objective = DiffusionObjective(dataset)
        
    #     options = OptimizerOptions(
    #         num_samp_range=[8,800],
    #         num_samp_decay=2.1,
    #         resamp_to_samp_ratio = 0.4,
    #         epsilon_range=[0.000001,0.00001],
    #         epsilon_growth=0.01
    #     )
        
    #     optimizer = Optimizer(objective, limits, names, options)
    #     optimizer.update(1000)
    #     print("TODO: implement single pass process")
    #     pass

if __name__ == "__main__":
    main()
