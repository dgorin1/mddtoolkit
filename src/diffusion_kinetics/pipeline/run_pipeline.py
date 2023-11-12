import pandas as pd
import numpy as np
import yaml
import torch
from diffusion_kinetics.pipeline import PipelineConfig, PipelineOutput
from diffusion_kinetics.optimization import (
    Dataset, 
    DiffusionObjective,
    diffEV_multiples,
)
from diffusion_kinetics.utils.plot_results import plot_results
from diffusion_kinetics.utils.organize_x import organize_x
from diffusion_kinetics.utils.save_results import save_results
from typing import Union
from  diffusion_kinetics.optimization import DiffusionOptimizer

def run_pipeline(
    input_path:str, 
    output_path:str, 
    config:Union[str, dict, PipelineConfig]=PipelineConfig(),
):
    if not isinstance(config, PipelineConfig):
        config = PipelineConfig.load(config)
    dataset = Dataset(pd.read_csv(input_path))
    pipe_out = PipelineOutput(output_path)
    optimizer = DiffusionOptimizer(dataset, config)
    optimizer.run(misfit_stat="chi2", ndom=1)
    breakpoint()
    # save config to output
    config.to_yaml(pipe_out.config_path)
    
    for misfit_stat in config.misfit_stat_list:
        save_params = np.empty((config.max_domains_to_model - 1, config.max_domains_to_model * 2 + 4))
        save_params.fill(np.NaN)
        i = 1
        prev_misfit = 11**17
        misfit_val = 10**17
        
        while i <= config.max_domains_to_model and misfit_val < prev_misfit:
            prev_misfit = misfit_val
            domains_to_model = i
            print(f"{misfit_stat} with {domains_to_model} domains")
            
            objective = DiffusionObjective(
                dataset,
                time_add=torch.tensor(config.time_add),
                temp_add=torch.tensor(config.temp_add),
                omitValueIndices=config.omit_value_indices,
                stat=misfit_stat,
                geometry=config.geometry,
                punish_degas_early = True
            )
            
            params, misfit_val = diffEV_multiples(
                objective,
                dataset,
                config.iteration_repeats,
                domains_to_model,
                Ea_bounds=config.ea_bounds,
                lnd0aa_bounds=config.lnd0aa_bounds,
                max_iters=config.max_iters,
            )
            
            plot_results(
                params,
                dataset,
                objective,
                pipe_out,
                quiet=True,
            )
            
            params = organize_x(params, len(params), chop_fracs=False)
            print(params)

            if i < config.max_domains_to_model:
                num_nans_insert = config.max_domains_to_model - i
                nan_insert = np.empty((num_nans_insert))
                nan_insert.fill(np.NaN)
                array_w_nans = np.insert(params, [2 + i], nan_insert, axis=0)
                array_w_nans = np.concatenate((array_w_nans, nan_insert), axis=0)

            else:
                array_w_nans = params
            add_num_doms = np.append(i, array_w_nans)
            params_to_add = np.append(add_num_doms, misfit_val)

            save_params[i - 1, 0 : len(params_to_add)] = params_to_add

            save_results(
                ndom=domains_to_model, params=save_params, pipe_out=pipe_out
            )
            misfit = misfit_val
            i = i + 1
                
        

if __name__ == "__main__":    
    run_pipeline(
        "/Users/josh/repos/diffusion_code_final/example/data/input_8DomSynthDataNoisyM3_plane_sheet.csv",
        "/Users/josh/repos/diffusion_code_final/example/output",
        "/Users/josh/repos/diffusion_code_final/example/data/config.yaml"
    )
