import pandas as pd
import numpy as np
import torch
from diffusion_kinetics.pipeline import PipelineConfig
from diffusion_kinetics.optimization import (
    Dataset, 
    DiffusionObjective,
    diffEV_multiples,
)
from diffusion_kinetics.utils.plot_results import plot_results

def run_pipeline(
    input_path:str, 
    output_path:str, 
    config:dict={}
):
    config = PipelineConfig(**config)
    dataset = Dataset(pd.read_csv(input_path))
    
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
                sample_name=sample_name,
                quiet=True,
                misfit_stat=misfit_stat,
            )
            print(sample_name)
                
        

if __name__ == "__main__":
    run_pipeline(
        "/Users/josh/repos/diffusion_code_final/public/data/input_8DomSynthDataNoisyM3_plane_sheet.csv",
        "/Users/josh/repos/diffusion_code_final/public/test_new_output_path"
    )
