import pandas as pd
import numpy as np
import yaml
import torch
from diffusion_kinetics.pipeline import MultiProcessPipelineConfig, PipelineOutput
from diffusion_kinetics.optimization import (
    Dataset, 
    DiffusionObjective,
    diffEV_multiples,
)
from diffusion_kinetics.utils.plot_results import plot_results
from diffusion_kinetics.utils.organize_x import organize_x
from diffusion_kinetics.utils.save_results import save_results
from typing import Union
from  diffusion_kinetics.pipeline import OptimizerPool

class MultiPipeline:
    def __init__(
        self,
        dataset:Union[str, pd.DataFrame, Dataset],
        config:Union[str, dict, MultiProcessPipelineConfig]=MultiProcessPipelineConfig(),
        output:Union[str, PipelineOutput]=None,
    ):
        # initialize the config
        if not isinstance(config, MultiProcessPipelineConfig):
            self.config = MultiProcessPipelineConfig.load(config)
            
        if isinstance(dataset, str):
            self.dataset = Dataset(pd.read_csv(dataset))
        if isinstance(dataset, pd.DataFrame):
            self.dataset = Dataset(dataset)
        if not isinstance(dataset, Dataset):
            raise ValueError(f"dataset must be a path to a csv file, a pandas dataframe, or a Dataset object. Got: {dataset.__class__.__name__}")
        
    
    def run(self):
        optimizer_pool = OptimizerPool(self.dataset)
        
        for misfit_stat in self.config.misfit_stat_list:
            sp_configs = self.config.single_pipeline_configs[misfit_stat]
            optimizer_pool.run(sp_configs)        
