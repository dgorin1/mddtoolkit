import pandas as pd
import numpy as np
import yaml
import torch
from diffusion_kinetics.pipeline import MultiProcessPipelineConfig, PipelineOutput
from diffusion_kinetics.optimization import (
    Dataset
)
from diffusion_kinetics.utils.plot_results import plot_results
from diffusion_kinetics.utils.organize_x import organize_x
from diffusion_kinetics.utils.save_results import save_results
from typing import Union
from  diffusion_kinetics.pipeline.optimizer_pool import OptimizerPool

class MultiPipeline:
    def __init__(
        self,
        dataset:Union[str, pd.DataFrame, Dataset],
        config:Union[str, dict, MultiProcessPipelineConfig]=MultiProcessPipelineConfig(),
        output:Union[str, PipelineOutput]=None,
    ):
        self.dataset = self._load_dataset(dataset)
        self.config = self._load_config(config)
        self.output = self._create_output(output)
    
    def run(self):
        optimizer_pool = OptimizerPool(self.dataset)
        results = []
        # run the optimization for each misfit statistic
        for misfit_stat in self.config.misfit_stat_list:
            sp_configs = self.config.single_pipeline_configs[misfit_stat]
            result = optimizer_pool.run(sp_configs)
            results.append(result)
            # save the results
            if self.output:
                self.output.save_results(result, self.config, self.dataset)
        return results
    
    def _load_config(self, config:Union[str, dict, MultiProcessPipelineConfig]):
        if isinstance(config, str):
            config = MultiProcessPipelineConfig.load(config)
        elif isinstance(config, dict):
            config = MultiProcessPipelineConfig(**config)
        elif not isinstance(config, MultiProcessPipelineConfig):
            raise ValueError(f"config must be a path to a yaml file, a dictionary, or a SingleProcessPipelineConfig object. Got: {config.__class__.__name__}")
        return config
            
    def _load_dataset(self, dataset:Union[str, pd.DataFrame, Dataset]):
        if isinstance(dataset, str):
            dataset = Dataset(pd.read_csv(dataset))
        elif isinstance(dataset, pd.DataFrame):
            dataset = Dataset(dataset)
        elif not isinstance(dataset, Dataset):
            raise ValueError(f"dataset must be a path to a csv file, a pandas dataframe, or a Dataset object. Got: {dataset.__class__.__name__}")
        return dataset

    def _create_output(self, output:Union[str, PipelineOutput]):
        if isinstance(output, PipelineOutput):
            pass
        elif isinstance(output, str):
            output = PipelineOutput(output)
        else:
            raise ValueError(f"output must be a path to a directory or a PipelineOutput object. Got: {output.__class__.__name__}")
        return output
