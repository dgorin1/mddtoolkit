import pandas as pd
from diffusion_kinetics.pipeline import MultiProcessPipelineConfig, PipelineOutput
from diffusion_kinetics.pipeline.pipeline import Pipeline
from diffusion_kinetics.optimization import Dataset
from typing import Union

class MultiPipeline:
    def __init__(
        self,
        dataset:Union[str, pd.DataFrame, Dataset],
        output:Union[str, PipelineOutput]=None,
    ):
        self.dataset = self._load_dataset(dataset)
        self.output = self._create_output(output)
    
    def run(self, config:Union[str, dict, MultiProcessPipelineConfig]):
        results = []
        config = self._load_config(config)
        pipeline = Pipeline(self.dataset, output=self.output)
        for misfit_type in config.single_pipeline_configs.keys():
            configs_for_each_domain_list = config.single_pipeline_configs[misfit_type]
            for single_pipeline_config in configs_for_each_domain_list:
                res = pipeline.run(single_pipeline_config)
                results.append(res)
                if self.output:
                    self.output.save_results(res, single_pipeline_config, self.dataset)
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
