import pandas as pd
from diffusion_kinetics.pipeline import SingleProcessPipelineConfig, PipelineOutput
from diffusion_kinetics.optimization import Dataset
from typing import Union
from  diffusion_kinetics.optimization import DiffusionOptimizer


class Pipeline:
    def __init__(
        self,
        dataset: Dataset,
        output: PipelineOutput = None,
    ):
        self.dataset = self._load_dataset(dataset)
        self.optimizer = DiffusionOptimizer(self.dataset)
        self.output = self._create_output(output)

    def run(self, config: Union[str, dict, SingleProcessPipelineConfig]):
        """
        Run the pipeline.
        """
        config = self._load_config(config)
        res = self.optimizer.run(config)

        if self.output:
            self.output.save_results(res, config, self.dataset)
            
        res = {
            "x": res.x,
            "fun": res.fun,
            "success": res.success,
            "message": res.message,
            "nit": res.nit,
            "nfev": res.nfev,
        }
        
        return res
    
    def _load_config(self, config:Union[str, dict, SingleProcessPipelineConfig]):
        if isinstance(config, str):
            config = SingleProcessPipelineConfig.load(config)
        elif isinstance(config, dict):
            config = SingleProcessPipelineConfig(**config)
        elif not isinstance(config, SingleProcessPipelineConfig):
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
        elif isinstance(output, type(None)):
            output = None
        else:
            raise ValueError(f"output must be a path to a directory or a PipelineOutput object. Got: {output.__class__.__name__}")
        return output
