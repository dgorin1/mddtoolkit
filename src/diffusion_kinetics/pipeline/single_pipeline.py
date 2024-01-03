import pandas as pd
from diffusion_kinetics.pipeline import SingleProcessPipelineConfig, PipelineOutput
from diffusion_kinetics.optimization import Dataset
from typing import Union
from  diffusion_kinetics.optimization import DiffusionOptimizer
from diffusion_kinetics.pipeline.base_pipeline import BasePipeline
import numpy as np
from diffusion_kinetics.utils.kinetics_dataframe import KineticsDataframe
from tabulate import tabulate


# hide constraint warning, since it's not relevant
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

class SinglePipeline(BasePipeline):
    def __init__(
        self,
        dataset: Dataset,
        output: PipelineOutput = None,
    ):
        self.dataset = SinglePipeline._load_dataset(dataset)
        self.optimizer = DiffusionOptimizer(self.dataset)
        self.output = SinglePipeline._create_output(output)

    def run(self, config: Union[str, dict, SingleProcessPipelineConfig]):
        """
        Run the pipeline.
        """
        config = SinglePipeline._load_config(config)
        misfits = []
        results = []
        
        # run the optimizer multiple times
        seed = config.seed if config.seed is not None else None
        for i in range(config.repeat_iterations):
            res = self.optimizer.run(config, seed=seed)
            misfits.append(res.fun)
            results.append(res)
            if seed is not None:
                seed += 1
            print(f"Finished iteration {i+1} of {config.repeat_iterations}. misfit: {res.fun}, iters: {res.nit}")
            
        # get the best result
        index = np.argmin(misfits)
        res = results[index]
        
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

        x = res["x"]
        df = KineticsDataframe(res, config).df
        df.drop("misfit", axis=1, inplace=True)
        df = df.T
        # add the indices as the first column
        df.insert(0, "index", df.index)
        print("Best Result:")
        print(f"{tabulate(df, tablefmt='fancy_grid', numalign='right', showindex=False)}")
        return res
    
    @staticmethod
    def _load_config(config:Union[str, dict, SingleProcessPipelineConfig]):
        if isinstance(config, str):
            config = SingleProcessPipelineConfig.load(config)
        elif isinstance(config, dict):
            config = SingleProcessPipelineConfig(**config)
        elif config == None:
            config = SingleProcessPipelineConfig()
        elif not isinstance(config, SingleProcessPipelineConfig):
            raise ValueError(f"config must be a path to a yaml file, a dictionary, or a SingleProcessPipelineConfig object. Got: {config.__class__.__name__}")
        return config
    
    @staticmethod
    def _load_dataset(dataset:Union[str, pd.DataFrame, Dataset]):
        if isinstance(dataset, str):
            dataset = Dataset(pd.read_csv(dataset))
        elif isinstance(dataset, pd.DataFrame):
            dataset = Dataset(dataset)
        elif not isinstance(dataset, Dataset):
            raise ValueError(f"dataset must be a path to a csv file, a pandas dataframe, or a Dataset object. Got: {dataset.__class__.__name__}")
        return dataset

    @staticmethod
    def _create_output(output:Union[str, PipelineOutput]):
        if isinstance(output, PipelineOutput):
            pass
        elif isinstance(output, str):
            output = PipelineOutput(output)
        elif isinstance(output, type(None)):
            output = None
        else:
            raise ValueError(f"output must be a path to a directory or a PipelineOutput object. Got: {output.__class__.__name__}")
        return output
