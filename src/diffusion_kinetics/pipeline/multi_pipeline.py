import pandas as pd
from diffusion_kinetics.pipeline import MultiProcessPipelineConfig, PipelineOutput
from diffusion_kinetics.pipeline.single_pipeline import SinglePipeline
from diffusion_kinetics.optimization import Dataset
from diffusion_kinetics.pipeline.base_pipeline import BasePipeline
from diffusion_kinetics.utils.kinetics_dataframe import KineticsDataframe
from typing import Union
import numpy as np

class MultiPipeline(BasePipeline):
    def __init__(
        self,
        dataset:Union[str, pd.DataFrame, Dataset],
        output:Union[str, PipelineOutput]=None,
    ):
        self.dataset = MultiPipeline._load_dataset(dataset)
        self.output = MultiPipeline._create_output(output)
    
    def run(self, config:Union[str, dict, MultiProcessPipelineConfig]):
        results = []
        combined_df = None
        config = MultiPipeline._load_config(config)
        pipeline = SinglePipeline(self.dataset, output=self.output)
        for misfit_type in config.single_pipeline_configs.keys():
            print(f"{'='*80}", "\n\033[1mRunning pipeline for misfit type:", misfit_type, "\033[0m\n")
            configs_for_each_domain_list = config.single_pipeline_configs[misfit_type]
            for single_pipeline_config in configs_for_each_domain_list:
                print(f"Running pipeline with {single_pipeline_config.num_domains} domains")
                res = pipeline.run(single_pipeline_config)
                # save the combined csv
                if combined_df is None:
                    combined_df = KineticsDataframe(res, single_pipeline_config).df
                else:
                    combined_df = self.combine_dfs(combined_df, KineticsDataframe(res, single_pipeline_config).df)
                combined_df.to_csv(self.output.get_dataframe_path(misfit_type))

                results.append(res)
                print("")
        return results
    
    @staticmethod
    def _load_config(config:Union[str, dict, MultiProcessPipelineConfig]):
        if isinstance(config, str):
            config = MultiProcessPipelineConfig.load(config)
        elif isinstance(config, dict):
            config = MultiProcessPipelineConfig(**config)
        elif config == None:
            config = MultiProcessPipelineConfig()
        elif not isinstance(config, MultiProcessPipelineConfig):
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
        else:
            raise ValueError(f"output must be a path to a directory or a PipelineOutput object. Got: {output.__class__.__name__}")
        return output
    
        
    @staticmethod
    def combine_dfs(d1, d2):
        col_names = d1.columns if len(d1.columns) > len(d2.columns) else d2.columns
        df_dict = {}
        for col in col_names:
            if col not in df_dict.keys():
                df_dict[col] = []
            if col in d1:
                df_dict[col] = df_dict[col] + d1[col].tolist()
            else:
                df_dict[col] = df_dict[col] + [None for _ in range(len(d1))]
            if col in d2:
                df_dict[col] = df_dict[col] + d2[col].tolist()
            else:
                df_dict[col] = df_dict[col] + [None for _ in range(len(d2))]
        return pd.DataFrame.from_dict(df_dict)
