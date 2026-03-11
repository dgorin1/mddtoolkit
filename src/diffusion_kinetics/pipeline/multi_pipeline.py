import pandas as pd
from diffusion_kinetics.pipeline import MultiProcessPipelineConfig, PipelineOutput
from diffusion_kinetics.pipeline.single_pipeline import SinglePipeline
from diffusion_kinetics.optimization import Dataset
from diffusion_kinetics.pipeline.base_pipeline import BasePipeline
from diffusion_kinetics.utils.kinetics_dataframe import KineticsDataframe
from typing import Union
from diffusion_kinetics.preprocessing.generate_inputs import generate_inputs
from pathlib import Path


class MultiPipeline(BasePipeline):
    """Orchestrates MDD fitting across multiple domain counts and misfit statistics.

    For each combination of misfit statistic and domain count specified in the
    config, a :class:`SinglePipeline` run is performed and its result is appended
    to a cumulative CSV file under the output directory.
    """

    def __init__(self, output: Union[str, PipelineOutput] = None):
        self.output = MultiPipeline._create_output(output)

    def run(self, config: Union[str, dict, MultiProcessPipelineConfig], dataset: str):
        """Run the full multi-domain, multi-statistic optimization.

        Args:
            config: Path to a YAML config file, a plain dict, or a
                :class:`MultiProcessPipelineConfig` instance.
            dataset (str): Path to the raw input CSV file.

        Returns:
            list: All optimizer result objects, in the order they were produced.
        """
        results = []
        config = MultiPipeline._load_config(config)

        filename = Path(dataset).stem
        input_dataset = generate_inputs(
            dataset, self.output.get_generated_input_path(filename), config.geometry
        )
        self.dataset = MultiPipeline._load_dataset(input_dataset)

        pipeline = SinglePipeline(self.dataset, output=self.output)
        print("\n\033[1m\033[4mRunning multi pipeline with config:\033[0m")
        print(config, "\n")

        for misfit_type, configs_for_domain in config.single_pipeline_configs.items():
            combined_df = None
            print(
                f"{'='*80}",
                f"\n\033[1mRunning pipeline for misfit type: {misfit_type}\033[0m",
                f"\n{'='*80}",
            )
            for single_config in configs_for_domain:
                print(f"\n\033[1m\033[4mFitting model with {single_config.num_domains} domains\033[0m")
                res = pipeline.run(single_config)
                kdf = KineticsDataframe(res, single_config).df
                combined_df = kdf if combined_df is None else self._combine_dfs(combined_df, kdf)
                combined_df.to_csv(self.output.get_dataframe_path(misfit_type))
                results.append(res)
                print("")

        return results

    @staticmethod
    def _load_config(config: Union[str, dict, MultiProcessPipelineConfig]):
        if isinstance(config, str):
            return MultiProcessPipelineConfig.load(config)
        elif isinstance(config, dict):
            return MultiProcessPipelineConfig(**config)
        elif config is None:
            return MultiProcessPipelineConfig()
        elif isinstance(config, MultiProcessPipelineConfig):
            return config
        else:
            raise ValueError(
                f"config must be a YAML path, dict, or MultiProcessPipelineConfig. "
                f"Got: {config.__class__.__name__}"
            )

    @staticmethod
    def _load_dataset(dataset: Union[str, pd.DataFrame, Dataset]):
        if isinstance(dataset, str):
            return Dataset(pd.read_csv(dataset))
        elif isinstance(dataset, pd.DataFrame):
            return Dataset(dataset)
        elif isinstance(dataset, Dataset):
            return dataset
        else:
            raise ValueError(
                f"dataset must be a CSV path, DataFrame, or Dataset. "
                f"Got: {dataset.__class__.__name__}"
            )

    @staticmethod
    def _create_output(output: Union[str, PipelineOutput]):
        if isinstance(output, PipelineOutput):
            return output
        elif isinstance(output, str):
            return PipelineOutput(output)
        else:
            raise ValueError(
                f"output must be a directory path or PipelineOutput. "
                f"Got: {output.__class__.__name__}"
            )

    @staticmethod
    def _combine_dfs(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
        """Concatenate two result DataFrames that may have different column sets.

        The wider DataFrame's column order is used so that results are always
        presented with all lnD0/a² columns first, followed by all fraction
        columns — matching the order produced by the highest-domain-count run.
        Columns missing from either frame are filled with NaN.
        """
        col_order = d2.columns if len(d2.columns) >= len(d1.columns) else d1.columns
        return pd.concat([d1, d2], ignore_index=True)[col_order]
