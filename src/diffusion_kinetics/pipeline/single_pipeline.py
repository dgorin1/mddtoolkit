import pandas as pd
from diffusion_kinetics.pipeline import SingleProcessPipelineConfig, PipelineOutput
from diffusion_kinetics.optimization.dataset import Dataset
from typing import Union
from diffusion_kinetics.pipeline.base_pipeline import BasePipeline
import numpy as np
from diffusion_kinetics.utils.kinetics_dataframe import KineticsDataframe
from tabulate import tabulate


# Suppress scipy's quasi-Newton delta_grad warning; it is not actionable here.
import warnings
warnings.filterwarnings(
    "ignore",
    message="delta_grad == 0.0. Check if the approximated function is linear.",
)


class SinglePipeline(BasePipeline):
    """Runs a single-configuration MDD optimization, repeating it
    ``config.repeat_iterations`` times and keeping the best result.
    """

    def __init__(self, dataset: Dataset, output: PipelineOutput = None):
        from diffusion_kinetics.optimization.diffusion_optimizer import DiffusionOptimizer
        self.dataset = SinglePipeline._load_dataset(dataset)
        self.optimizer = DiffusionOptimizer(self.dataset)
        self.output = SinglePipeline._create_output(output)

    def run(self, config: Union[str, dict, SingleProcessPipelineConfig]):
        """Run the optimization and return the best result.

        Args:
            config: Path to a YAML config file, a plain dict, or a
                :class:`SingleProcessPipelineConfig` instance.

        Returns:
            dict: Best result with keys ``x``, ``fun``, ``success``,
            ``message``, ``nit``, ``nfev``.
        """
        config = SinglePipeline._load_config(config)
        misfits = []
        results = []

        seed = config.seed
        for i in range(config.repeat_iterations):
            res = self.optimizer.run(config, seed=seed)
            misfits.append(res.fun)
            results.append(res)
            if seed is not None:
                seed += 1
            print(
                f"Finished iteration {i+1} of {config.repeat_iterations}. "
                f"misfit: {res.fun}, iters: {res.nit}"
            )

        res = results[np.argmin(misfits)]

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

        # Pretty-print the best parameters
        df = KineticsDataframe(res, config).df.drop("misfit", axis=1).T
        df.insert(0, "index", df.index)
        print("Best Result:")
        print(tabulate(df, tablefmt="fancy_grid", numalign="right", showindex=False))
        return res

    @staticmethod
    def _load_config(config: Union[str, dict, SingleProcessPipelineConfig]):
        if isinstance(config, str):
            return SingleProcessPipelineConfig.load(config)
        elif isinstance(config, dict):
            return SingleProcessPipelineConfig(**config)
        elif config is None:
            return SingleProcessPipelineConfig()
        elif isinstance(config, SingleProcessPipelineConfig):
            return config
        else:
            raise ValueError(
                f"config must be a YAML path, dict, or SingleProcessPipelineConfig. "
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
    def _create_output(output: Union[str, PipelineOutput, None]):
        if isinstance(output, PipelineOutput):
            return output
        elif isinstance(output, str):
            return PipelineOutput(output)
        elif output is None:
            return None
        else:
            raise ValueError(
                f"output must be a directory path or PipelineOutput. "
                f"Got: {output.__class__.__name__}"
            )
