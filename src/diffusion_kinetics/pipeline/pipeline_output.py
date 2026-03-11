import os
import json
import pandas as pd
from diffusion_kinetics.optimization import Dataset, DiffusionObjective
from diffusion_kinetics.utils.plot_results import plot_results
from diffusion_kinetics.utils.organize_x import organize_x
from diffusion_kinetics.pipeline import SingleProcessPipelineConfig


class PipelineOutput:
    """Manages the output directory structure for a pipeline run.

    All results (plots, JSON optimizer output, combined CSVs, and the
    pre-processed input dataset) are written under ``output_dir``, organised
    by misfit statistic.

    Args:
        output_dir (str): Root directory for all output files. Created
            automatically if it does not exist.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._setup()

    def _setup(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def get_plot_path(self, config: SingleProcessPipelineConfig, file_type: str = "pdf") -> str:
        """Return the path for the diagnostic plot for a given config.

        Args:
            config (SingleProcessPipelineConfig): Config whose ``misfit_stat``
                and ``num_domains`` determine the file name.
            file_type (str): File extension. Defaults to ``"pdf"``.
        """
        return os.path.join(
            self.output_dir, config.misfit_stat,
            f"{config.num_domains}_dom_best_params.{file_type}",
        )

    def get_results_path(self, config: SingleProcessPipelineConfig, file_type: str = "json") -> str:
        """Return the path for the raw optimizer output for a given config.

        Args:
            config (SingleProcessPipelineConfig): Config whose ``misfit_stat``
                and ``num_domains`` determine the file name.
            file_type (str): File extension. Defaults to ``"json"``.
        """
        return os.path.join(
            self.output_dir, config.misfit_stat,
            f"{config.num_domains}_dom_optimizer_output.{file_type}",
        )

    def get_dataframe_path(self, misfit_type: str, file_type: str = "csv") -> str:
        """Return the path for the combined results CSV for a misfit statistic."""
        return os.path.join(
            self.output_dir, misfit_type, f"combined_results_{misfit_type}.{file_type}"
        )

    def get_generated_input_path(self, input_filename: str, file_type: str = "csv") -> str:
        """Return the path at which the pre-processed input dataset is saved."""
        return os.path.join(self.output_dir, f"input_{input_filename}.{file_type}")

    def serialize_results(self, results, config: SingleProcessPipelineConfig) -> dict:
        """Serialise optimizer results and config to a JSON-compatible dict.

        Args:
            results: scipy ``OptimizeResult`` object.
            config (SingleProcessPipelineConfig): Configuration used for the run.

        Returns:
            dict: ``{"results": {...}, "config": {...}}``.
        """
        return {
            "results": {
                "fun": results.fun,
                "message": results.message,
                "nfev": results.nfev,
                "nit": results.nit,
                "success": results.success,
                "x": results.x.tolist(),
            },
            "config": config.serialize(),
        }

    def save_results(self, results, config: SingleProcessPipelineConfig, dataset: Dataset):
        """Save the optimizer output, config, and diagnostic plot to disk.

        Args:
            results: scipy ``OptimizeResult`` object.
            config (SingleProcessPipelineConfig): Configuration used for the run.
            dataset (Dataset): Dataset used for the run.
        """
        stat_dir = os.path.join(self.output_dir, config.misfit_stat)
        os.makedirs(stat_dir, exist_ok=True)

        json.dump(
            self.serialize_results(results, config),
            open(self.get_results_path(config), "w"),
            indent=4,
        )

        objective = DiffusionObjective(
            dataset,
            config.omit_value_indices,
            config.misfit_stat,
            config.geometry,
            config.punish_degas_early,
        )
        plot_results(organize_x(results.x), dataset, objective, self.get_plot_path(config))
