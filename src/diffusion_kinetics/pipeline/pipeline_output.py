import os
import pandas as pd
from diffusion_kinetics.optimization import Dataset
from diffusion_kinetics.utils.plot_results import plot_results
from diffusion_kinetics.utils.organize_x import organize_x
from diffusion_kinetics.optimization import DiffusionObjective
from diffusion_kinetics.pipeline import SingleProcessPipelineConfig
import numpy as np
import json

class PipelineOutput:
    """Class to handle the construction and storage of pipeline output.
    """
    def __init__(self, output_dir:str):
        self.output_dir = output_dir
        self.config_path = os.path.join(output_dir, "config.yaml")
        self._setup()
        
    def _setup(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def get_plot_path(self, config:SingleProcessPipelineConfig, file_type:str="pdf"):
        """Get the path to a plot file.

        Args:
            ndom (int): Number of domains.
            file_type (str, optional): File type. Defaults to "pdf".

        Returns:
            str: Path to plot file.
        """
        return os.path.join(self.output_dir, config.misfit_stat, f"{config.num_domains}_dom_best_params.{file_type}")
    
    def get_results_path(self, config:SingleProcessPipelineConfig, file_type:str="json"):
        """
        Args:
            ndom (int): Number of domains.
            file_type (str, optional): File type. Defaults to "csv".

        Returns:
            str: Path to results file.
        """
        return os.path.join(self.output_dir, config.misfit_stat, f"{config.num_domains}_dom_optimizer_output.{file_type}")
    
    def get_dataframe_path(self, misfit_type:str, file_type:str="csv"):
        return os.path.join(self.output_dir, misfit_type, f"combined_results_{misfit_type}.{file_type}")
    
    def get_generated_input_path(self, input_filename:str, file_type:str="csv"):
        return os.path.join(self.output_dir, f"input_{input_filename}.{file_type}")
    
    def serialize_results(self, results:dict, config:dict):
        """serialize the results to a json friendly format

        Args:
            results (dict): _description_
            config (dict): _description_
            dataset (Dataset): _description_
        """

        # serialize results
        serialized_results = {
            "fun": results.fun,
            "message": results.message,
            "nfev": results.nfev,
            "nit": results.nit,
            "success": results.success,
            "x": results.x.tolist()
        }
        
        serialized_config = config.serialize()
        
        return {
            "results": serialized_results,
            "config": serialized_config
        }
    
    def save_results(self, results:dict, config:dict, dataset:Dataset, quiet:bool=True):
        """Save the results to a csv file.

        Args:
            results (dict): Results dictionary.
            config (dict): Config dictionary.
            dataset (Dataset): Dataset object.
        """
        if not os.path.exists(os.path.join(self.output_dir, config.misfit_stat)):
            os.makedirs(os.path.join(self.output_dir, config.misfit_stat))
        results_path = self.get_results_path(config)
        res = self.serialize_results(results, config)
        json.dump(res, open(results_path, "w"), indent=4)
        
        objective = DiffusionObjective(
            dataset, 
            config.time_add,
            config.temp_add,
            config.omit_value_indices,
            config.misfit_stat,
            config.geometry,
            config.punish_degas_early
        )

        plot_results(
            organize_x(results.x),
            dataset,
            objective,
            self.get_plot_path(config),
        )
    