import pathos.multiprocessing as pathos_mp
from diffusion_kinetics.optimization import Dataset
from diffusion_kinetics.pipeline import SingleProcessPipelineConfig
from diffusion_kinetics.pipeline import pipeline

class OptimizerPool:
    def __init__(self, dataset:Dataset, num_workers:int=pathos_mp.cpu_count()):
        self.num_workers = num_workers
        self.pool = pathos_mp.Pool(self.num_workers)
        self.dataset = dataset
        

    def run(self, sp_configs: list[SingleProcessPipelineConfig], seed: int = 0):
        """
        Run the optimization for a given misfit statistic and number of domains.

        Args:
            - sp_configs (list[SingleProcessPipelineConfig]): List of SingleProcessPipelineConfig instances.
            - seed (int): The random seed for the optimization.
        """
        arguments = [(sp_config, self.dataset.copy(), seed) for sp_config in sp_configs]
        results = self.pool.map(self._run_single_config, arguments)
        

    def _run_single_config(self, args):
        """
        Helper method to run the pipeline for a single configuration.

        Args:
            - sp_config (SingleProcessPipelineConfig): SingleProcessPipelineConfig instance.
        """
        # Modify the following line based on your requirements
        result = pipeline(args[0], args[1], None)
        return result
        
