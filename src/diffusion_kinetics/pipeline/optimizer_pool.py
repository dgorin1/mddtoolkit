import pathos.multiprocessing as pathos_mp
from diffusion_kinetics.optimization import Dataset
from diffusion_kinetics.pipeline import SingleProcessPipelineConfig
from diffusion_kinetics.pipeline import Pipeline

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
        # arguments = [(sp_config, self.dataset.copy(), seed) for sp_config in sp_configs]
        # results = self.pool.map(self._run_single_config, arguments)
        results = []
        for sp_config in sp_configs:
            pipeline = Pipeline(self.dataset.copy(), sp_config, output=None)
            result = pipeline.run()
            results.append(result)
        return results
