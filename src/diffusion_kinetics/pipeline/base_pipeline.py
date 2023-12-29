from diffusion_kinetics.pipeline.pipeline_config import BasePipelineConfig
from diffusion_kinetics.pipeline.pipeline_output import PipelineOutput
from typing import Union


class BasePipeline:
    """Base class for all pipelines.

    Parameters
    ----------
    data : :obj:`dict`
        Dictionary containing the data to be processed.
    """
    def __init__(self, data):
        self.data = data

    def run(self):
        """Runs the pipeline."""
        raise NotImplementedError
    
    @staticmethod
    def _load_config(config:Union[str, dict, BasePipelineConfig]):
        """Loads the configuration.

        Parameters
        ----------
        config : :obj:`str`, :obj:`dict`, :obj:`PipelineConfig`
            Configuration to be loaded.

        Returns
        -------
        :obj:`PipelineConfig`
            Loaded configuration.
        """
        raise NotImplementedError
    
    @staticmethod
    def _load_dataset(dataset:Union[str, dict, BasePipelineConfig]):
        """Loads the dataset.

        Parameters
        ----------
        dataset : :obj:`str`, :obj:`dict`, :obj:`Dataset`
            Dataset to be loaded.

        Returns
        -------
        :obj:`Dataset`
            Loaded dataset.
        """
        raise NotImplementedError
    
    @staticmethod
    def _create_output(output:Union[str, PipelineOutput]):
        """Creates the output.

        Parameters
        ----------
        output : :obj:`str`, :obj:`PipelineOutput`
            Output to be created.

        Returns
        -------
        :obj:`PipelineOutput`
            Created output.
        """
        raise NotImplementedError