import numpy as np
import pandas as pd
from diffusion_kinetics.pipeline.pipeline_config import SingleProcessPipelineConfig
from diffusion_kinetics.utils.organize_x import organize_x

class KineticsDataframe:
    """a dataframe that contains the results of the optimization for a single misfit statistic and number of domains.
    
    Args:
    ------
        - res (dict): The result of the optimization.
        - config (SingleProcessPipelineConfig): The configuration for the optimization.
    """
    def __init__(self, res:dict, config:SingleProcessPipelineConfig):
        self._res = res
        self._config = config
        self.dataframe = self._create_dataframe()

    @property
    def df(self):
        return self._df
    
    def to_csv(self, path:str):
        self.df.to_csv(path)

    def _create_dataframe(self):
        col_names = self._create_column_names()
        x = np.concatenate((organize_x(self._res["x"], chop_fracs=False), np.array([self._res["fun"]])))
        self._df = pd.DataFrame(x.reshape(-1, len(x)).tolist(), columns=col_names)
    
    def _create_column_names(self):
        column_names = []
        if self._config.misfit_stat in ["chisq", "l1_moles", "l2_moles"]:
            column_names.append("total_moles")
        column_names.append("Ea_(kJ/mol)")
        
        lnd0aa = []
        fracs = []
        for domain in range(1, self._config.num_domains + 1):
            lnd0aa.append(f"ln(D0/a^2)_{domain}")
            fracs.append(f"frac_{domain}")

        return column_names + lnd0aa + fracs + ["misfit"]

        