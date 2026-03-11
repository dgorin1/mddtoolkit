import numpy as np
import pandas as pd
from diffusion_kinetics.pipeline.pipeline_config import SingleProcessPipelineConfig
from diffusion_kinetics.utils.organize_x import organize_x


class KineticsDataframe:
    """Formats optimization results as a tidy DataFrame for a single misfit statistic
    and domain count.

    Args:
        res (dict): Result dictionary returned by the optimizer.
        config (SingleProcessPipelineConfig): Configuration used for the optimization run.
    """

    def __init__(self, res: dict, config: SingleProcessPipelineConfig):
        self._res = res
        self._config = config
        self._create_dataframe()

    @property
    def df(self):
        return self._df

    def to_csv(self, path: str):
        self.df.to_csv(path)

    def _create_dataframe(self):
        col_names = self._create_column_names()
        x = np.concatenate(
            (organize_x(self._res["x"], chop_fracs=False), np.array([self._res["fun"]]))
        )
        self._df = pd.DataFrame(x.reshape(-1, len(x)).tolist(), columns=col_names)

    def _create_column_names(self):
        # chisq optimises total_moles as an extra free parameter
        column_names = ["total_moles"] if self._config.misfit_stat == "chisq" else []
        column_names.append("Ea_(kJ/mol)")

        lnd0aa_cols = [f"ln(D0/a^2)_{d}" for d in range(1, self._config.num_domains + 1)]
        frac_cols = [f"frac_{d}" for d in range(1, self._config.num_domains + 1)]

        return column_names + lnd0aa_cols + frac_cols + ["misfit"]
