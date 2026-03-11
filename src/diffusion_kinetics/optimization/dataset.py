import pandas as pd
import torch
import warnings


class Dataset(pd.DataFrame):
    """A pandas DataFrame subclass that validates and caches the columns
    required by the MDD optimization.

    Expected columns:
        - ``TC``: Step temperature (°C).
        - ``thr``: Step duration (hours).
        - ``ln(D/a^2)``: Experimental ln(D/a²) values.
        - ``Fi``: Cumulative fractional gas release.
        - ``delM``: Uncertainty on gas moles released per step.

    Args:
        data (pd.DataFrame): Raw input DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        super().__init__(data=data, index=data.index, columns=data.columns)

        required = ["TC", "thr", "ln(D/a^2)", "Fi", "delM"]
        missing = [col for col in required if col not in self.columns]
        assert not missing, f"Dataset is missing required columns: {missing}"

        # Cache torch tensors for the columns accessed most heavily during
        # optimization. The UserWarning suppression is needed because
        # pd.DataFrame raises a warning when setting custom attributes on a
        # DataFrame subclass.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._tc = torch.tensor(self["TC"].values)
            self._thr = torch.tensor(self["thr"].values)
            self._ln_daa = torch.tensor(self["ln(D/a^2)"].values)
            self._fi_exp = torch.tensor(self["Fi"].values)
            self.uncert = torch.tensor(self["delM"].values)
