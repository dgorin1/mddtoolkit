import pandas as pd
import torch
import warnings


class Dataset(pd.DataFrame):
    def __init__(self, method: str, data: pd.DataFrame):
        super().__init__(data=data, index=data.index, columns=data.columns)
        assert self["TC"] is not None, "given dataset does not contain TC parameter."
        assert self["thr"] is not None, "given dataset does not contain thr parameter."
        assert (
            self["ln(D/a^2)"] is not None
        ), "given dataset does not contain ln(D/a^2) parameter."
        assert self["Fi"] is not None, "given dataset does not contain Fi parameter."
        # temporarily ignores atribute setting warning from base dataframe class
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)


            self._TC = torch.tensor(self["TC"].values)
            self._thr = torch.tensor(self["thr"].values)
            self._lnDaa = torch.tensor(self["ln(D/a^2)"].values)
            self._Fi_exp = torch.tensor(self["Fi"].values)
            self.uncert = torch.tensor(self["delM"].values)
