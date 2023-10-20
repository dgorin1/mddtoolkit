from typing import Union

class PipelineConfig:
    """
    A class to hold the configuration for the pipeline.
    
    Args:
        - lnD0aa_bounds (list[float]): The bounds for the lnD0aa parameter. Defaults to [-5.0,50.0].
        - Ea_bounds (list[float]): The bounds for the Ea parameter. Defaults to [50.0,500.0].
        - max_domains_to_model (float): The maximum number of domains to model. Defaults to 8.0.
        - geometry (str): The geometry of the model. Defaults to "spherical".
        - misfit_stat_list (list[str]): The misfit statistics to use. Defaults to [
            "lnd0aa_chisq",
            "chisq",
            "percent_frac",
            "l1_frac_cum",
            "l1_frac",
            "l1_moles",
            "l2_moles",
            "l2_frac",
            "lnd0aa",
        ].
        - max_iters (float): The maximum number of iterations to run. Defaults to 100000.
        - iteration_repeats (float): The number of times to repeat each iteration. Defaults to 10.
        - punish_degas_early (bool): Whether to punish degassing early. Defaults to True.
    """
    def __init__(
        self, 
        lnd0aa_bounds:list[Union[float,int],Union[float,int]]=[-5.0,50.0],
        ea_bounds:list[Union[float,int],Union[float,int]]=[50.0,500.0],
        time_add:list[Union[float,int]]=[],
        temp_add:list[Union[float,int]]=[],
        max_domains_to_model:int=8,
        geometry:str="spherical",
        omit_value_indices:list[int]=[],
        misfit_stat_list:list[str]=[
            "lnd0aa_chisq",
            "chisq",
            "percent_frac",
            "l1_frac_cum",
            "l1_frac",
            "l1_moles",
            "l2_moles",
            "l2_frac",
            "lnd0aa",
        ],
        max_iters:Union[float,int]=100000,
        iteration_repeats:Union[float,int]=10,
        punish_degas_early:bool=True,
    ):
        self.lnd0aa_bounds = lnd0aa_bounds
        self.ea_bounds = ea_bounds
        self.time_add = time_add
        self.temp_add = temp_add
        self.max_domains_to_model = max_domains_to_model
        self.geometry = geometry
        self.omit_value_indices = omit_value_indices
        self.misfit_stat_list = misfit_stat_list
        self.max_iters = max_iters
        self.iteration_repeats = iteration_repeats
        self.punish_degas_early = punish_degas_early
        self._assert_is_valid()
        
    
    def _assert_is_valid(self):
        # assert all the types are valid
        assert isinstance(self.lnd0aa_bounds,list), "lnd0aa_bounds must be a list"
        assert isinstance(self.ea_bounds,list), "ea_bounds must be a list"
        assert isinstance(self.time_add,list), "time_add must be a list"
        assert isinstance(self.temp_add,list), "temp_add must be a list"
        assert isinstance(self.max_domains_to_model,int), "max_domains_to_model must be an int"
        assert isinstance(self.geometry,str), "geometry must be a string"
        assert isinstance(self.omit_value_indices,list), "omit_value_indices must be a list"
        assert isinstance(self.misfit_stat_list,list), "misfit_stat_list must be a list"
        assert isinstance(self.max_iters,(float,int)), "max_iters must be a float or int"
        assert isinstance(self.iteration_repeats,(float,int)), "iteration_repeats must be a float or int"
        assert isinstance(self.punish_degas_early,bool), "punish_degas_early must be a bool"
        
        # assert all values are valid
        assert len(self.lnd0aa_bounds) == 2, "lnd0aa_bounds must be a list of length 2"
        assert len(self.ea_bounds) == 2, "ea_bounds must be a list of length 2"
        assert self.lnd0aa_bounds[0] < self.lnd0aa_bounds[1], "lnd0aa_bounds must be in increasing order"
        assert self.ea_bounds[0] < self.ea_bounds[1], "ea_bounds must be in increasing order"
        assert all([isinstance(val,(float,int)) for val in self.time_add]), "time_add must be a list of floats or ints"
        assert all([isinstance(val,(float,int)) for val in self.temp_add]), "temp_add must be a list of floats or ints"
        assert self.max_domains_to_model > 0, "max_domains_to_model must be greater than 0"
        assert self.geometry in ["spherical","plane_sheet"], "geometry must be either 'spherical' or 'plane_sheet'"
        assert all([isinstance(stat,str) for stat in self.misfit_stat_list]), "misfit_stat_list must be a list of strings"
        assert self.max_iters > 0, "max_iters must be greater than 0"
        assert self.iteration_repeats > 0, "iteration_repeats must be greater than 0"
        
    def __repr__(self):
        attrs = [
            f"    {attr}={getattr(self, attr)},\n"
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        return f"PipelineConfig(\n{''.join(attrs)})"

