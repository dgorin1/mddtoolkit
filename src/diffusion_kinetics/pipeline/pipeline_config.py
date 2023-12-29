from typing import Union
import yaml
import torch


# hardcoded list of possible misfit stats...
# TODO: make this be dynamically populated based on the available misfit stat functions.
MISFIT_STAT_LIST = [
    "chisq",
    "l2_moles",
    "l1_moles",
    "l1_frac",
    "l1_frac_cum",
    "percent_frac",
    "lnd0aa",
    "lnd0aa_chisq", 
    "l2_frac"
]
    
    
class BasePipelineConfig:
    """
    A base class for pipeline configurations.
    """
    
    @classmethod
    def load(cls, config:Union[str, dict]):
        """ load the config from a yaml file """
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config)
    
    def __getitem__(self, key):
        """ get an attribute of the class """
        return getattr(self, key)
    
    def __repr__(self):
        # for each attribute of the class, if it's not a function, add it to the string
        string = f"{self.__class__.__name__} : (\n"
        for attr in dir(self):
            if not attr.startswith("_") and not callable(getattr(self, attr)):
                string += f"    {attr}: {getattr(self, attr)}\n"
        string += ")"
        return string
    

class SingleProcessPipelineConfig(BasePipelineConfig):
    """
    A class to hold the configuration for a single-processing pipeline.
    
    Args:
        - lnD0aa_bounds (list[float]): The bounds for the lnD0aa parameter. Defaults to [-5.0,50.0].
        - Ea_bounds (list[float]): The bounds for the Ea parameter. Defaults to [50.0,500.0].
        - geometry (str): The geometry of the model. Defaults to "spherical".
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
        num_domains:int=1,
        misfit_stat:str="lnd0aa_chisq",
        geometry:str="spherical",
        omit_value_indices:list[int]=[],
        max_iters:Union[float,int]=100000,
        punish_degas_early:bool=True,
        repeat_iterations:int=10,
        seed:int=0,
    ):
        self.lnd0aa_bounds = lnd0aa_bounds
        self.ea_bounds = ea_bounds
        self.time_add = torch.tensor(time_add)
        self.temp_add = torch.tensor(temp_add)
        self.geometry = geometry
        self.omit_value_indices = omit_value_indices
        self.max_iters = max_iters
        self.punish_degas_early = punish_degas_early
        self.num_domains = num_domains
        self.misfit_stat = misfit_stat
        self.repeat_iterations = repeat_iterations
        self.seed = seed
        self._assert_is_valid()
        
    def _assert_is_valid(self):
        """ assert that the config is valid """
        # assert all the types are valid
        assert isinstance(self.lnd0aa_bounds,list), "lnd0aa_bounds must be a list"
        assert isinstance(self.ea_bounds,list), "ea_bounds must be a list"
        assert isinstance(self.time_add,torch.Tensor), "time_add must be a torch.tensor"
        assert isinstance(self.temp_add,torch.Tensor), "temp_add must be a torch.tensor"
        assert isinstance(self.geometry,str), "geometry must be a string"
        assert isinstance(self.omit_value_indices,list), "omit_value_indices must be a list"
        assert isinstance(self.max_iters,(float,int)), "max_iters must be a float or int"
        assert isinstance(self.punish_degas_early,bool), "punish_degas_early must be a bool"
        assert len(self.lnd0aa_bounds) == 2, "lnd0aa_bounds must be a list of length 2"
        assert len(self.ea_bounds) == 2, "ea_bounds must be a list of length 2"
        assert self.lnd0aa_bounds[0] < self.lnd0aa_bounds[1], "lnd0aa_bounds must be in increasing order"
        assert self.ea_bounds[0] < self.ea_bounds[1], "ea_bounds must be in increasing order"        
        assert self.geometry in ["spherical","plane_sheet"], "geometry must be either 'spherical' or 'plane_sheet'"
        assert self.max_iters > 0, "max_iters must be greater than 0"  
        assert self.num_domains >= 0, "num_domains must be greater than 0"
        assert self.misfit_stat in MISFIT_STAT_LIST, f"misfit_stat must be a valid misfit statistic. got {self.misfit_stat}"  
        assert self.repeat_iterations > 0, "repeat_iterations must be greater than 0"
        assert self.repeat_iterations == int(self.repeat_iterations), "repeat_iterations must be an integer"
        assert self.seed == int(self.seed), "seed must be an integer"
    
    def to_dict(self):
        """ convert the config to a dictionary """
        return {
            "lnd0aa_bounds":self.lnd0aa_bounds,
            "ea_bounds":self.ea_bounds,
            "time_add":self.time_add.tolist(),
            "temp_add":self.temp_add.tolist(),
            "geometry":self.geometry,
            "omit_value_indices":self.omit_value_indices,
            "max_iters":self.max_iters,
            "punish_degas_early":self.punish_degas_early,
            "num_domains":self.num_domains,
            "misfit_stat":self.misfit_stat,
        }
      
        
class MultiProcessPipelineConfig(BasePipelineConfig):
    """
    A class to hold the configuration for a multi-processing pipeline.
    
    Args:
        - max_domains_to_model (int): The maximum number of domains to model. Defaults to 8.
        - misfit_stat_list (list[str]): The list of misfit statistics to use. Defaults to MISFIT_STAT_LIST.
        - lnD0aa_bounds (list[float]): The bounds for the lnD0aa parameter. Defaults to [-5.0,50.0].
        - Ea_bounds (list[float]): The bounds for the Ea parameter. Defaults to [50.0,500.0].
        - time_add (list[float]): The times to add to the dataset. Defaults to [].
        - temp_add (list[float]): The temperatures to add to the dataset. Defaults to [].
        - geometry (str): The geometry of the model. Defaults to "spherical".
        - omit_value_indices (list[int]): The indices of the values to omit from the dataset. Defaults to [].
        - max_iters (float): The maximum number of iterations to run. Defaults to 100000.
        - punish_degas_early (bool): Whether to punish degassing early. Defaults to True.
    """
    
    def __init__(
        self, 
        max_domains_to_model:int=8, 
        misfit_stat_list:list[str]=MISFIT_STAT_LIST,
        lnd0aa_bounds:list[Union[float,int],Union[float,int]]=[-5.0,50.0],
        ea_bounds:list[Union[float,int],Union[float,int]]=[50.0,500.0],
        time_add:list[Union[float,int]]=[],
        temp_add:list[Union[float,int]]=[],
        geometry:str="spherical",
        omit_value_indices:list[int]=[],
        max_iters:Union[float,int]=100000,
        punish_degas_early:bool=True,
    ):
        self.max_domains_to_model = max_domains_to_model
        self.misfit_stat_list = misfit_stat_list
        
        self.single_pipeline_configs = {}
        for stat in self.misfit_stat_list:
            self.single_pipeline_configs[stat] = []
            for i in range(1, self.max_domains_to_model):
                self.single_pipeline_configs[stat].append(
                    SingleProcessPipelineConfig(
                        lnd0aa_bounds=lnd0aa_bounds,
                        ea_bounds=ea_bounds,
                        time_add=time_add,
                        temp_add=temp_add,
                        num_domains=i,
                        misfit_stat=stat,
                        geometry=geometry,
                        omit_value_indices=omit_value_indices,
                        max_iters=max_iters,
                        punish_degas_early=punish_degas_early,
                    )
                )
        self._assert_is_valid()
                        
    def _assert_is_valid(self):
        assert isinstance(self.max_domains_to_model,int), "max_domains_to_model must be an int"
        assert isinstance(self.misfit_stat_list,list), "misfit_stat_list must be a list"
        assert all([isinstance(stat,str) for stat in self.misfit_stat_list]), "misfit_stat_list must be a list of strings"
        assert all([stat.lower() in MISFIT_STAT_LIST for stat in self.misfit_stat_list]), "misfit_stat_list must be a list of valid misfit statistics"
        assert len(set(self.misfit_stat_list)) == len(self.misfit_stat_list), "misfit_stat_list must not contain duplicate values"
        assert len(self.misfit_stat_list) > 0, "misfit_stat_list must contain at least one misfit statistic"
        assert self.max_domains_to_model >= 0, "max_domains_to_model must be greater than 0"
