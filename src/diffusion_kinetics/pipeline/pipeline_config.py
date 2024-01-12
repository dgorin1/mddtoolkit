from typing import Union
import yaml
import json
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

DEFAULT_MISFIT_STAT = [
    "chisq",
    "l1_frac",
    "l1_moles",
    "l1_frac_cum",
    "percent_frac",
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
            if not attr.startswith("_") and not callable(getattr(self, attr)) and not attr == "single_pipeline_configs":
                string += f"    {attr}: {getattr(self, attr)}\n"
        string += ")"
        return string
    
    def serialize(self):
        """ serialize the config to a dictionary """
        d = {}
        for attr in dir(self):
            if not attr.startswith("_") and not callable(getattr(self, attr)):
                d[attr] = getattr(self, attr)
                # make sure the attribute is json serializable
                try:
                    json.dumps(d[attr])
                except TypeError:
                    d[attr] = d[attr].tolist()
        return d
    

class SingleProcessPipelineConfig(BasePipelineConfig):
    """
    A class to hold the configuration for a single-processing pipeline.
    
    Args:
    -------
        - lnd0aa_bounds (list[float]): The bounds for the lnD0aa parameter. Defaults to [-5.0,50.0].
        - ea_bounds (list[float]): The bounds for the Ea parameter. Defaults to [50.0,500.0].
        - time_add (list[float]): The times to add to the dataset. Defaults to [].
        - temp_add (list[float]): The temperatures to add to the dataset. Defaults to [].
        - num_domains (int): The number of domains to use. Defaults to 1.
        - misfit_stat (str): The misfit statistic to use. Defaults to "lnd0aa_chisq".
        - geometry (str): The geometry of the model. Defaults to "spherical".
        - omit_value_indices (list[int]): The indices of the values to omit from the dataset. Defaults to [].
        - max_iters (float): The maximum number of iterations to run. Defaults to 100000.
        - punish_degas_early (bool): Whether to punish degassing early. Defaults to True.
        - repeat_iterations (int): The number of times to repeat each iteration. Defaults to 10.
        - seed (int): The seed for the optimization. Defaults to None.
        - tol (float): The tolerance for the optimization. Defaults to 0.0001.
        - popsize (int): The population size for the optimization. Defaults to 15.
        - updating (str): The updating method for the optimization. Defaults to "deferred".
        - strategy (str): The strategy for the optimization. Defaults to "best1bin".
        - mutation (float): The mutation rate for the optimization. Defaults to 0.5.
        - recombination (float): The recombination rate for the optimization. Defaults to 0.7.
        - init (str): The initialization method for the optimization. Defaults to "latinhypercube".
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
        seed:int=None,
        tol:float=0.0001,
        popsize:int=15,
        updating:str="deferred",
        strategy:str="best1bin",
        mutation:Union[float, list[float,float]]=[0.5,1.],
        recombination:float=0.7,
        init:str="latinhypercube",
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
        self.tol = tol
        self.popsize = popsize
        self.updating = updating
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
        self.init = init
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
        assert self.ea_bounds[0] <= self.ea_bounds[1], "ea_bounds must be in increasing order, or equal to one another for fixed Ea"        
        assert self.geometry in ["spherical","plane sheet"], "geometry must be either 'spherical' or 'plane sheet'"
        assert self.max_iters > 0, "max_iters must be greater than 0"  
        assert self.num_domains >= 0, "num_domains must be greater than 0"
        assert self.misfit_stat in MISFIT_STAT_LIST, f"misfit_stat must be a valid misfit statistic. got {self.misfit_stat}"  
        assert self.repeat_iterations > 0, "repeat_iterations must be greater than 0"
        assert self.repeat_iterations == int(self.repeat_iterations), "repeat_iterations must be an integer"
        assert self.seed == None or self.seed == int(self.seed), "seed must be an integer"
        assert self.tol > 0, "tol must be greater than 0"
        assert self.popsize > 0, "popsize must be greater than 0"
        assert self.updating in ["immediate","deferred"], "updating must be either 'immediate' or 'deferred'"
        assert self.strategy in ["best1bin","best1exp","rand1exp","randtobest1exp","currenttobest1exp","best2exp","rand2exp","randtobest1bin","currenttobest1bin","best2bin","rand2bin","rand1bin"], "strategy must be a valid strategy"
        assert self.recombination > 0 and self.recombination < 1, "recombination must be between 0 and 1"
        # mutation can be given as a float or a list
        if isinstance(self.mutation, float):
            assert self.mutation > 0 and self.mutation < 2, "mutation must be between 0 and 2"
        elif isinstance(self.mutation, list):
            assert len(self.mutation) == 2, "mutation must be a list of length 2"
            assert self.mutation[0] > 0 and self.mutation[0] < 2, "mutation[0] must be between 0 and 2"
            assert self.mutation[1] > 0 and self.mutation[1] < 2, "mutation[1] must be between 0 and 2"
        else:
            raise ValueError("mutation must be a float or a list")
        assert self.init in ["latinhypercube","sobol", "halton", "random"], "init must be a valid init method"
    
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
    -------
        - domains_to_model (int): The maximum number of domains to model. Defaults to 8.
        - misfit_stat_list (list[str]): The misfit statistics to use. Defaults to MISFIT_STAT_LIST.
        - lnD0aa_bounds (list[float]): The bounds for the lnD0aa parameter. Defaults to [-5.0,50.0].
        - Ea_bounds (list[float]): The bounds for the Ea parameter. Defaults to [50.0,500.0].
        - time_add (list[float]): The times to add to the dataset. Defaults to [].
        - temp_add (list[float]): The temperatures to add to the dataset. Defaults to [].
        - geometry (str): The geometry of the model. Defaults to "spherical".
        - max_iters (float): The maximum number of iterations to run. Defaults to 100000.
        - iteration_repeats (float): The number of times to repeat each iteration. Defaults to 10.
        - punish_degas_early (bool): Whether to punish degassing early. Defaults to True.
        - repeat_iterations (int): The number of times to repeat each iteration. Defaults to 10.
        - seed (int): The seed for the optimization. Defaults to None.
        - tol (float): The tolerance for the optimization. Defaults to 0.0001.
        - popsize (int): The population size for the optimization. Defaults to 15.
        - updating (str): The updating method for the optimization. Defaults to "deferred".
        - strategy (str): The strategy for the optimization. Defaults to "best1bin".
        - mutation (float): The mutation rate for the optimization. Defaults to 0.5.
        - recombination (float): The recombination rate for the optimization. Defaults to 0.7.
        - init (str): The initialization method for the optimization. Defaults to "latinhypercube".
    """
    
    def __init__(
        self, 
        domains_to_model:int=8, 
        misfit_stat_list:list[str]=DEFAULT_MISFIT_STAT,
        lnd0aa_bounds:list[Union[float,int],Union[float,int]]=[-5.0,50.0],
        ea_bounds:list[Union[float,int],Union[float,int]]=[50.0,500.0],
        time_add:list[Union[float,int]]=[],
        temp_add:list[Union[float,int]]=[],
        geometry:str="spherical",
        omit_value_indices:list[int]=[],
        max_iters:Union[float,int]=100000,
        punish_degas_early:bool=True,
        repeat_iterations:int=10,
        seed:int=None,
        tol:float=0.0001,
        popsize:int=15,
        updating:str="deferred",
        strategy:str="best1bin",
        mutation:Union[float, list[float,float]]=[0.5,1.],
        recombination:float=0.7,
        init:str="latinhypercube",
    ):
        self.domains_to_model = domains_to_model
        self.misfit_stat_list = misfit_stat_list
        self.geometry = geometry
        self.omit_value_indices = omit_value_indices
        self.max_iters = max_iters
        self.punish_degas_early = punish_degas_early
        self.repeat_iterations = repeat_iterations
        self.seed = seed
        self.tol = tol
        self.popsize = popsize
        self.updating = updating
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
        self.init = init

        self._assert_is_valid()
        
        self.single_pipeline_configs = {}
        for stat in self.misfit_stat_list:
            self.single_pipeline_configs[stat] = []
            # if max domains to model is an array, then use the first value as the min and the second as the max
            r = range(self.domains_to_model, self.domains_to_model + 1) if isinstance(self.domains_to_model, int) else range(self.domains_to_model[0], self.domains_to_model[1] + 1)
            for i in r:
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
                        repeat_iterations=repeat_iterations,
                        seed=seed,
                        tol=tol,
                        popsize=popsize,
                        updating=updating,
                        strategy=strategy,
                        mutation=mutation,
                        recombination=recombination,
                        init=init,
                    )
                )
        self._assert_is_valid()
                        
    def _assert_is_valid(self):
        # assert max domains is either an int or a list of length 2 of ints
        assert isinstance(self.domains_to_model, (int, list)), "domains_to_model must be an int or a list of ints"
        if isinstance(self.domains_to_model, list):
            assert len(self.domains_to_model) == 2, "domains_to_model must be an int or a list of length 2 of ints"
            assert isinstance(self.domains_to_model[0], int) and isinstance(self.domains_to_model[1], int), "domains_to_model must be an int or a list of length 2 of ints"
            assert self.domains_to_model[0] < self.domains_to_model[1], "domains_to_model must be in increasing order"
            # lower bound is 1
            assert self.domains_to_model[0] > 0, "domains_to_model must be greater than 0"
        assert isinstance(self.misfit_stat_list,list), "misfit_stat_list must be a list"
        assert all([isinstance(stat,str) for stat in self.misfit_stat_list]), "misfit_stat_list must be a list of strings"
        assert all([stat.lower() in MISFIT_STAT_LIST for stat in self.misfit_stat_list]), "misfit_stat_list must be a list of valid misfit statistics"
        assert len(set(self.misfit_stat_list)) == len(self.misfit_stat_list), "misfit_stat_list must not contain duplicate values"
        assert len(self.misfit_stat_list) > 0, "misfit_stat_list must contain at least one misfit statistic"
