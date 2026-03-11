from typing import Union
import yaml
import json


MISFIT_STAT_LIST = [
    "chisq",
    "percent_frac",
]

DEFAULT_MISFIT_STAT = [
    "chisq",
    "percent_frac",
]


class BasePipelineConfig:
    """
    A base class for pipeline configurations.
    """

    @classmethod
    def load(cls, config: Union[str, dict]):
        """load the config from a yaml file"""
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        string = f"{self.__class__.__name__} : (\n"
        for attr in dir(self):
            if not attr.startswith("_") and not callable(getattr(self, attr)) and not attr == "single_pipeline_configs":
                string += f"    {attr}: {getattr(self, attr)}\n"
        string += ")"
        return string

    def serialize(self):
        """serialize the config to a dictionary"""
        d = {}
        for attr in dir(self):
            if not attr.startswith("_") and not callable(getattr(self, attr)):
                d[attr] = getattr(self, attr)
                try:
                    json.dumps(d[attr])
                except TypeError:
                    d[attr] = d[attr].tolist()
        return d


class SingleProcessPipelineConfig(BasePipelineConfig):
    """
    Configuration for a single-domain optimization run.

    Args:
    -------
        - lnd0aa_bounds (list[float]): Bounds for lnD0aa (ln(1/s)). Defaults to [-5.0, 50.0].
        - ea_bounds (list[float]): Bounds for Ea (kJ/mol). Defaults to [50.0, 500.0].
        - num_domains (int): Number of domains to fit. Defaults to 1.
        - misfit_stat (str): Misfit statistic ("chisq" or "percent_frac"). Defaults to "chisq".
        - geometry (str): Diffusion geometry ("spherical" or "plane sheet"). Defaults to "spherical".
        - omit_value_indices (list[int]): Heating-step indices to exclude from fitting. Defaults to [].
        - max_iters (int): Maximum optimizer iterations. Defaults to 100000.
        - punish_degas_early (bool): Penalise models that exhaust gas before the experiment ends. Defaults to True.
        - repeat_iterations (int): Number of times to repeat the optimization (best result kept). Defaults to 10.
        - seed (int): Random seed for reproducibility. Defaults to None.
        - tol (float): Convergence tolerance. Defaults to 0.0001.
        - popsize (int): Differential-evolution population size. Defaults to 15.
        - updating (str): DE updating strategy ("immediate" or "deferred"). Defaults to "deferred".
        - strategy (str): DE mutation strategy. Defaults to "best1bin".
        - mutation (float | list[float]): DE mutation constant. Defaults to [0.5, 1.0].
        - recombination (float): DE recombination constant. Defaults to 0.7.
        - init (str): DE population initialisation method. Defaults to "latinhypercube".
    """

    def __init__(
        self,
        lnd0aa_bounds: list = [-5.0, 50.0],
        ea_bounds: list = [50.0, 500.0],
        num_domains: int = 1,
        misfit_stat: str = "chisq",
        geometry: str = "spherical",
        omit_value_indices: list = [],
        max_iters: Union[float, int] = 100000,
        punish_degas_early: bool = True,
        repeat_iterations: int = 10,
        seed: Union[int, None] = None,
        tol: float = 0.0001,
        popsize: int = 15,
        updating: str = "deferred",
        strategy: str = "best1bin",
        mutation: Union[float, list] = [0.5, 1.0],
        recombination: float = 0.7,
        init: str = "latinhypercube",
    ):
        self.lnd0aa_bounds = lnd0aa_bounds
        self.ea_bounds = ea_bounds
        self.num_domains = num_domains
        self.misfit_stat = misfit_stat
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

    def _assert_is_valid(self):
        if not isinstance(self.lnd0aa_bounds, list):
            raise ValueError("lnd0aa_bounds must be a list")
        if not isinstance(self.ea_bounds, list):
            raise ValueError("ea_bounds must be a list")
        if not isinstance(self.geometry, str):
            raise ValueError("geometry must be a string")
        if not isinstance(self.omit_value_indices, list):
            raise ValueError("omit_value_indices must be a list")
        if not isinstance(self.max_iters, (float, int)):
            raise ValueError("max_iters must be a float or int")
        if not isinstance(self.punish_degas_early, bool):
            raise ValueError("punish_degas_early must be a bool")
        if len(self.lnd0aa_bounds) != 2:
            raise ValueError("lnd0aa_bounds must be a list of length 2")
        if len(self.ea_bounds) != 2:
            raise ValueError("ea_bounds must be a list of length 2")
        if self.lnd0aa_bounds[0] >= self.lnd0aa_bounds[1]:
            raise ValueError("lnd0aa_bounds must be in increasing order")
        if self.ea_bounds[0] > self.ea_bounds[1]:
            raise ValueError("ea_bounds must be in increasing order, or equal for fixed Ea")
        if self.geometry not in ["spherical", "plane sheet"]:
            raise ValueError("geometry must be 'spherical' or 'plane sheet'")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be greater than 0")
        if self.num_domains < 0:
            raise ValueError("num_domains must be greater than 0")
        if self.misfit_stat not in MISFIT_STAT_LIST:
            raise ValueError(f"misfit_stat must be one of {MISFIT_STAT_LIST}, got '{self.misfit_stat}'")
        if self.repeat_iterations <= 0:
            raise ValueError("repeat_iterations must be greater than 0")
        if self.repeat_iterations != int(self.repeat_iterations):
            raise ValueError("repeat_iterations must be an integer")
        if self.seed is not None and self.seed != int(self.seed):
            raise ValueError("seed must be an integer")
        if self.tol <= 0:
            raise ValueError("tol must be greater than 0")
        if self.popsize <= 0:
            raise ValueError("popsize must be greater than 0")
        if self.updating not in ["immediate", "deferred"]:
            raise ValueError("updating must be 'immediate' or 'deferred'")
        _valid_strategies = [
            "best1bin", "best1exp", "rand1exp", "randtobest1exp", "currenttobest1exp",
            "best2exp", "rand2exp", "randtobest1bin", "currenttobest1bin",
            "best2bin", "rand2bin", "rand1bin",
        ]
        if self.strategy not in _valid_strategies:
            raise ValueError("strategy must be a valid differential-evolution strategy")
        if not (0 < self.recombination < 1):
            raise ValueError("recombination must be between 0 and 1")
        if isinstance(self.mutation, float):
            if not (0 < self.mutation < 2):
                raise ValueError("mutation must be between 0 and 2")
        elif isinstance(self.mutation, list):
            if len(self.mutation) != 2:
                raise ValueError("mutation must be a list of length 2")
            if not (0 < self.mutation[0] < 2):
                raise ValueError("mutation[0] must be between 0 and 2")
            if not (0 < self.mutation[1] < 2):
                raise ValueError("mutation[1] must be between 0 and 2")
        else:
            raise ValueError("mutation must be a float or a list")
        if self.init not in ["latinhypercube", "sobol", "halton", "random"]:
            raise ValueError("init must be a valid init method")

    def to_dict(self):
        return {
            "lnd0aa_bounds": self.lnd0aa_bounds,
            "ea_bounds": self.ea_bounds,
            "geometry": self.geometry,
            "omit_value_indices": self.omit_value_indices,
            "max_iters": self.max_iters,
            "punish_degas_early": self.punish_degas_early,
            "num_domains": self.num_domains,
            "misfit_stat": self.misfit_stat,
        }


class MultiProcessPipelineConfig(BasePipelineConfig):
    """
    Configuration for a multi-domain, multi-statistic optimization run.

    Args:
    -------
        - domains_to_model (int | list[int]): Number of domains to fit. Accepts a plain integer
          (e.g. ``4``), a single-element list (e.g. ``[4]``), or a two-element ``[min, max]``
          range that fits every count from min to max inclusive. Defaults to 8.
        - misfit_stat_list (list[str]): Misfit statistics to use. Defaults to ["chisq", "percent_frac"].
        - lnd0aa_bounds (list[float]): Bounds for lnD0aa (ln(1/s)). Defaults to [-5.0, 50.0].
        - ea_bounds (list[float]): Bounds for Ea (kJ/mol). Defaults to [50.0, 500.0].
        - geometry (str): Diffusion geometry. Defaults to "spherical".
        - omit_value_indices (list[int]): Heating-step indices to exclude. Defaults to [].
        - max_iters (int): Maximum optimizer iterations. Defaults to 100000.
        - punish_degas_early (bool): Penalise early degassing. Defaults to True.
        - repeat_iterations (int): Repeats per optimization. Defaults to 10.
        - seed (int): Random seed. Defaults to None.
        - tol (float): Convergence tolerance. Defaults to 0.0001.
        - popsize (int): DE population size. Defaults to 15.
        - updating (str): DE updating strategy. Defaults to "deferred".
        - strategy (str): DE mutation strategy. Defaults to "best1bin".
        - mutation (float | list[float]): DE mutation constant. Defaults to [0.5, 1.0].
        - recombination (float): DE recombination constant. Defaults to 0.7.
        - init (str): DE initialisation method. Defaults to "latinhypercube".
    """

    def __init__(
        self,
        domains_to_model: Union[int, list] = 8,
        misfit_stat_list: list = DEFAULT_MISFIT_STAT,
        lnd0aa_bounds: list = [-5.0, 50.0],
        ea_bounds: list = [50.0, 500.0],
        geometry: str = "spherical",
        omit_value_indices: list = [],
        max_iters: Union[float, int] = 100000,
        punish_degas_early: bool = True,
        repeat_iterations: int = 10,
        seed: Union[int, None] = None,
        tol: float = 0.0001,
        popsize: int = 15,
        updating: str = "deferred",
        strategy: str = "best1bin",
        mutation: Union[float, list] = [0.5, 1.0],
        recombination: float = 0.7,
        init: str = "latinhypercube",
    ):
        # Normalise [4] → 4 so a single-element list behaves identically to a plain int.
        if isinstance(domains_to_model, list) and len(domains_to_model) == 1:
            domains_to_model = domains_to_model[0]
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
            r = (
                range(self.domains_to_model, self.domains_to_model + 1)
                if isinstance(self.domains_to_model, int)
                else range(self.domains_to_model[0], self.domains_to_model[1] + 1)
            )
            for i in r:
                self.single_pipeline_configs[stat].append(
                    SingleProcessPipelineConfig(
                        lnd0aa_bounds=lnd0aa_bounds,
                        ea_bounds=ea_bounds,
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
        if not isinstance(self.domains_to_model, (int, list)):
            raise ValueError("domains_to_model must be an int or a [min, max] list")
        if isinstance(self.domains_to_model, list):
            if len(self.domains_to_model) != 2:
                raise ValueError("domains_to_model list must have exactly 2 elements: [min, max]")
            if not all(isinstance(d, int) for d in self.domains_to_model):
                raise ValueError("domains_to_model entries must be ints")
            if self.domains_to_model[0] >= self.domains_to_model[1]:
                raise ValueError("domains_to_model must be in increasing order")
            if self.domains_to_model[0] <= 0:
                raise ValueError("domains_to_model lower bound must be greater than 0")
        if not isinstance(self.misfit_stat_list, list):
            raise ValueError("misfit_stat_list must be a list")
        if not all(isinstance(s, str) for s in self.misfit_stat_list):
            raise ValueError("misfit_stat_list must be a list of strings")
        if not all(s.lower() in MISFIT_STAT_LIST for s in self.misfit_stat_list):
            raise ValueError(f"misfit_stat_list must only contain {MISFIT_STAT_LIST}")
        if len(set(self.misfit_stat_list)) != len(self.misfit_stat_list):
            raise ValueError("misfit_stat_list must not contain duplicates")
        if len(self.misfit_stat_list) == 0:
            raise ValueError("misfit_stat_list must contain at least one statistic")
