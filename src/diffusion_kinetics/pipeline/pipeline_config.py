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
        assert isinstance(self.lnd0aa_bounds, list), "lnd0aa_bounds must be a list"
        assert isinstance(self.ea_bounds, list), "ea_bounds must be a list"
        assert isinstance(self.geometry, str), "geometry must be a string"
        assert isinstance(self.omit_value_indices, list), "omit_value_indices must be a list"
        assert isinstance(self.max_iters, (float, int)), "max_iters must be a float or int"
        assert isinstance(self.punish_degas_early, bool), "punish_degas_early must be a bool"
        assert len(self.lnd0aa_bounds) == 2, "lnd0aa_bounds must be a list of length 2"
        assert len(self.ea_bounds) == 2, "ea_bounds must be a list of length 2"
        assert self.lnd0aa_bounds[0] < self.lnd0aa_bounds[1], "lnd0aa_bounds must be in increasing order"
        assert self.ea_bounds[0] <= self.ea_bounds[1], "ea_bounds must be in increasing order, or equal for fixed Ea"
        assert self.geometry in ["spherical", "plane sheet"], "geometry must be 'spherical' or 'plane sheet'"
        assert self.max_iters > 0, "max_iters must be greater than 0"
        assert self.num_domains >= 0, "num_domains must be greater than 0"
        assert self.misfit_stat in MISFIT_STAT_LIST, f"misfit_stat must be one of {MISFIT_STAT_LIST}, got '{self.misfit_stat}'"
        assert self.repeat_iterations > 0, "repeat_iterations must be greater than 0"
        assert self.repeat_iterations == int(self.repeat_iterations), "repeat_iterations must be an integer"
        assert self.seed is None or self.seed == int(self.seed), "seed must be an integer"
        assert self.tol > 0, "tol must be greater than 0"
        assert self.popsize > 0, "popsize must be greater than 0"
        assert self.updating in ["immediate", "deferred"], "updating must be 'immediate' or 'deferred'"
        assert self.strategy in [
            "best1bin", "best1exp", "rand1exp", "randtobest1exp", "currenttobest1exp",
            "best2exp", "rand2exp", "randtobest1bin", "currenttobest1bin",
            "best2bin", "rand2bin", "rand1bin",
        ], "strategy must be a valid differential-evolution strategy"
        assert 0 < self.recombination < 1, "recombination must be between 0 and 1"
        if isinstance(self.mutation, float):
            assert 0 < self.mutation < 2, "mutation must be between 0 and 2"
        elif isinstance(self.mutation, list):
            assert len(self.mutation) == 2, "mutation must be a list of length 2"
            assert 0 < self.mutation[0] < 2, "mutation[0] must be between 0 and 2"
            assert 0 < self.mutation[1] < 2, "mutation[1] must be between 0 and 2"
        else:
            raise ValueError("mutation must be a float or a list")
        assert self.init in ["latinhypercube", "sobol", "halton", "random"], "init must be a valid init method"

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
        - domains_to_model (int | list[int]): Number of domains, or [min, max] range. Defaults to 8.
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
        assert isinstance(self.domains_to_model, (int, list)), "domains_to_model must be an int or a list of ints"
        if isinstance(self.domains_to_model, list):
            assert len(self.domains_to_model) == 2, "domains_to_model must be an int or a list of length 2"
            assert all(isinstance(d, int) for d in self.domains_to_model), "domains_to_model entries must be ints"
            assert self.domains_to_model[0] < self.domains_to_model[1], "domains_to_model must be in increasing order"
            assert self.domains_to_model[0] > 0, "domains_to_model lower bound must be greater than 0"
        assert isinstance(self.misfit_stat_list, list), "misfit_stat_list must be a list"
        assert all(isinstance(s, str) for s in self.misfit_stat_list), "misfit_stat_list must be a list of strings"
        assert all(s.lower() in MISFIT_STAT_LIST for s in self.misfit_stat_list), (
            f"misfit_stat_list must only contain {MISFIT_STAT_LIST}"
        )
        assert len(set(self.misfit_stat_list)) == len(self.misfit_stat_list), "misfit_stat_list must not contain duplicates"
        assert len(self.misfit_stat_list) > 0, "misfit_stat_list must contain at least one statistic"
