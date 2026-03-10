from diffusion_kinetics.pipeline.pipeline_config import SingleProcessPipelineConfig, MultiProcessPipelineConfig


def test_single_process_pipeline_config():
    test_config = {
        "lnd0aa_bounds": [-5.0, 50.0],
        "ea_bounds": [50.0, 500.0],
        "geometry": "spherical",
        "num_domains": 1,
        "max_iters": 1000,
        "misfit_stat": "chisq",
        "omit_value_indices": [0],
        "punish_degas_early": False,
        "seed": 0,
        "tol": 1e-3,
        "updating": "deferred",
        "popsize": 15,
    }

    config = SingleProcessPipelineConfig(**test_config)
    assert config.ea_bounds == test_config["ea_bounds"]
    assert config.geometry == test_config["geometry"]
    assert config.lnd0aa_bounds == test_config["lnd0aa_bounds"]
    assert config.num_domains == test_config["num_domains"]
    assert config.max_iters == test_config["max_iters"]
    assert config.misfit_stat == test_config["misfit_stat"]
    assert config.omit_value_indices == test_config["omit_value_indices"]
    assert config.punish_degas_early == test_config["punish_degas_early"]
    assert config.seed == test_config["seed"]
    assert config.tol == test_config["tol"]
    assert config.updating == test_config["updating"]
    assert config.popsize == test_config["popsize"]


def test_sp_config_fails_on_bad_misfit_stat():
    test_config = {
        "lnd0aa_bounds": [-5.0, 50.0],
        "ea_bounds": [50.0, 500.0],
        "geometry": "spherical",
        "num_domains": 1,
        "max_iters": 1000,
        "misfit_stat": "bad_stat",
        "omit_value_indices": [0],
        "punish_degas_early": False,
        "seed": 0,
        "tol": 1e-3,
        "updating": "deferred",
        "popsize": 15,
    }
    try:
        SingleProcessPipelineConfig(**test_config)
        assert False, "Should have raised an error for invalid misfit_stat"
    except Exception:
        assert True
