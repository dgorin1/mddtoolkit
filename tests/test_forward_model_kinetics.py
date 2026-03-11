"""
Regression tests for forward_model_kinetics using sample_synthetic_data.csv.

These tests verify that the optimizer can recover kinetics for 1-5 domain
models from the example dataset. Run them after code changes to confirm
that forward_model_kinetics and the optimization pipeline still work correctly.
"""
import numpy as np
import pytest
import torch
from pathlib import Path
from scipy.optimize import differential_evolution, NonlinearConstraint

from diffusion_kinetics.optimization.dataset import Dataset
from diffusion_kinetics.optimization.forward_model_kinetics import forward_model_kinetics
from diffusion_kinetics.optimization.diffusion_objective import DiffusionObjective
from diffusion_kinetics.optimization.con_he_param import con_he_param
from diffusion_kinetics.pipeline.pipeline_config import SingleProcessPipelineConfig
from diffusion_kinetics.preprocessing.generate_inputs import generate_inputs

SAMPLE_DATA_PATH = Path(__file__).parent.parent / "example" / "sample_synthetic_data.csv"
GEOMETRY = "plane sheet"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    tmp_output = tmp_path_factory.mktemp("data") / "processed.csv"
    df = generate_inputs(str(SAMPLE_DATA_PATH), str(tmp_output), geometry=GEOMETRY)
    return Dataset(df)


def _r2(predicted: torch.Tensor, observed: torch.Tensor) -> float:
    ss_res = torch.sum((observed - predicted) ** 2).item()
    ss_tot = torch.sum((observed - observed.mean()) ** 2).item()
    return 1.0 - ss_res / ss_tot


def _run_optimizer(dataset: Dataset, config: SingleProcessPipelineConfig, seed: int):
    """Minimal optimizer runner that avoids importing DiffusionOptimizer."""
    n_dom = config.num_domains
    objective = DiffusionObjective(
        dataset,
        config.omit_value_indices,
        config.misfit_stat,
        config.geometry,
        config.punish_degas_early,
    )

    # percent_frac does not include moles in the parameter vector
    if n_dom == 1:
        bounds = [config.ea_bounds, config.lnd0aa_bounds]
    else:
        bounds = (
            [config.ea_bounds]
            + n_dom * [config.lnd0aa_bounds]
            + (n_dom - 1) * [(0, 1)]
        )

    nlcs = NonlinearConstraint(con_he_param, lb=[0], ub=[np.inf]) if n_dom > 1 else []

    return differential_evolution(
        objective,
        bounds,
        disp=False,
        tol=config.tol,
        maxiter=config.max_iters,
        constraints=nlcs,
        vectorized=True,
        updating=config.updating,
        seed=seed,
        popsize=config.popsize,
        strategy=config.strategy,
        mutation=config.mutation,
        recombination=config.recombination,
        init=config.init,
    )


# ── Basic sanity checks (no optimizer) ───────────────────────────────────────

def test_forward_model_output_shape(dataset):
    """forward_model_kinetics returns tensors of the correct shape."""
    n_dom = 2
    # [Ea, lnD0aa_1, lnD0aa_2, frac_1]
    params = torch.tensor([150.0, 10.0, 5.0, 0.5])
    tsec = dataset._thr * 3600
    tc = dataset._tc

    fi_pred, _ = forward_model_kinetics(params, tsec, tc, geometry=GEOMETRY, added_steps=0)

    assert fi_pred.shape[0] == len(tc), "Output length must match number of heating steps"
    assert fi_pred[-1].item() <= 1.0 + 1e-4, "Cumulative release fraction must not exceed 1"
    assert fi_pred[0].item() >= 0.0, "Cumulative release fraction must be non-negative"
    assert torch.all(fi_pred[1:] >= fi_pred[:-1] - 1e-5), (
        "Cumulative release fractions must be non-decreasing"
    )


def test_forward_model_monotone_with_time(dataset):
    """Longer heating time → at least as much gas released."""
    params = torch.tensor([150.0, 10.0])  # 1-domain: [Ea, lnD0aa]
    tc = dataset._tc

    tsec_short = dataset._thr * 3600
    tsec_long = tsec_short * 10

    fi_short, _ = forward_model_kinetics(params, tsec_short, tc, geometry=GEOMETRY)
    fi_long, _ = forward_model_kinetics(params, tsec_long, tc, geometry=GEOMETRY)

    assert fi_long[-1].item() >= fi_short[-1].item() - 1e-4


# ── Recovery tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_dom", [1, 2, 3, 4, 5])
def test_fwdmodelkinetics_recovers_kinetics(dataset, n_dom):
    """
    Optimizer + forward_model_kinetics must recover kinetics for an n-domain
    model such that predicted Fi matches experimental Fi with R² > 0.90.
    """
    config = SingleProcessPipelineConfig(
        num_domains=n_dom,
        misfit_stat="percent_frac",
        geometry=GEOMETRY,
        lnd0aa_bounds=[-5.0, 50.0],
        ea_bounds=[50.0, 500.0],
        punish_degas_early=False,
        seed=42,
        tol=1e-3,
        popsize=15,
        max_iters=10000,
        repeat_iterations=1,
        updating="deferred",
    )

    result = _run_optimizer(dataset, config, seed=42)

    assert np.isfinite(result.fun), (
        f"{n_dom}-domain optimizer returned non-finite misfit: {result.fun}"
    )

    # Evaluate the forward model at the optimised parameters
    best_params = torch.tensor(result.x)
    tsec = dataset._thr * 3600
    tc = dataset._tc

    fi_pred, _ = forward_model_kinetics(
        best_params, tsec, tc, geometry=GEOMETRY, added_steps=0
    )
    fi_pred = fi_pred.squeeze()
    fi_exp = dataset._fi_exp

    r2 = _r2(fi_pred, fi_exp)

    assert r2 > 0.90, (
        f"Poor fit for {n_dom}-domain model: R²={r2:.4f}. "
        "This likely indicates a bug in forward_model_kinetics or the optimizer."
    )
