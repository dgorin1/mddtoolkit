from typing import Union

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from diffusion_kinetics.optimization.forward_model_kinetics import (
    forward_model_kinetics,
    calc_ln_d0aa,
)


def plot_results(
    params: Union[np.ndarray, torch.Tensor, list],
    dataset: pd.DataFrame,
    objective,
    plot_path: str,
) -> None:
    """Generate and save a three-panel diagnostic figure for an MDD fit.

    The figure contains:
        - **Left panel**: Arrhenius plot (ln(D/a²) vs 10000/T).
        - **Bottom-right panel**: Step-wise fractional gas release.
        - **Top-right panel**: Residuals vs cumulative gas release.

    In every panel, data points excluded from the fit are drawn at reduced
    opacity, and model predictions are shown as filled black symbols.

    Args:
        params: Optimised parameter vector ``[Ea, lnD0/a²_1, ..., frac_1, ...]``.
            If the vector is odd-length (chisq stat), the leading ``total_moles``
            entry is stripped before use.
        dataset (Dataset): Experimental dataset.
        objective (DiffusionObjective): Objective used for the fit (provides
            ``tsec``, ``geometry``, and ``omit_value_indices``).
        plot_path (str): File path at which to save the figure.
    """
    R = 0.008314  # gas constant (kJ mol⁻¹ K⁻¹)
    params = torch.tensor(params)

    # Strip the leading total_moles parameter if present (odd-length vector)
    if len(params) % 2 != 0:
        params = params[1:]

    n_dom = len(params) // 2
    tsec = objective.tsec
    tc = torch.tensor(dataset.TC)

    # --- Forward model ---
    fi_mdd, _ = forward_model_kinetics(
        params, tsec, tc, geometry=objective.geometry, added_steps=0
    )
    lnd0aa_mdd = calc_ln_d0aa(fi_mdd, objective.tsec, objective.geometry)

    # Replace infinities with NaN so they don't distort the axes
    fi_mdd = fi_mdd.masked_fill(torch.isinf(fi_mdd), float("nan"))
    fi_mdd = np.array(fi_mdd.ravel())

    lnd0aa_mdd = lnd0aa_mdd.masked_fill(torch.isinf(lnd0aa_mdd), float("nan"))
    lnd0aa_mdd = np.array(lnd0aa_mdd.ravel())

    # Replace infinities in the experimental columns with NaN
    for col in ["ln(D/a^2)", "ln(D/a^2)-del", "ln(D/a^2)+del", "Fi"]:
        dataset[col] = dataset[col].replace([np.inf, -np.inf], np.nan)

    # 10000/T axis used throughout
    t_plot = 10000 / (dataset["TC"] + 273.15)

    # --- Included / omitted indices ---
    # Using np.where avoids the scalar-squeeze ambiguity from .nonzero().squeeze()
    omit_mask = objective.omit_value_indices.numpy().astype(bool)
    included = np.where(~omit_mask)[0]
    omitted = np.where(omit_mask)[0]

    # --- Domain line weights (linewidth ∝ gas fraction for visual emphasis) ---
    if n_dom > 1:
        fracs = params[n_dom + 1 :]
        fracs = torch.concat(
            (fracs, 1 - torch.sum(fracs, axis=0, keepdim=True)), axis=-1
        )
        scale = n_dom + 5 if n_dom <= 6 else n_dom + 10
        frac_weights = (fracs * scale).clamp(min=0.4)
    else:
        frac_weights = [2]

    # =========================================================================
    # Figure layout: 2 rows × 4 columns; Arrhenius plot spans the left half,
    # fractional release and residuals share the right half.
    # =========================================================================
    fig = plt.figure(1)
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)  # Arrhenius
    ax2 = plt.subplot2grid((2, 4), (1, 2), colspan=2, rowspan=1)  # fractional release
    ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=1)  # residuals

    for ax in (ax1, ax2, ax3):
        for spine in ax.spines.values():
            spine.set_linewidth(1.15)

    # =========================================================================
    # Panel 1 – Arrhenius plot
    # =========================================================================
    # Draw one dashed Arrhenius line per domain; linewidth encodes gas fraction.
    t_arr = (10000 / (tc[:-1] + 273.15)).numpy()
    t_line = np.array([t_arr.min(), t_arr.max()])
    slope = params[0].item() / (R * 10000)  # Ea / (R · 10000) for t_plot units
    for i in range(n_dom):
        d_line = params[i + 1].item() - slope * t_line
        ax1.plot(
            t_line, d_line,
            "--", linewidth=frac_weights[i], zorder=0, color=(0.6, 0, 0), alpha=0.5,
        )

    # Mask model ln(D/a²) at steps where the experimental value is also NaN
    if dataset["ln(D/a^2)"].isna().any():
        lnd0aa_mdd[dataset["ln(D/a^2)"].isna()] = np.nan

    # Error arrays for errorbar (shape 2 × n)
    errors_included = np.array(
        pd.concat(
            [dataset["ln(D/a^2)-del"][included], dataset["ln(D/a^2)+del"][included]],
            axis=1,
        ).T
    )
    if len(omitted) == 1:
        errors_omitted = [
            [dataset["ln(D/a^2)-del"].iloc[omitted[0]]],
            [dataset["ln(D/a^2)+del"].iloc[omitted[0]]],
        ]
    else:
        errors_omitted = np.array(
            pd.concat(
                [dataset[["ln(D/a^2)-del"]].iloc[omitted], dataset[["ln(D/a^2)+del"]].iloc[omitted]],
                axis=1,
            ).T
        )

    # Experimental data
    ax1.errorbar(
        t_plot[included], dataset["ln(D/a^2)"].loc[included], yerr=errors_included,
        fmt="o", markersize=12, color=(0.69, 0.69, 0.69), linewidth=1, mec="black", zorder=1,
    )
    ax1.errorbar(
        t_plot[omitted], dataset["ln(D/a^2)"].loc[omitted], yerr=errors_omitted,
        fmt="o", markersize=12, color=(0.69, 0.69, 0.69), linewidth=1, mec="black", zorder=1, alpha=0.4,
    )

    # Model predictions
    ax1.plot(
        t_plot[included], pd.Series(lnd0aa_mdd[included].tolist()),
        "o", markersize=5, color="black", linewidth=1, mec="black", zorder=2,
    )
    ax1.plot(
        t_plot[omitted], pd.Series(lnd0aa_mdd[omitted].tolist()),
        "o", markersize=5, color="black", linewidth=1, mec="black", zorder=2, alpha=0.4,
    )

    ax1.set_ylabel("ln(D/a$^2$)", fontsize=15.5)
    ax1.set_xlabel("10000/T (K)", fontsize=15.5)
    ax1.tick_params(labelsize=12)
    ax1.set_box_aspect(1)

    # =========================================================================
    # Panel 2 – Step-wise fractional gas release
    # =========================================================================
    # Convert cumulative → differential release for both model and experiment
    fi_mdd_diff = np.insert(fi_mdd[1:] - fi_mdd[:-1], 0, fi_mdd[0])
    fi_exp = np.array(dataset["Fi"])
    fi_exp_diff = np.insert(fi_exp[1:] - fi_exp[:-1], 0, fi_exp[0])

    step_nums = np.arange(1, len(fi_mdd_diff) + 1)

    # Experimental points
    ax2.errorbar(
        step_nums[included], fi_exp_diff[included],
        fmt="o", markersize=12, mfc=(0.69, 0.69, 0.69), mec="black", zorder=5, linewidth=1, color="k",
    )
    ax2.errorbar(
        step_nums[omitted], fi_exp_diff[omitted],
        fmt="o", markersize=12, mfc=(0.69, 0.69, 0.69), mec="black", zorder=5, linewidth=1, color="k", alpha=0.3,
    )

    # Model points
    ax2.plot(
        step_nums[included], fi_mdd_diff[included],
        "o", markersize=5.25, color="black", linewidth=1, mec="black", zorder=10,
    )
    ax2.plot(
        step_nums[omitted], fi_mdd_diff[omitted],
        "o", markersize=5.25, color="black", linewidth=1, mec="black", zorder=10, alpha=0.55,
    )

    # Connecting lines (reduced opacity if either endpoint was omitted)
    for i in range(len(fi_mdd_diff) - 1):
        alpha_val = 0.45 if (i in omitted or i + 1 in omitted) else 1.0
        ax2.plot(step_nums[i : i + 2], fi_mdd_diff[i : i + 2], "k-", alpha=alpha_val, zorder=10)
        ax2.plot(step_nums[i : i + 2], fi_exp_diff[i : i + 2], "--", color=(0.69, 0.69, 0.69), alpha=alpha_val, zorder=1)

    ax2.set_xlabel("Step Number", fontsize=12)
    ax2.set_ylabel("Fractional Release (%)", fontsize=12)
    ax2.set_box_aspect(1)

    # =========================================================================
    # Panel 3 – Residuals vs cumulative gas release
    # =========================================================================
    # Residuals are computed relative to the most-retentive domain's Arrhenius line.
    # In 10000/T coordinates: slope = Ea / (R · 10000), intercept = lnD0/a²_1.
    resid_exp = dataset["ln(D/a^2)"] - (params[1].item() - slope * t_plot)
    resid_model = lnd0aa_mdd - (params[1].item() - slope * np.array(t_plot))

    cum_fi_mdd = np.cumsum(fi_mdd_diff) * 100
    cum_fi_exp = np.cumsum(fi_exp_diff) * 100

    # Experimental residuals
    ax3.plot(
        cum_fi_exp[included], resid_exp[included],
        "o", markersize=12, color=(0.69, 0.69, 0.69), mec="black", alpha=0.8,
    )
    ax3.plot(
        cum_fi_exp[omitted], resid_exp[omitted],
        "o", markersize=12, color=(0.69, 0.69, 0.69), mec="black", alpha=0.3,
    )

    # Model residuals
    ax3.plot(
        cum_fi_mdd[included], resid_model[included],
        "o", markersize=5, color="black", linewidth=1, mec="black",
    )
    ax3.plot(
        cum_fi_mdd[omitted], resid_model[omitted],
        "o", markersize=5, color="black", linewidth=1, mec="black", alpha=0.3,
    )

    # Connecting lines
    for i in range(len(cum_fi_mdd) - 1):
        alpha_val = 0.45 if (i in omitted or i + 1 in omitted) else 1.0
        ax3.plot(cum_fi_mdd[i : i + 2], resid_model[i : i + 2], "k-", alpha=alpha_val, zorder=250)
        ax3.plot(cum_fi_exp[i : i + 2], resid_exp[i : i + 2], "--", color=(0.69, 0.69, 0.69), alpha=alpha_val, zorder=1)

    ax3.set_xlabel("Cumulative Gas Release (%)", fontsize=11)
    ax3.set_ylabel("Residual ln(1/s)", fontsize=11)
    ax3.set_box_aspect(1)

    fig.tight_layout()
    fig.set_size_inches(w=15, h=7)
    plt.savefig(plot_path)
    plt.close(fig)
