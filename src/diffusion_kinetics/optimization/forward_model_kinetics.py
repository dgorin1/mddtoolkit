import math
from typing import Union

import numpy as np
import torch


def forward_model_kinetics(
    kinetics: Union[np.ndarray, torch.Tensor],
    tsec: torch.Tensor,
    tc: torch.Tensor,
    geometry: str = "spherical",
    added_steps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cumulative fractional gas release at each heating step using the
    Multi-Domain Diffusion (MDD) forward model.

    Args:
        kinetics: 1-D or 2-D parameter vector(s) with layout
            ``[Ea, lnD0/a²_1, ..., lnD0/a²_N, frac_1, ..., frac_{N-1}]``.
            May be a numpy array or torch Tensor.
        tsec: Duration of each heating step in seconds.
        tc: Temperature of each heating step in degrees Celsius.
        geometry: Diffusion geometry — ``"spherical"`` or ``"plane sheet"``.
        added_steps: Number of pre-experiment heating steps prepended to
            ``tsec`` and ``tc``. Results are re-normalised after stripping these
            steps so the output matches what would have been measured in the lab.

    Returns:
        ``(sumf_mdd, punishment_flag)`` where

        - ``sumf_mdd``: Cumulative fractional gas release at each step,
          shape ``(n_steps, n_vectors)``.
        - ``punishment_flag``: Boolean tensor of shape ``(n_vectors,)``;
          ``True`` where the model exhausts all gas before the final step.
    """
    R = 0.008314  # gas constant (kJ mol⁻¹ K⁻¹)

    # torch.pi is not available in all torch versions; define it explicitly.
    torch.pi = torch.tensor(math.acos(-1.0))

    if not isinstance(kinetics, torch.Tensor):
        kinetics = torch.tensor(kinetics)

    # Ensure kinetics is 2-D: (n_params, n_vectors)
    if kinetics.ndim <= 1:
        kinetics = torch.unsqueeze(kinetics, 1)
    n_vectors = kinetics.shape[1]

    # Infer domain count from parameter length (total_moles is not passed here)
    n_dom = len(kinetics) // 2

    # Unpack parameters
    ea = kinetics[0]
    ln_d0aa = kinetics[1 : n_dom + 1].unsqueeze(0).expand(len(tc), n_dom, -1)
    fracs_temp = kinetics[n_dom + 1 :]
    fracs = (
        torch.cat((fracs_temp, 1 - torch.sum(fracs_temp, axis=0, keepdim=True)))
        .unsqueeze(0)
        .expand(len(tc), -1, -1)
    )
    ea = ea.unsqueeze(0).expand(len(tc), n_dom, -1)

    # Cumulative and incremental time, broadcast to (n_steps, n_dom, n_vectors)
    cum_tsec = torch.cumsum(tsec, dim=0).unsqueeze(-1).repeat(1, n_dom)
    cum_tsec = cum_tsec.unsqueeze(2).repeat(1, 1, n_vectors)
    tsec_exp = tsec.unsqueeze(-1).repeat(1, n_dom).unsqueeze(2).repeat(1, 1, n_vectors)
    t_k = (tc + 273.15).unsqueeze(-1).repeat(1, n_dom).unsqueeze(2).repeat(1, 1, n_vectors)

    # D/a² for each domain at each step
    daa = torch.exp(ln_d0aa) * torch.exp(-ea / (R * t_k))

    # Incremental Dt/a²
    dtaa_for_sum = torch.zeros(daa.shape)
    dtaa_for_sum[0, :, :] = daa[0, :, :] * tsec_exp[0, :, :]
    dtaa_for_sum[1:, :, :] = daa[1:, :, :] * (cum_tsec[1:, :, :] - cum_tsec[:-1, :, :])

    if geometry == "spherical":
        dtaa = torch.cumsum(dtaa_for_sum, axis=0)
        bt = dtaa * torch.pi**2
        f = torch.zeros(daa.shape)
        f[bt <= 1.401] = (
            6 / (torch.pi ** (3 / 2)) * (torch.pi**2 * dtaa[bt <= 1.401]) ** (1 / 2)
            - (3 / (torch.pi**2)) * (torch.pi**2 * dtaa[bt <= 1.401])
        )
        f[bt > 1.401] = 1 - (6 / (torch.pi**2)) * torch.exp(
            -(torch.pi**2) * dtaa[bt > 1.401]
        )

    elif geometry == "plane sheet":
        dtaa = torch.cumsum(dtaa_for_sum, axis=0)
        f = (2 / torch.sqrt(torch.pi)) * torch.sqrt(dtaa)
        f[f > 0.6] = 1 - (8 / (torch.pi**2)) * torch.exp(
            -1 * torch.pi**2 * dtaa[f > 0.6] / 4
        )

    # Weight each domain's release by its gas fraction and sum across domains
    f_mdd = f * fracs
    sumf_mdd = torch.sum(f_mdd, axis=1)

    if added_steps > 0:
        # Strip the pre-experiment steps and re-normalise so the output looks
        # like a normal lab measurement.
        new_f = torch.zeros(sumf_mdd.shape)
        new_f[0] = sumf_mdd[0]
        new_f[1:] = sumf_mdd[1:] - sumf_mdd[:-1]
        new_f = new_f[added_steps:]

        punishment_flag = torch.round(sumf_mdd[-1, :], decimals=3) < 1

        normalization_factor = torch.max(torch.cumsum(new_f, 0), axis=0).values
        diff_fi = new_f / normalization_factor
        sumf_mdd = torch.cumsum(diff_fi, axis=0)
    else:
        punishment_flag = torch.round(sumf_mdd[-1, :], decimals=3) < 1

    # Zero out any columns that are entirely NaN (un-physical parameter vectors)
    nan_mask = torch.isnan(sumf_mdd).all(dim=0)
    sumf_mdd[:, nan_mask] = 0.0

    return sumf_mdd, punishment_flag


def calc_ln_d0aa(
    sumf_mdd: torch.Tensor,
    diff_ti: torch.Tensor,
    geometry: str,
) -> torch.Tensor:
    """Convert MDD cumulative fractional release back to ln(D/a²) at each step
    using the Fechtig-Kalbitzer equations. Used for Arrhenius plot visualisation.

    Args:
        sumf_mdd: Cumulative fractional release, shape
            ``(n_steps,)`` or ``(n_steps, n_vectors)``.
        diff_ti: Duration of each heating step in seconds.
        geometry: ``"spherical"`` or ``"plane sheet"``.

    Returns:
        ln(D/a²) at each step, same shape as ``sumf_mdd``.
    """
    if len(sumf_mdd.size()) > 1:
        diff_ti = diff_ti.unsqueeze(1).repeat(1, sumf_mdd.size()[1])

    if geometry == "spherical":
        diff_fi = sumf_mdd[1:] - sumf_mdd[:-1]
        fi = sumf_mdd

        if len(diff_ti.shape) == 1:
            cum_tsec = torch.cumsum(diff_ti, 0)
        else:
            cum_tsec = torch.cumsum(diff_ti, 1)
        diff_ti_inner = diff_ti[1:]

        dr2_a = torch.zeros(sumf_mdd.shape)
        dr2_b = torch.zeros(sumf_mdd.shape)

        # Fechtig-Kalbitzer equation 5a (low-release regime, Fi ≤ 0.85)
        dr2_a[0] = (
            1
            / ((torch.pi**2) * cum_tsec[0])
            * (
                2 * torch.pi
                - (torch.pi**2 / 3) * fi[0]
                - 2 * torch.pi * torch.sqrt(1 - (torch.pi / 3) * fi[0])
            )
        )
        dr2_a[1:] = (1 / (torch.pi**2 * diff_ti_inner)) * (
            -(torch.pi**2 / 3) * diff_fi
            - 2
            * torch.pi
            * (
                torch.sqrt(1 - (torch.pi / 3) * fi[1:])
                - torch.sqrt(1 - (torch.pi / 3) * fi[:-1])
            )
        )

        # Fechtig-Kalbitzer equation 5b (high-release regime, Fi > 0.85)
        dr2_b[0] = (
            -1 / (torch.pi**2 * cum_tsec[0]) * torch.log((1 - fi[0]) * (torch.pi**2 / 6))
        )
        dr2_b[1:] = (
            -1 / (torch.pi**2 * diff_ti_inner) * torch.log((1 - fi[1:]) / (1 - fi[:-1]))
        )

        use_a = (fi <= 0.85) & (fi > 0.00000001)
        use_b = fi > 0.85
        daa_mdd = torch.nan_to_num(use_a * dr2_a) + use_b * dr2_b

    elif geometry == "plane sheet":
        dr2_a = torch.zeros(sumf_mdd.shape)
        dr2_b = torch.zeros(sumf_mdd.shape)

        dr2_a[0] = (sumf_mdd[0] ** 2 * torch.pi) / (4 * diff_ti[0])
        dr2_a[1:] = ((sumf_mdd[1:] ** 2 - sumf_mdd[:-1] ** 2) * torch.pi) / (
            4 * diff_ti[1:]
        )
        dr2_b[1:] = (4 / ((torch.pi**2) * diff_ti[1:])) * torch.log(
            (1 - sumf_mdd[:-1]) / (1 - sumf_mdd[1:])
        )

        use_a = (sumf_mdd > 0) & (sumf_mdd < 0.6)
        use_b = (sumf_mdd >= 0.6) & (sumf_mdd <= 1)
        daa_mdd = use_a * dr2_a + use_b * dr2_b

    return torch.log(daa_mdd)
