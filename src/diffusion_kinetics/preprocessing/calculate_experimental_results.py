import math
import numpy as np
import pandas as pd
import warnings


def calculate_diffusivity(exp_data: pd.DataFrame, geometry: str) -> pd.DataFrame:
    """Calculate ln(D/a²) and propagated uncertainties from raw step-heating data.

    Applies the Fechtig-Kalbitzer equations to convert measured gas fractions
    and step durations into diffusivity values, then propagates measurement
    uncertainties using the analytical expressions of Ginster (2018).

    Args:
        exp_data (pd.DataFrame): Raw experimental data with columns:
            ``TC`` (°C), ``thr`` (hours), ``M`` (gas amount), ``delM``
            (uncertainty on gas amount).
        geometry (str): Diffusion geometry — ``"spherical"`` or ``"plane sheet"``.

    Returns:
        pd.DataFrame: Columns ``Tplot``, ``Fi``, ``Daa``, ``Daa uncertainty``,
        ``ln(D/a^2)``, ``ln(D/a^2)-del``, ``ln(D/a^2)+del``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        tc = exp_data.loc[:, "TC"].array
        thr = exp_data.loc[:, "thr"].array
        M = exp_data.loc[:, "M"].array
        del_m = exp_data.loc[:, "delM"].array

        # Unit conversions
        t_k = 273.15 + tc          # temperature in Kelvin
        tsec = thr * 3600           # step duration in seconds
        t_plot = 1e4 / t_k          # 10000/T axis for Arrhenius plots
        n_steps = len(M)
        cum_tsec = np.cumsum(tsec)

        # Cumulative and fractional gas release
        cum_moles = np.cumsum(M)
        total_moles = np.amax(cum_moles)
        fi = cum_moles / total_moles

        # Incremental step duration and fractional release (length n-1)
        diff_ti = cum_tsec[1:] - cum_tsec[:-1]
        diff_fi = fi[1:] - fi[:-1]

        # Diffusivity arrays (one per Fechtig-Kalbitzer equation)
        dr2_a = np.zeros(n_steps)
        dr2_b = np.zeros(n_steps)

        if geometry == "spherical":
            dr2_c = np.zeros(n_steps)

            # Fechtig-Kalbitzer equation 5a (updated Reiners form, p. 90)
            dr2_a[0] = (
                1 / ((math.pi**2) * cum_tsec[0])
                * (
                    2 * math.pi
                    - (math.pi**2 / 3) * fi[0]
                    - 2 * math.pi * np.sqrt(1 - (math.pi / 3) * fi[0])
                )
            )
            dr2_a[1:] = (
                1 / (math.pi**2 * diff_ti)
                * (
                    -math.pi**2 / 3 * diff_fi
                    - 2 * math.pi
                    * (
                        np.sqrt(1 - (math.pi / 3) * fi[1:])
                        - np.sqrt(1 - (math.pi / 3) * fi[:-1])
                    )
                )
            )

            # Fechtig-Kalbitzer equation 5b
            dr2_b[0] = (
                -1 / (math.pi**2 * cum_tsec[0])
                * np.log((1 - fi[0]) * (math.pi**2 / 6))
            )
            dr2_b[1:] = (
                -1 / (math.pi**2 * diff_ti)
                * np.log((1 - fi[1:]) / (1 - fi[:-1]))
            )

            use_a = (fi <= 0.85) & (fi > 0.00000001)
            use_b = fi > 0.85
            dr2 = np.nan_to_num(use_a * dr2_a) + use_b * dr2_b

            # Uncertainty propagation — Ginster (2018)
            uncert_a = np.zeros(n_steps)
            uncert_b = np.zeros(n_steps)
            for i in range(1, n_steps):
                uncert_a[i] = (
                    1 / (3 * (cum_tsec[i] - cum_tsec[i - 1]) * np.sum(M))
                    * np.sqrt(
                        (
                            (1 - fi[i]) * (1 / np.sqrt(1 - (math.pi / 3) * fi[i]) - 1)
                            + (1 - fi[i - 1]) * (1 - 1 / np.sqrt(1 - (math.pi / 3) * fi[i - 1]))
                        ) ** 2
                        * np.sum(del_m[:i - 1] ** 2)
                        + (
                            (1 - fi[i]) * (1 / np.sqrt(1 - (math.pi / 3) * fi[i]) - 1)
                            - fi[i - 1] * (1 - 1 / np.sqrt(1 - (math.pi / 3) * fi[i - 1]))
                        ) ** 2
                        * del_m[i] ** 2
                        + (
                            fi[i] * (1 - 1 / np.sqrt(1 - (math.pi / 3) * fi[i]))
                            - fi[i - 1] * (1 - 1 / np.sqrt(1 - (math.pi / 3) * fi[i - 1]))
                        ) ** 2
                        * np.sum(del_m[i + 1:] ** 2)
                    )
                )
                uncert_b[i] = (
                    1 / (math.pi**2 * (cum_tsec[i] - cum_tsec[i - 1]) * np.sum(M))
                    * np.sqrt(
                        (1 + fi[i - 1] / (1 - fi[i - 1])) ** 2 * del_m[i] ** 2
                        + (fi[i - 1] / (1 - fi[i - 1]) - fi[i] / (1 - fi[i])) ** 2
                        * np.sum(del_m[i + 1:] ** 2)
                    )
                )

            dr2_uncert = np.nan_to_num(uncert_a * use_a) + uncert_b * use_b

            # Special case: first step (Ginster, 2018)
            if fi[0] <= 0.85:
                dr2_uncert[0] = (
                    1 / (3 * cum_tsec[0] * np.sum(M))
                    * (1 / np.sqrt(1 - (math.pi / 3) * fi[0]) - 1)
                    * np.sqrt(
                        ((1 - fi[0]) * del_m[0]) ** 2
                        + fi[0] ** 2 * np.sum(del_m[1:] ** 2)
                    )
                )
            else:
                dr2_uncert[0] = (
                    1 / (math.pi**2 * cum_tsec[0] * np.sum(M))
                    * np.sqrt(
                        del_m[0] ** 2
                        + (fi[0] / (1 - fi[0])) ** 2 * np.sum(del_m[1:] ** 2)
                    )
                )

        elif geometry == "plane sheet":
            # Fechtig-Kalbitzer equation 5a
            dr2_a[0] = (fi[0] ** 2 * math.pi) / (4 * tsec[0])
            dr2_a[1:] = ((fi[1:] ** 2 - fi[:-1] ** 2) * math.pi) / (4 * tsec[1:])
            dr2_b[1:] = (4 / ((math.pi**2) * tsec[1:])) * np.log(
                (1 - fi[:-1]) / (1 - fi[1:])
            )

            use_a = (fi > 0) & (fi < 0.6)
            use_b = (fi >= 0.6) & (fi <= 1)
            dr2 = use_a * dr2_a + use_b * dr2_b

            uncert_b = np.zeros(n_steps)
            uncert_c = np.zeros(n_steps)

            # Uncertainty propagation — Ginster (2018)
            for i in range(1, n_steps):
                uncert_b[i] = (
                    math.pi / (2 * (cum_tsec[i] - cum_tsec[i - 1]) * np.sum(M))
                    * np.sqrt(
                        ((fi[i] * (1 - fi[i])) - fi[i - 1] * (1 - fi[i - 1])) ** 2
                        * np.sum(del_m[:i] ** 2)
                        + (fi[i] * (1 - fi[i]) + fi[i - 1] ** 2) ** 2 * del_m[i] ** 2
                        + (fi[i - 1] ** 2 - fi[i] ** 2) ** 2
                        * np.sum(del_m[i + 1:] ** 2)
                    )
                )
                uncert_c[i] = (
                    4 / (math.pi**2 * (cum_tsec[i] - cum_tsec[i - 1]) * np.sum(M))
                    * np.sqrt(
                        (1 + fi[i - 1] / (1 - fi[i - 1])) ** 2 * del_m[i] ** 2
                        + (fi[i - 1] / (1 - fi[i - 1]) - fi[i] / (1 - fi[i])) ** 2
                        * np.sum(del_m[i + 1:] ** 2)
                    )
                )

            dr2_uncert = np.nan_to_num(use_a * uncert_b) + np.nan_to_num(use_b * uncert_c)

            # Special case: first step (Ginster, 2018)
            if fi[0] < 0.6:
                dr2_uncert[0] = (
                    math.pi / (2 * diff_ti[0])
                    * (diff_fi[0] / np.sum(M))
                    * np.sqrt(
                        ((1 - diff_fi[0]) * del_m[0]) ** 2
                        + diff_fi[0] ** 2 * np.sum(del_m[1:] ** 2)
                    )
                )
            else:
                dr2_uncert[0] = (
                    4 / (math.pi**2 * diff_ti[0] * np.sum(M))
                    * np.sqrt(
                        del_m[0] ** 2
                        + (diff_fi[0] / (1 - diff_fi[0])) ** 2 * np.sum(del_m[2:] ** 2)
                    )
                )

        return pd.DataFrame(
            {
                "Tplot": t_plot,
                "Fi": fi.ravel(),
                "Daa": dr2,
                "Daa uncertainty": dr2_uncert.ravel(),
                "ln(D/a^2)": np.log(dr2),
                "ln(D/a^2)-del": dr2_uncert.ravel() / dr2,
                "ln(D/a^2)+del": dr2_uncert.ravel() / dr2,
            }
        )
