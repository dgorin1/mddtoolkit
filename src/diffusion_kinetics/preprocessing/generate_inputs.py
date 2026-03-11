import pandas as pd
from diffusion_kinetics.preprocessing.calculate_experimental_results import calculate_diffusivity


def generate_inputs(
    input_csv_path: str,
    output_csv_path: str,
    geometry: str = "spherical",
) -> pd.DataFrame:
    """Pre-process raw step-heating data into the format expected by the optimizer.

    Reads a raw CSV, computes ln(D/a²) values and their uncertainties via
    :func:`calculate_diffusivity`, and writes the combined result to disk.

    Args:
        input_csv_path (str): Path to the raw input CSV file. Expected columns
            (no header row): temperature (°C), step duration (hours), gas amount,
            and gas-amount uncertainty.
        output_csv_path (str): Path at which to save the processed CSV.
        geometry (str): Diffusion geometry — ``"spherical"`` or ``"plane sheet"``.
            Defaults to ``"spherical"``.

    Returns:
        pd.DataFrame: Combined DataFrame containing both the raw experimental
        columns (``TC``, ``thr``, ``M``, ``delM``) and the computed diffusion
        parameters.
    """
    exp_data = pd.read_csv(input_csv_path, header=None)

    # Trim to the four expected columns if extra columns were read in
    if exp_data.shape[1] >= 4:
        exp_data = exp_data.loc[:, 1:4]

    exp_data.columns = ["TC", "thr", "M", "delM"]

    exp_results = calculate_diffusivity(exp_data, geometry)

    results = exp_data.join(exp_results)
    results.to_csv(output_csv_path)
    return results
