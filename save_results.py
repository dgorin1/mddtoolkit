import os
import datetime as datetime
import csv
import numpy as np


def save_results(
    sample_name: str = "", misfit_stat: str = "", params=[], moves_type=""
):
    """
    Generate a plot based on the number of domains, the type of plot, and the sample name.

    Args:
        sample_name (str, optional): The name of the sample. Defaults to "".
        misfit_stat (str, optional): The type of misfit statistic used in the optimization. Defaults to "".
        params (list, optional): The parameters from the optimization. Defaults to [].
        moves_type (str, optional): The type of moves used in the optimization. Defaults to "".
    """
    # Get the current date and time

    run_name = f"{misfit_stat}"

    folder_name = os.path.join("results", f"{sample_name}", run_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Generate the file name

    name = os.path.join(folder_name, f"output_kinetics.csv")

    with open(name, "w", newline="") as file:
        # writer = csv.writer(file)
        # writer.writerow(np.append(params,misfit))
        np.savetxt(file, params, delimiter=",")
    # Save the figure as a PDF file
