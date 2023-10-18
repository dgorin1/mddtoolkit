import os
import datetime as datetime
from diffusion_kinetics import RESULTS_DIR

def get_plot_name(
    num_domains: int,
    plot_type,
    sample_name: str = "",
    extra_label: str = "",
    file_type: str = "pdf",
    moves_type: str = "",
    misfit_stat: str = "",
):
    """
    Generate a name for a plot based on the number of domains, the type of plot,
    and the sample name. The plot will be saved in the results folder.

    Args:
        num_domains (int): The number of domains in the optimization problem.
        plot_type (str): The type of plot to generate (e.g., "arrhenius", "moles", "misfit", "bounds").
        sample_name (str, optional): The name of the sample. Defaults to "".
        extra_label (str, optional): An extra label to add to the plot name. Defaults to "".
        file_type (str, optional): The file type to save the plot as. Defaults to "pdf".
        moves_type (str, optional): The type of moves used in the optimization. Defaults to "".
        misfit_stat (str, optional): The type of misfit statistic used in the optimization. Defaults to "".

    Returns:
        str: The path to the created plot.
    """
    # Get the current date and time
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    run_name = f"{misfit_stat}"
    # Create the folder if it doesn't exist

    folder_name = os.path.join(RESULTS_DIR, f"{sample_name}", run_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Generate the file name

    return os.path.join(folder_name, f"{num_domains}domains.{file_type}")

    # Save the figure as a PDF file
