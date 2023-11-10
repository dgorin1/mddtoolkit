import os
import numpy as np
from diffusion_kinetics.pipeline import PipelineOutput


def save_results(
    ndom:int, params:list[float], pipe_out:PipelineOutput
):
    """
    Save the resulting parameters for an optimization run to a CSV.

    Args:
        - sample_name (str, optional): The name of the sample. Defaults to "".
        - params (list, optional): The parameters from the optimization. Defaults to [].
        - moves_type (str, optional): The type of moves used in the optimization. Defaults to "".
    """

    name = pipe_out.get_results_path(ndom)

    with open(name, "w", newline="") as file:
        # writer = csv.writer(file)
        # writer.writerow(np.append(params,misfit))
        np.savetxt(file, params, delimiter=",")
