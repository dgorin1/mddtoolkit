import pandas as pd
from diffusion_kinetics.preprocessing.calculate_experimental_results import D0calc_MonteCarloErrors
import os


def generate_inputs(
    nameOfInputCSVFile, nameOfExperimentalResultsFile, geometry: str = "spherical"
):
    """
    Generates the input file for the optimization problem from the experimental data.

    Right now, I've just been calling on this individually from the command line. If you do that
    it will run main, which executes the code. The deal is that you need to enter the name of the input csv file
    and the name of the output file and the desired geometry before you run this code. 
    I suppose this should maybe be a part of the gui?


    Args:
        nameOfInputCSVFile (str): the name of the input .csv file.
        nameOfExperimentalResultsFile (str): the name of the output .csv file.
        geometry (str, optional): the geometry of the sample. Defaults to "spherical".
    """

    expData = pd.read_csv(nameOfInputCSVFile, header=None)

    # If extra columns get read in, trim them down to just 3
    if expData.shape[1] >= 4:
        expData = expData.loc[:, 1:4]

    # Name the columsn of the iput data
    expData.columns = ["TC", "thr", "M", "delM"]

    # Calculate Daa from experimental results
    expResults = D0calc_MonteCarloErrors(expData, geometry)

    # Combine the diffusion parameters with the experimental setup (T, thr, M, delM)
    # to get a final dataframe that will be passed into the optimizer
    diffusionExperimentResults = expData.join(expResults)
    # Write dataframe to a .csv file
    diffusionExperimentResults.to_csv(nameOfExperimentalResultsFile)
    return diffusionExperimentResults