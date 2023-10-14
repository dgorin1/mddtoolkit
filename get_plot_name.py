import os
import datetime as datetime



def get_plot_name(num_domains:int,plot_type, sample_name:str = "",extra_label:str = "",file_type:str = "pdf",moves_type:str = "",misfit_stat:str = ""):

    # Get the current date and time
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{num_domains}domains_{misfit_stat}_{moves_type}_{current_date}"
    # Create the folder if it doesn't exist
    if file_type == "h5":
        folder_name = os.path.join("MCMC_data",sample_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    else:
        folder_name = os.path.join("MCMC_plots", f"{sample_name}", run_name)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)



    # Generate the file name


    return os.path.join(folder_name, f"{num_domains}domains_{moves_type}_{plot_type}_{current_date}_{misfit_stat}.{file_type}")


    # Save the figure as a PDF file