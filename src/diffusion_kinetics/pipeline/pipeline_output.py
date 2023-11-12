import os

class PipelineOutput:
    """Class to handle the construction and storage of pipeline output.
    """
    def __init__(self, output_dir:str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        self.config_path = os.path.join(output_dir, "config.yaml")
        self._setup()
        
    def _setup(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            
    def get_plot_path(self, ndom:int, file_type:str="pdf"):
        """Get the path to a plot file.

        Args:
            ndom (int): Number of domains.
            file_type (str, optional): File type. Defaults to "pdf".

        Returns:
            str: Path to plot file.
        """
        return os.path.join(self.plots_dir, f"{ndom}domains.{file_type}")
    
    def get_results_path(self, ndom:int, file_type:str="csv"):
        """
        Args:
            ndom (int): Number of domains.
            file_type (str, optional): File type. Defaults to "csv".

        Returns:
            str: Path to results file.
        """
        return os.path.join(self.output_dir, f"{ndom}results.{file_type}")
    