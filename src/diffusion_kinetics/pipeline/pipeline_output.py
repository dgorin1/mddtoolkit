import os

class PipelineOutput:
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
        return os.path.join(self.plots_dir, f"{ndom}domains.{file_type}")
    
    def get_results_path(self, ndom:int, file_type:str="csv"):
        return os.path.join(self.output_dir, f"{ndom}results.{file_type}")
    