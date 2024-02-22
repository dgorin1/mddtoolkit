from src import diffusion_kinetics


breakpoint()
dir_path = os.path.dirname(os.path.realpath(__file__))

data_input = pd.read_csv(f"{dir_path}/data/input_N13ksp_python_test.csv")


# Plot results...



# def plot_results(
#     params,
#     dataset,
#     objective,
#     plot_path:str,
# ):


params_low = [1.73365444341471E-13,	3.91457517581200E+02,	4.62200768821121E+01,	3.90180663316288E+01,	3.28200751498657E+01,	2.72285786410262E+01,	2.24478990017536E+01,	6.93795393437542E+00,	4.81875188588035E+00,	2.22805911996708E-01,	6.39612889665484E-02,	9.76777710994787E-02,	3.73290912608320E-02,	1.98939822739980E-01,	3.52387260186070E-01,	7.29078123887070E-02,	4.66218454753439E-02,	1.30175107883039E-01]


dataset = Dataset("diffEV", data_input)