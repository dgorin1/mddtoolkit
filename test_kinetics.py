import pandas as pd
import torch
import numpy as np
import math as math
from fwdmodelkinetics_for_test import forwardModelKinetics
from fwdmodelkinetics_for_test import calc_lnd0aa

# Import heating schedule from file
sched = pd.read_csv("schedule_no_extra_steps.csv")
TC = torch.tensor(sched["TC"].values)
tsec = torch.tensor(sched["t_hour"].values*3600)
breakpoint()
lookup_table = pd.read_csv("src/diffusion_kinetics/pipeline/production_to_production_plus_diffusion_ratio_table.csv")


x0 = [[90.79163129],	[18.11860881],	[16.15275072],	[15.11652312],	[13.94510743],	[12.05288458],	[9.120148509],	[7.542228132],	[5.384645069],	[-4.999682419],		[0.157886404],	[0.129898332]	,[0.22817251],	[0.115142899],	[0.065852832],	[0.083774195],	[0.059892712],	[0.155396737]] # kinetics to test...


sumf_MDD, out = forwardModelKinetics(x0, tsec, TC, lookup_table, added_steps=0)


out =  calc_lnd0aa(sumf_MDD, tsec, "spherical", True, 0)
breakpoint()

