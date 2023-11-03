
from optimization.forwardModelKinetics import forwardModelKinetics
import torch as torch
import pandas as pd
import os
import numpy as np

# get this file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))

input = pd.read_csv(f"{dir_path}/data/input_8DomSynthData_spherical.csv")

TC = input.TC 
thr = input.thr


kinetics = torch.tensor([2.301827982315843997e+02, 2.695453701006638170e+01,	2.399562245075363975e+01,	2.162033080439059063e+01,	2.005857470165684120e+01,	1.840672131604174666e+01,	1.441347501664621511e+01,	1.146143697456085064e+01,	7.403164238243396866e+00 ,3.407813798677972184e-02,	6.168645862319671602e-02,	5.323458631337435465e-02,	3.165096967255043303e-02	,2.695325102213830260e-02,	2.569035505844752265e-01,	3.566193305741533481e-01]).T
temp = forwardModelKinetics(kinetics,torch.tensor(thr*3600),torch.tensor(TC),"spherical",)

uncertainty = torch.tensor([3.9884E-02,
5.2077E-02,
1.1747E-02,
1.9902E-02,
4.9049E-03,
1.1822E-02,
4.8722E-03,
9.4712E-03,
3.7671E-03,
3.9967E-03,
2.2163E-03,
3.9967E-03,
2.6630E-03,
3.8487E-03,
2.2125E-03,
4.4761E-03,
4.3084E-03,
3.6335E-03,
2.8744E-03,
2.2996E-03,
2.4725E-03,
1.7816E-03,
1.5179E-03,
1.1926E-03,
1.7623E-03,
9.2805E-04,
1.4024E-03,
1.5502E-03,
1.6186E-03,
1.0231E-03,
2.1546E-03,
2.7016E-03,
1.5382E-03,
3.3274E-03,
2.0579E-03,
1.7519E-03,
1.1106E-03,
2.1456E-03,
5.7535E-03,
7.9534E-03,
8.9813E-03,
1.8657E-01]).T

total_moles = 4*10**9


moles = np.array((temp[0][1:]-temp[0][0:-1]).ravel().tolist())
moles = np.append(temp[0][0].item(),moles)
moles = moles*total_moles
#moles = temp[0]*total_moles
breakpoint()
moles_uncertainty = moles*np.array(uncertainty)

data_out = pd.DataFrame({'TC': TC, 'thr': thr, 'M': moles.ravel(), 'delM':moles_uncertainty})

data_out.to_csv(f"{dir_path}/data/spherical_test.csv")
