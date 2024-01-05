# Create the synthetic data

import torch as torch
import numpy as np
import pandas as pd
import os 

# Example of well behaved sample
fracs = torch.tensor([0.0253, 0.0969, 0.0824, 0.2263, 0.1266, 0.2730, 0.1696])
Ea = torch.tile(torch.tensor(200.219193288), (1,7))
lnD0aa = [19.4483046687238, 16.1765962791235,	13.9300726859056,	8.89366400206864,	8.094775342,	7.51289980055765,	6.97580077979784]

#Example of poorly behaved sample
# gas_moved = np.array(0.01)
# fracs = torch.tensor(np.append(gas_moved, ( fracs*(1-gas_moved))))

dir_path = os.path.dirname(os.path.realpath(__file__))

# Ea = torch.tile(torch.tensor(200.219193288), (1,8))
# lnD0aa = torch.tensor([23.8, 19.4483046687238, 16.1765962791235,	13.9300726859056,	8.89366400206864,	8.67665088325817,	7.51289980055765,	6.97580077979784])
#Take gas from all other domains and transfer into a new domain on top...


data_input = pd.read_csv(f"{dir_path}/syntheticArArKspar.csv", names = ["T","t"])

total_moles = 5.2129E-12

TC = torch.tensor(data_input["T"])
tsec = torch.tensor(data_input["t"]*3600)
kinetics = [200.219193288, 19.4483046687238, 16.1765962791235,	13.9300726859056,	8.89366400206864,	8.67665088325817,	7.51289980055765,	6.97580077979784, 0.0253, 0.0969, 0.0824, 0.2263, 0.1266, 0.2730]
 # Define some necessary variables
R = 0.008314 #gas constant
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)


# Convert to a tensor for speed
kinetics = torch.tensor(kinetics)

# Check the shape of kinetics and make sure it's a tensor in the shape [num_parameters, num_input_vectors_to_test]

# If dimension is <= 1, then we need to unsqueeze it so it's a 2D tensor
if kinetics.ndim <= 1:
    kinetics = torch.unsqueeze(kinetics,1)
    num_vectors = kinetics.shape[1]


# Infer the number of domains from input
ndom = (len(kinetics))//2

# Separate the kinetics vector into its components so that calculations can be performed
Ea = kinetics[0]   # Moles aren't passed into this function, so first entry of kinetics is Ea

# Assign the various parameters to variables and put them in the correct shapes
lnD0aa = kinetics[1:ndom+1].unsqueeze(0).expand(len(TC), ndom, -1)
fracstemp = kinetics[ndom+1:]
fracs = torch.cat((fracstemp, 1 - torch.sum(fracstemp, axis=0, keepdim=True))).unsqueeze(0).expand(len(TC), -1, -1)
Ea = Ea.unsqueeze(0).expand(len(TC),ndom,-1)
cumtsec = torch.cumsum(tsec,dim=0).unsqueeze(-1).repeat(1,ndom) # Expand dimensions to match the desired shape
cumtsec = cumtsec.unsqueeze(2).repeat(1,1,num_vectors)
tsec = tsec.unsqueeze(-1).repeat(1,ndom)
tsec = tsec.unsqueeze(2).repeat(1,1,num_vectors)
TK = (TC+273.15).unsqueeze(-1).repeat(1,ndom)
TK = TK.unsqueeze(2).repeat(1,1,num_vectors)


# Calculate D/a^2 for each domain
Daa = torch.exp(lnD0aa)*torch.exp(-Ea/(R*TK))

# Pre-allocate fraction and Dtaa
f = torch.zeros(Daa.shape)
Dtaa = torch.zeros(Daa.shape)
DtaaForSum = torch.zeros(Daa.shape)

# Calculate Dtaa in incremental (not cumulative) form including the added heating steps
DtaaForSum[0,:,:] = Daa[0,:,:]*tsec[0,:,:]
DtaaForSum[1:,:,:] = Daa[1:,:,:]*(cumtsec[1:,:,:]-cumtsec[0:-1,:,:])


Dtaa = torch.cumsum(DtaaForSum, axis = 0)
Bt = Dtaa*torch.pi**2
f = (2/torch.sqrt(torch.pi))*torch.sqrt((Dtaa))
f[f > 0.6] = 1-(8/(torch.pi**2))*torch.exp(-1*torch.pi**2*Dtaa[f > 0.6]/4)


# Multiply each gas realease by the percent gas located in each domain (prescribed by input)
f_MDD = f*fracs

# Renormalize everything by first calculating the fractional releases at each step, summing back up, 
# and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
sumf_MDD = torch.sum(f_MDD,axis=1)

# If added steps are used, then we need to remove them and renormalize the results 
# so that it appears that we hadn't measured the gas from the first X steps.


punishmentFlag = torch.round(sumf_MDD[-1,:],decimals = 3) < 1

# Turn all nans into zeros so that 
nan_mask = torch.isnan(sumf_MDD).all(dim=0)

sumf_MDD[:,nan_mask]= 0.0



trueFracMDD = sumf_MDD[1:] - sumf_MDD[0:-1]
trueFracMDD = torch.concat(
    (torch.unsqueeze(sumf_MDD[0], dim=0), trueFracMDD), dim=0
)

uncertainty = torch.tensor([3.9884E-02,5.2077E-02,1.1747E-02,1.9902E-02,4.9049E-03,1.1822E-02,4.8722E-03,9.4712E-03,3.7671E-03,3.9967E-03,2.2163E-03,3.9967E-03,2.6630E-03,3.8487E-03,2.2125E-03, \
4.4761E-03,4.3084E-03,3.6335E-03,2.8744E-03,2.2996E-03,2.4725E-03,1.7816E-03,1.5179E-03,1.1926E-03,1.7623E-03,9.2805E-04,1.4024E-03,1.5502E-03,1.6186E-03, \
1.0231E-03,2.1546E-03,2.7016E-03,1.5382E-03,3.3274E-03,2.0579E-03,1.7519E-03,1.1106E-03,2.1456E-03,5.7535E-03,7.9534E-03,8.9813E-03,1.8657E-01])



# get other variables ready to go the other direction


tsec = tsec[:,0,0]

M = trueFracMDD*total_moles
delM = M.ravel()*uncertainty
diffti = tsec




#def calc_lnd0aa(sumf_MDD,diffti,geometry,extra_steps,added_steps):
## here's calc lnd0aa to calculate the rest of the stuff
if len(sumf_MDD.size())>1: #if there are multiple entries
    diffti = diffti.unsqueeze(1).repeat(1,sumf_MDD.size()[1])




DR2_a = torch.zeros(sumf_MDD.shape)
DR2_b = torch.zeros(sumf_MDD.shape)


#Fechtig and Kalbitzer Equation 5a

DR2_a[0] = ((((sumf_MDD[0]**2) - 0**2))*torch.pi)/(4*diffti[0])
DR2_a[1:] = ((((sumf_MDD[1:]**2)-(sumf_MDD[0:-1])**2))*torch.pi)/(4*diffti[1:])
DR2_b[1:] = (4/((torch.pi**2)*diffti[1:]))*torch.log((1-sumf_MDD[0:-1])/(1-sumf_MDD[1:]))
usea = (sumf_MDD > 0) & (sumf_MDD < 0.6)
useb = (sumf_MDD >= 0.6) & (sumf_MDD <= 1)

Daa_MDD = usea*DR2_a + useb*DR2_b
    
lnd0aa_MDD = torch.log(Daa_MDD)

output = pd.DataFrame({"TC": TC.unsqueeze(1).ravel(), "thr": (tsec/3600).unsqueeze(1).ravel(), "M": M.ravel(), "delM": delM.unsqueeze(1).ravel()})



dir_path = os.path.dirname(os.path.realpath(__file__))

nameOfCSVFile = f"{dir_path}/N13ksp_good_example.csv"
breakpoint()
output.to_csv(nameOfCSVFile)
