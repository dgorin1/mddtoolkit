import math
import numpy as np
import pandas as pd
import torch
import math as math


def forwardModelKinetics(kinetics, tsec, TC, geometry:str = "spherical", added_steps:int = 0):  #I NEED TO FIX THE ADDED STEPS PROBLEM HERE STILL
    # Define some necessary variables
    R = 0.008314 #gas constant
    torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)
    
    # Convert to a tensor for speed
    if not isinstance(kinetics, torch.Tensor):
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


    if geometry == "spherical":
        Dtaa = torch.cumsum(DtaaForSum, axis = 0)

        # Calculate f at each step
        Bt = Dtaa*torch.pi**2 # Use tabulation from XXXX to calculate
        f[Bt <= 1.401] = 6/(torch.pi**(3/2))*(torch.pi**2*Dtaa[Bt <= 1.401])**(1/2) - (3/(torch.pi**2))*(torch.pi**2*Dtaa[Bt <= 1.401])
        f[Bt > 1.401] = 1 - (6/(torch.pi**2))*torch.exp(-(torch.pi**2)*Dtaa[Bt > 1.401])


    elif geometry == "plane sheet":

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
    if added_steps > 0:

        newf = torch.zeros(sumf_MDD.shape)

        # Create newf, the gas fractions in noncumulative form.
        newf[0] = sumf_MDD[0]
        newf[1:] = sumf_MDD[1:]-sumf_MDD[0:-1]
        
        # Omit gas from the added steps
        newf = newf[added_steps:]
        
        # Calculate a punishment flag if the experiment degassed fully before the end of the experiment.
        # If true, then you will lose the ability to notice that after we re-normalize. 

        punishmentFlag = torch.round(sumf_MDD[-1,:],decimals = 3) < 1

        # Find the largest value in the newf, which will be used to renormalize the values.
        # Then proceed with the normalization.
        normalization_factor = torch.max(torch.cumsum(newf,0),axis=0).values
        diffFi= newf/normalization_factor 

        # Resum the gas fractions into cumulative space that doesn't include the two added steps
        sumf_MDD = torch.cumsum(diffFi,axis=0)
    else:
        punishmentFlag = torch.round(sumf_MDD[-1,:],decimals = 3) < 1
 

    # Turn all nans into zeros so that 
    nan_mask = torch.isnan(sumf_MDD).all(dim=0)
    if sum(nan_mask > 0):
        pass
    sumf_MDD[:,nan_mask]= 0.0


    return sumf_MDD,punishmentFlag
    


def calc_lnd0aa(sumf_MDD,diffti,geometry,extra_steps,added_steps):
    
    if len(sumf_MDD.size())>1: #if there are multiple entries
        diffti = diffti.unsqueeze(1).repeat(1,sumf_MDD.size()[1])
    if extra_steps == True:
        diffti = diffti[added_steps:]

    if geometry == "spherical":
        Daa_MDD_a = torch.zeros(sumf_MDD.shape)
        Daa_MDD_b = torch.zeros(sumf_MDD.shape)
        Daa_MDD_c = torch.zeros(sumf_MDD.shape)
        Daa_MDD_a[0] = ( (sumf_MDD[0]**2 - 0.**2 )*torch.pi/(36*(diffti[0])))


        # Equation 5a for all other steps

        diffFi = torch.zeros(sumf_MDD.shape)
        #diffFi[0] = sumf_MDD[0]
        diffFi = sumf_MDD[1:]-sumf_MDD[0:-1]


        Daa_MDD_a[0] = ( (sumf_MDD[0]**2 - 0.**2 )*torch.pi/(36*(diffti[0])))


   
        if len(diffti.shape) == 1:
            cumtsec = torch.cumsum(diffti,0)
        else:
            cumtsec = torch.cumsum(diffti,1)
        diffti = diffti[1:]
        Fi = sumf_MDD


        DR2_a = torch.zeros(sumf_MDD.shape)
        DR2_b = torch.zeros(sumf_MDD.shape)
        #Fi is cumulative F at each step i, cumtsec is cumulative t in seconds, diffFi is differential Fi at each step, so length is len(Fi)-1, diffti is analagously length n-1
        DR2_a[0] = 1/((torch.pi**2)*cumtsec[0])*(2*torch.pi - (torch.pi**2/3)*Fi[0] - 2*torch.pi*torch.sqrt(1-(torch.pi/3)*Fi[0]))
        DR2_a[1:] = (1/(torch.pi**2*diffti)) * (-torch.pi**2/3 * diffFi - 2*torch.pi*( torch.sqrt(1-(torch.pi/3)*Fi[1:]) - torch.sqrt(1-(torch.pi/3)*Fi[0:-1])))
        DR2_b[0] = -1/(torch.pi**2* cumtsec[0]) * torch.log((1-Fi[0]) * (torch.pi**2/6))
        DR2_b[1:] = -1/(torch.pi**2*diffti) * torch.log( (1-Fi[1:])/(1-Fi[0:-1]))


        # Decide which equation to use based on the cumulative gas fractions from each step
        use_a = (Fi<= 0.85) & (Fi> 0.00000001)
        use_b = Fi > 0.85
        
        # Compute the final values
        Daa_MDD = torch.nan_to_num(use_a*DR2_a) + use_b*DR2_b


    elif geometry == "plane sheet":

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


    return lnd0aa_MDD