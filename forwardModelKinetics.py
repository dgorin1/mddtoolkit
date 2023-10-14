import math
import numpy as np
import pandas as pd
import torch
import math as math
from jax import numpy as jnp

def forwardModelKineticsDiffEV(kinetics, lookup_table,tsec,TC, geometry:str = "spherical"): 


    # Check the number of dimensions being passed in to see how many vectors we're dealing with. Code handles 1 vs >1 differently
    if kinetics.ndim > 1:
        num_vectors = len(kinetics[0,:])
    else:
        num_vectors = 1
    

    # Infer the number of domains from input
    if len(kinetics) <= 3:
        ndom = 1
    else:
        ndom = (len(kinetics))//2

    # Convert to a tensor for speed

    kinetics = torch.tensor(kinetics)
    Ea = kinetics[0] # Moles isn't passed into this function, so first entry of kinetics is Ea
    kinetics = kinetics[1:] 
    temp = kinetics[1:]
    # kinetics: (Ea, lnd0aa_x, fracs_x). To make this compatible with other functions, if there are x fracs, input x-1 fractions, and the code will determine the
    # final fraction.

    R = 0.008314 #gas constant
    torch.pi = torch.acos(torch.zeros(1)).item() * 2


    if num_vectors == 1:
  
        lnD0aa = torch.tile(kinetics[0:ndom].T,(len(TC),1)) # Do this for LnD0aa
        fracstemp = kinetics[ndom:] # Grab fracs that were input (one will be missing because it is pre-determined by the others)

        fracs = torch.tile(torch.concat((fracstemp,1-torch.sum(fracstemp,axis=0,keepdim=True)),axis=0).T,(len(TC),1)) # Add the last frac as 1-sum(other fracs)
        Ea = torch.tile(Ea,(len(TC),ndom)) # Do for Ea

    


        # Put time and cumulative time in the correct shape
        if ndom > 1:
            tsec = torch.tile(torch.reshape(tsec,(-1,1)),(1,Ea.shape[1])) #This is a complicated-looking way of getting tsec into a numdom x numstep matrix for multiplication
            cumtsec = torch.tile(torch.reshape(torch.cumsum(tsec[:,1],dim=0),(-1,1)),(1,Ea.shape[1])) #Same as above, but for cumtsec        
            # Convert TC to TK and put in correct shape for quick computation                                                 
            TK = torch.tile(torch.reshape((TC + 273.15),(-1,1)),(1,Ea.shape[1])) #This is a complicated-looking way of turning TC from a 1-d array to a 2d array and making two column copies of it

        else:
            cumtsec = torch.reshape(torch.cumsum(tsec,-1),(-1,1))
            TK = torch.reshape(TC+273.15,(-1,1))
            tsec = torch.reshape(tsec,(-1,1))

        # Calculate D/a^2 for each domain

        Daa = torch.exp(lnD0aa)*torch.exp(-Ea/(R*TK))

        # Pre-allocate fraction and Dtaa
        f = torch.zeros(Daa.shape)
        Dtaa = torch.zeros(Daa.shape)
        DtaaForSum = torch.zeros(Daa.shape)


        # Calculate Dtaa in incremental (not cumulative) form including the added heating steps

        DtaaForSum[0,:] = Daa[0,:]*tsec[0,:]
        DtaaForSum[1:,:] = Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])

        if geometry == "spherical":
            # # Make the correction for P_D vs D_only
            # for i in range(len(DtaaForSum[0,:])): #This is a really short loop... range of i is # domains. Maybe we could vectorize to improve performance?
            #     if DtaaForSum[0,i] <= 1.347419e-17:
            #         DtaaForSum[0,i] *= 0
            #     elif DtaaForSum[0,i] >= 4.698221e-06:
            #         pass
            #     else:
            #         DtaaForSum[0,i] *= lookup_table(DtaaForSum[0,i])

            # Calculate Dtaa in cumulative form.
            Dtaa = torch.cumsum(DtaaForSum, axis = 0)


            # Calculate f at each step
            Bt = Dtaa*torch.pi**2
            f = (6/(math.pi**(3/2)))*torch.sqrt((math.pi**2)*Dtaa)
            f[Bt>0.0091] = (6/(torch.pi**(3/2)))*torch.sqrt((torch.pi**2)*Dtaa[Bt>0.0091])-(3/(torch.pi**2))* \
                    ((torch.pi**2)*Dtaa[Bt>0.0091])
            f[Bt >1.8] = 1 - (6/(torch.pi**2))*torch.exp(-(torch.pi**2)*Dtaa[Bt > 1.8])
            


            # Multiply each gas realease by the percent gas located in each domain (prescribed by input)
        elif geometry == "plane sheet":
            # Need to derive a correction for the plane sheet... for now I just won't do an irradiation correction
            Dtaa = torch.cumsum(DtaaForSum, axis = 0)
            f = (2/np.sqrt(math.pi))*np.sqrt((Dtaa))
            f[f > 0.6] = 1-(8/(math.pi**2))*np.exp(-math.pi**2*Dtaa[f > 0.6]/4)

        f_MDD = f*fracs

        # Renormalize everything by first calculating the fractional releases at each step, summing back up, 
        # and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
        sumf_MDD = torch.sum(f_MDD,axis=1)

        # If the second heating step gets gas release all the way to 100%, then the rest of the calculation is not necessary. 
        # Return that sumf_MDD == 0
        if (torch.round(sumf_MDD[2],decimals=6) == 1):
            return torch.zeros(len(sumf_MDD)-2),0
            

        # Remove the two steps we added, recalculate the total sum, and renormalize.
        newf = torch.zeros(sumf_MDD.shape)
        newf[0] = sumf_MDD[0]
        newf[1:] = sumf_MDD[1:]-sumf_MDD[0:-1]

        newf = newf[2:]
        normalization_factor = torch.max(torch.cumsum(newf,0))

        punishmentFlag = torch.round(sumf_MDD[-1],decimals=3) < 1.0
        #punishmentFlag = torch.round(newf[-1,:],decimals = 5) < 1
        
        diffFi= newf/normalization_factor 


        # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
        # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
        # special case when i = 1; need to insert 0 for previous amount released



        # Resum the gas fractions into cumulative space that doesn't include the two added steps
        sumf_MDD = torch.cumsum(diffFi,axis=0)
  
        return sumf_MDD, punishmentFlag 


    else:
        lnD0aa = kinetics[0:ndom].unsqueeze(0).expand(len(TC), ndom, -1)
        fracstemp = kinetics[ndom:]
        fracs = torch.cat((fracstemp, 1 - torch.sum(fracstemp, axis=0, keepdim=True))).unsqueeze(0).expand(len(TC), -1, -1)
        Ea = Ea.unsqueeze(0).expand(len(TC),ndom,-1)



    # THIS IS TEMPORARY-- WE NEED TO ADD THIS AS AN INPUT.. THE INPUTS WILL NEED TO BE
    # 1. Duration of irradiation
    # 2. Temperature during irradiation
    # 3. Duration of lab storage
    # 4. Temperature during lab storage

    # We might also want to make this all optional at some point, since some minerals are so retentive 
    # that they wont lease any helium during irradiation and storage.


        if ndom > 1:
            cumtsec = torch.cumsum(tsec,dim=0).unsqueeze(-1).repeat(1,ndom) # Expand dimensions to match the desired shape
            cumtsec = cumtsec.unsqueeze(2).repeat(1,1,num_vectors)

            TK = (TC+273.15).unsqueeze(-1).repeat(1,ndom)
            TK = TK.unsqueeze(2).repeat(1,1,num_vectors)

            tsec = tsec.unsqueeze(-1).repeat(1,ndom)
            tsec = tsec.unsqueeze(2).repeat(1,1,num_vectors)

            # Expand lnD0aa to match the desired shape



        else:

            cumtsec = torch.cumsum(tsec,dim=0).unsqueeze(-1).repeat(1,ndom) # Expand dimensions to match the desired shape
            cumtsec = cumtsec.unsqueeze(2).repeat(1,1,num_vectors)

            TK = (TC+273.15).unsqueeze(-1).repeat(1,ndom)
            TK = TK.unsqueeze(2).repeat(1,1,num_vectors)

            tsec = tsec.unsqueeze(-1).repeat(1,ndom)
            tsec = tsec.unsqueeze(2).repeat(1,1,num_vectors)

            # Expand lnD0aa to match the desired shape


        # Calculate D/a^2 for each domain
        Daa = torch.exp(lnD0aa)*torch.exp(-Ea/(R*TK))
        # Pre-allocate fraction and Dtaa
        f = torch.zeros(Daa.shape)
        Dtaa = torch.zeros(Daa.shape)
        DtaaForSum = torch.zeros(Daa.shape)


        # Calculate Dtaa in incremental (not cumulative) form including the added heating steps
        if num_vectors > 1:

            DtaaForSum[0,:,:] = Daa[0,:,:]*tsec[0,:,:]
            DtaaForSum[1:,:,:] = Daa[1:,:,:]*(cumtsec[1:,:,:]-cumtsec[0:-1,:,:])
        else:
            DtaaForSum[0,:] = Daa[0,:]*tsec[0,:]
            DtaaForSum[1:,:,:] = Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])
        if geometry == "spherical":
            # # Make the correction for P_D vs D_only
            # for j in range(len(DtaaForSum[0,0,:])    ):
            #     for i in range(len(DtaaForSum[0,:,0])): #This is a really short loop... range of i is # domains. Maybe we could vectorize to improve performance?
            #         if DtaaForSum[0,i,j] <= 1.347419e-17:
            #             DtaaForSum[0,i,j] *= 0
            #         elif DtaaForSum[0,i,j] >= 4.698221e-06:
            #             pass
            #         else:
            #             DtaaForSum[0,i,j] *= lookup_table(DtaaForSum[0,i,j])

            # Calculate Dtaa in cumulative form.
            Dtaa = torch.cumsum(DtaaForSum, axis = 0)


            # Calculate f at each step
            Bt = Dtaa*torch.pi**2

            f = (6/(math.pi**(3/2)))*torch.sqrt((math.pi**2)*Dtaa)
            f[Bt>0.0091] = (6/(torch.pi**(3/2)))*torch.sqrt((torch.pi**2)*Dtaa[Bt>0.0091])-(3/(torch.pi**2))* ((torch.pi**2)*Dtaa[Bt>0.0091])
            f[Bt >1.8] = 1 - (6/(torch.pi**2))*torch.exp(-(torch.pi**2)*Dtaa[Bt > 1.8])

        elif geometry == "plane sheet":
            # Need to derive a correction for the plane sheet... for now I just won't do an irradiation correction

            Dtaa = torch.cumsum(DtaaForSum, axis = 0)
            f = (2/np.sqrt(math.pi))*np.sqrt((Dtaa))
            f[f > 0.6] = 1-(8/(math.pi**2))*np.exp(-1*math.pi**2*Dtaa[f > 0.6]/4)



        # Multiply each gas realease by the percent gas located in each domain (prescribed by input)
        f_MDD = f*fracs

        # Renormalize everything by first calculating the fractional releases at each step, summing back up, 
        # and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
        sumf_MDD = torch.sum(f_MDD,axis=1)

        # If the second heating step gets gas release all the way to 100%, then the rest of the calculation is not necessary. 
        # Return that sumf_MDD == 0
        if num_vectors == 1:

            if (torch.round(sumf_MDD[2],decimals=6) == 1):

                return torch.zeros(len(sumf_MDD)-2),0
            


        newf = torch.zeros(sumf_MDD.shape)
        newf[0] = sumf_MDD[0]
        newf[1:] = sumf_MDD[1:]-sumf_MDD[0:-1]

        newf = newf[2:]

        normalization_factor = torch.max(torch.cumsum(newf,0),axis=0).values
    
        punishmentFlag = torch.round(sumf_MDD[-1,:],decimals = 3) < 1

        diffFi= newf/normalization_factor 




        # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
        # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
        # special case when i = 1; need to insert 0 for previous amount released

        # has_positive_value = (tensor > 0).any()

        # # Add a breakpoint if positive values are present
        # if has_positive_value:
        #     breakpoint()

        # Resum the gas fractions into cumulative space that doesn't include the two added steps
        sumf_MDD = torch.cumsum(diffFi,axis=0)
        nan_mask = torch.isnan(sumf_MDD).all(dim=0)
        sumf_MDD[:,nan_mask]= 0.0

     
        return sumf_MDD,punishmentFlag
    



 #This function will not do the re-normalize step and is for when there is no irradiation or other lab storage heating to be considered
def forward_model_kinetics_no_extra_heating(kinetics, lookup_table,tsec,TC, geometry:str = "spherical"): 


    # Check the number of dimensions being passed in to see how many vectors we're dealing with. Code handles 1 vs >1 differently
    if kinetics.ndim > 1:
        num_vectors = len(kinetics[0,:])
    else:
        num_vectors = 1
    

    # Infer the number of domains from input
    if len(kinetics) <= 3:
        ndom = 1
    else:
        ndom = (len(kinetics))//2

    # Convert to a tensor for speed

    kinetics = torch.tensor(kinetics)
    Ea = kinetics[0] # Moles isn't passed into this function, so first entry of kinetics is Ea
    kinetics = kinetics[1:] 
    temp = kinetics[1:]
    # kinetics: (Ea, lnd0aa_x, fracs_x). To make this compatible with other functions, if there are x fracs, input x-1 fractions, and the code will determine the
    # final fraction.

    R = 0.008314 #gas constant
    torch.pi = torch.acos(torch.zeros(1)).item() * 2


    if num_vectors == 1:
  
        lnD0aa = torch.tile(kinetics[0:ndom].T,(len(TC),1)) # Do this for LnD0aa
        fracstemp = kinetics[ndom:] # Grab fracs that were input (one will be missing because it is pre-determined by the others)

        fracs = torch.tile(torch.concat((fracstemp,1-torch.sum(fracstemp,axis=0,keepdim=True)),axis=0).T,(len(TC),1)) # Add the last frac as 1-sum(other fracs)
        Ea = torch.tile(Ea,(len(TC),ndom)) # Do for Ea

    


        # Put time and cumulative time in the correct shape
        if ndom > 1:
            tsec = torch.tile(torch.reshape(tsec,(-1,1)),(1,Ea.shape[1])) #This is a complicated-looking way of getting tsec into a numdom x numstep matrix for multiplication
            cumtsec = torch.tile(torch.reshape(torch.cumsum(tsec[:,1],dim=0),(-1,1)),(1,Ea.shape[1])) #Same as above, but for cumtsec        
            # Convert TC to TK and put in correct shape for quick computation                                                 
            TK = torch.tile(torch.reshape((TC + 273.15),(-1,1)),(1,Ea.shape[1])) #This is a complicated-looking way of turning TC from a 1-d array to a 2d array and making two column copies of it

        else:
            cumtsec = torch.reshape(torch.cumsum(tsec,-1),(-1,1))
            TK = torch.reshape(TC+273.15,(-1,1))
            tsec = torch.reshape(tsec,(-1,1))

        # Calculate D/a^2 for each domain

        Daa = torch.exp(lnD0aa)*torch.exp(-Ea/(R*TK))

        # Pre-allocate fraction and Dtaa
        f = torch.zeros(Daa.shape)
        Dtaa = torch.zeros(Daa.shape)
        DtaaForSum = torch.zeros(Daa.shape)


        # Calculate Dtaa in incremental (not cumulative) form including the added heating steps

        DtaaForSum[0,:] = Daa[0,:]*tsec[0,:]
        DtaaForSum[1:,:] = Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])

        if geometry == "spherical":
            # # Make the correction for P_D vs D_only
            # for i in range(len(DtaaForSum[0,:])): #This is a really short loop... range of i is # domains. Maybe we could vectorize to improve performance?
            #     if DtaaForSum[0,i] <= 1.347419e-17:
            #         DtaaForSum[0,i] *= 0
            #     elif DtaaForSum[0,i] >= 4.698221e-06:
            #         pass
            #     else:
            #         DtaaForSum[0,i] *= lookup_table(DtaaForSum[0,i])

            # Calculate Dtaa in cumulative form.
            Dtaa = torch.cumsum(DtaaForSum, axis = 0)


            # Calculate f at each step
            Bt = Dtaa*torch.pi**2
            f = (6/(math.pi**(3/2)))*torch.sqrt((math.pi**2)*Dtaa)
            f[Bt>0.0091] = (6/(torch.pi**(3/2)))*torch.sqrt((torch.pi**2)*Dtaa[Bt>0.0091])-(3/(torch.pi**2))* \
                    ((torch.pi**2)*Dtaa[Bt>0.0091])
            f[Bt >1.8] = 1 - (6/(torch.pi**2))*torch.exp(-(torch.pi**2)*Dtaa[Bt > 1.8])
            


            # Multiply each gas realease by the percent gas located in each domain (prescribed by input)
        elif geometry == "plane sheet":
            # Need to derive a correction for the plane sheet... for now I just won't do an irradiation correction

            Dtaa = torch.cumsum(DtaaForSum, axis = 0)
            f = (2/np.sqrt(math.pi))*np.sqrt((Dtaa))
            f[f > 0.6] = 1-(8/(math.pi**2))*np.exp(-math.pi**2*Dtaa[f > 0.6]/4)

        f_MDD = f*fracs

        # Renormalize everything by first calculating the fractional releases at each step, summing back up, 
        # and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
        sumf_MDD = torch.sum(f_MDD,axis=1) 
        punishmentFlag = torch.round(sumf_MDD[-1],decimals=3) < 1.0
        return sumf_MDD, punishmentFlag  


    else:
        lnD0aa = kinetics[0:ndom].unsqueeze(0).expand(len(TC), ndom, -1)
        fracstemp = kinetics[ndom:]
        fracs = torch.cat((fracstemp, 1 - torch.sum(fracstemp, axis=0, keepdim=True))).unsqueeze(0).expand(len(TC), -1, -1)
        Ea = Ea.unsqueeze(0).expand(len(TC),ndom,-1)



    # THIS IS TEMPORARY-- WE NEED TO ADD THIS AS AN INPUT.. THE INPUTS WILL NEED TO BE
    # 1. Duration of irradiation
    # 2. Temperature during irradiation
    # 3. Duration of lab storage
    # 4. Temperature during lab storage

    # We might also want to make this all optional at some point, since some minerals are so retentive 
    # that they wont lease any helium during irradiation and storage.


        if ndom > 1:
            cumtsec = torch.cumsum(tsec,dim=0).unsqueeze(-1).repeat(1,ndom) # Expand dimensions to match the desired shape
            cumtsec = cumtsec.unsqueeze(2).repeat(1,1,num_vectors)

            TK = (TC+273.15).unsqueeze(-1).repeat(1,ndom)
            TK = TK.unsqueeze(2).repeat(1,1,num_vectors)

            tsec = tsec.unsqueeze(-1).repeat(1,ndom)
            tsec = tsec.unsqueeze(2).repeat(1,1,num_vectors)

            # Expand lnD0aa to match the desired shape



        else:

            cumtsec = torch.cumsum(tsec,dim=0).unsqueeze(-1).repeat(1,ndom) # Expand dimensions to match the desired shape
            cumtsec = cumtsec.unsqueeze(2).repeat(1,1,num_vectors)

            TK = (TC+273.15).unsqueeze(-1).repeat(1,ndom)
            TK = TK.unsqueeze(2).repeat(1,1,num_vectors)

            tsec = tsec.unsqueeze(-1).repeat(1,ndom)
            tsec = tsec.unsqueeze(2).repeat(1,1,num_vectors)

            # Expand lnD0aa to match the desired shape


        # Calculate D/a^2 for each domain
        Daa = torch.exp(lnD0aa)*torch.exp(-Ea/(R*TK))
        # Pre-allocate fraction and Dtaa
        f = torch.zeros(Daa.shape)
        Dtaa = torch.zeros(Daa.shape)
        DtaaForSum = torch.zeros(Daa.shape)


        # Calculate Dtaa in incremental (not cumulative) form including the added heating steps
        if num_vectors > 1:

            DtaaForSum[0,:,:] = Daa[0,:,:]*tsec[0,:,:]
            DtaaForSum[1:,:,:] = Daa[1:,:,:]*(cumtsec[1:,:,:]-cumtsec[0:-1,:,:])
        else:
            DtaaForSum[0,:] = Daa[0,:]*tsec[0,:]
            DtaaForSum[1:,:,:] = Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])
        if geometry == "spherical":
            # # Make the correction for P_D vs D_only
            # for j in range(len(DtaaForSum[0,0,:])    ):
            #     for i in range(len(DtaaForSum[0,:,0])): #This is a really short loop... range of i is # domains. Maybe we could vectorize to improve performance?
            #         if DtaaForSum[0,i,j] <= 1.347419e-17:
            #             DtaaForSum[0,i,j] *= 0
            #         elif DtaaForSum[0,i,j] >= 4.698221e-06:
            #             pass
            #         else:
            #             DtaaForSum[0,i,j] *= lookup_table(DtaaForSum[0,i,j])

            # Calculate Dtaa in cumulative form.
            Dtaa = torch.cumsum(DtaaForSum, axis = 0)


            # Calculate f at each step
            Bt = Dtaa*torch.pi**2

            f = (6/(math.pi**(3/2)))*torch.sqrt((math.pi**2)*Dtaa)
            f[Bt>0.0091] = (6/(torch.pi**(3/2)))*torch.sqrt((torch.pi**2)*Dtaa[Bt>0.0091])-(3/(torch.pi**2))* ((torch.pi**2)*Dtaa[Bt>0.0091])
            f[Bt >1.8] = 1 - (6/(torch.pi**2))*torch.exp(-(torch.pi**2)*Dtaa[Bt > 1.8])

        elif geometry == "plane sheet":
            # Need to derive a correction for the plane sheet... for now I just won't do an irradiation correction
            Dtaa = torch.cumsum(DtaaForSum, axis = 0)
            
            f = (2/np.sqrt(math.pi))*np.sqrt((Dtaa))
            f[f > 0.6] = 1-(8/(math.pi**2))*np.exp(-math.pi**2*Dtaa[f > 0.6]/4)



        # Multiply each gas realease by the percent gas located in each domain (prescribed by input)
        f_MDD = f*fracs

        # Renormalize everything by first calculating the fractional releases at each step, summing back up, 
        # and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
        sumf_MDD = torch.sum(f_MDD,axis=1)

        # If the second heating step gets gas release all the way to 100%, then the rest of the calculation is not necessary. 

        nan_mask = torch.isnan(sumf_MDD).all(dim=0)
        sumf_MDD[:,nan_mask]= 0.0

        punishmentFlag = torch.round(sumf_MDD[-1,:],decimals = 3) < 1
        return sumf_MDD, punishmentFlag
    


def calc_lnd0aa(sumf_MDD,diffti,geometry,extra_steps):

    if len(sumf_MDD.size())>1: #if there are multiple entries
        diffti = diffti.unsqueeze(1).repeat(1,sumf_MDD.size()[1])
    if extra_steps == True:
        diffti = diffti[2:]

    if geometry == "spherical":
        Daa_MDD_a = torch.zeros(sumf_MDD.shape)
        Daa_MDD_b = torch.zeros(sumf_MDD.shape)
        Daa_MDD_c = torch.zeros(sumf_MDD.shape)
        Daa_MDD_a[0] = ( (sumf_MDD[0]**2 - 0.**2 )*torch.pi/(36*(diffti[0])))


        # Equation 5a for all other steps

        diffFi = torch.zeros(sumf_MDD.shape)
        diffFi[0] = sumf_MDD[0]
        diffFi[1:] = sumf_MDD[1:]-sumf_MDD[0:-1]

        Daa_MDD_a[1:] = ((sumf_MDD[1:])**2 - (sumf_MDD[0:-1])**2 )*math.pi/(36*(diffti[1:]))

        # Fechtig and Kalbitzer Equation 5b, for cumulative gas fractions between 10 and 90%
        Daa_MDD_b[0] = (1/((torch.pi**2)*diffti[0]))*((2*torch.pi)-((math.pi*math.pi/3)*sumf_MDD[0])\
                                            - (2*math.pi)*(torch.sqrt(1-(math.pi/3)*sumf_MDD[0])))

        Daa_MDD_b[1:] = (1/((math.pi**2)*diffti[1:]))*(-(math.pi*math.pi/3)*diffFi[1:] \
                                            - (2*math.pi)*( torch.sqrt(1-(math.pi/3)*sumf_MDD[1:]) \
                                                - torch.sqrt(1 - (math.pi/3)*sumf_MDD[0:-1]) ))

        # Fechtig and Kalbitzer Equation 5c, for cumulative gas fractions greater than 90%
        Daa_MDD_c[1:] = (1/(math.pi*math.pi*diffti[1:]))*(torch.log((1-sumf_MDD[0:-1])/(1-sumf_MDD[1:])))

        # Decide which equation to use based on the cumulative gas fractions from each step
        use_a = (sumf_MDD<= 0.1) & (sumf_MDD> 0.00000001)
        use_b = (sumf_MDD > 0.1) & (sumf_MDD<= 0.9)
        use_c = (sumf_MDD > 0.9) & (sumf_MDD<= 1.0)
        Daa_MDD = use_a*Daa_MDD_a + torch.nan_to_num(use_b*Daa_MDD_b) + use_c*Daa_MDD_c

    elif geometry == "plane sheet":
        
        DR2_a = torch.zeros(sumf_MDD.shape)
        DR2_b = torch.zeros(sumf_MDD.shape)


        #Fechtig and Kalbitzer Equation 5a

        DR2_a[0] = ((((sumf_MDD[0]**2) - 0**2))*math.pi)/(4*diffti[0])
        DR2_a[1:] = ((((sumf_MDD[1:]**2)-(sumf_MDD[0:-1])**2))*math.pi)/(4*diffti[1:])
        DR2_b[1:] = (4/((math.pi**2)*diffti[1:]))*np.log((1-sumf_MDD[0:-1])/(1-sumf_MDD[1:]))
        usea = (sumf_MDD > 0) & (sumf_MDD < 0.6)
        useb = (sumf_MDD >= 0.6) & (sumf_MDD <= 1)

        Daa_MDD = usea*DR2_a + useb*DR2_b
        
    lnd0aa_MDD = torch.log(Daa_MDD)

    # if torch.sum(torch.isnan(sumf_MDD))>0:
    #     breakpoint()

    return lnd0aa_MDD


