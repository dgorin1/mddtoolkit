import math
import numpy as np
import pandas as pd
import math as math


def D0calc_MonteCarloErrors(expdata,geometry:str = "spherical"):
    # Function for calculating D0 and D/a^2 from experimental data. Input should be a
    # Pandas DataFrame with columns "TC", "thr",
    # M, and, and delM, which correspond to heating temperature (deg C), 
    # heating step duration (time in hours),
    # M (measured concentration in cps, atoms, or moles), delM (same units)
    
    # Calculate diffusivities from the previous experiment
    TC = expdata.loc[:,"TC"].array
    thr = expdata.loc[:,"thr"].array
    M = expdata.loc[:,"M"].array
    delM = expdata.loc[:,"delM"].array

    #Check if units are in minutes and convert from hours to minutes if necesssary



    #Convert units
    TK = 273.15+TC
    tsec = thr*60*60
    Tplot = 1*10**4/TK
    nstep = len(M)
    cumtsec = np.cumsum(tsec)
    Si = np.cumsum(M)
    S = np.amax(Si)
    Fi = Si/S



    # initialize diffusivity vectors fore each Fechtig and Kalbitzer equation
    DR2_a = np.zeros([nstep])
    DR2_b = np.zeros([nstep])


    # Create the a list of times for each heating step
    diffti = cumtsec[1:]-cumtsec[0:-1]

    # Create a list of the gas fraction differences between steps
    diffFi = Fi[1:]-Fi[0:-1]



    #If the geometry is spherical...
    if geometry == "spherical":
        #If geometry is spherical, then a third DR2 vector is needed.
        DR2_c = np.zeros([nstep])
        # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
        # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
        # special case when i = 1; need to insert 0 for previous amount released

        DR2_a[0] = ( (Fi[0]**2 - 0.**2 )*math.pi/(36*(cumtsec[0])))

        # Equation 5a for all other steps

        DR2_a[1:] = ((Fi[1:])**2 - (Fi[0:-1])**2 )*math.pi/(36*(diffti))

        # Fechtig and Kalbitzer Equation 5b, for cumulative gas fractions between 10 and 90%

        DR2_b[0] = (1/((math.pi**2)*tsec[0]))*((2*math.pi)-((math.pi*math.pi/3)*Fi[0])\
                                            - (2*math.pi)*(np.sqrt(1-(math.pi/3)*Fi[0])))
        DR2_b[1:] = (1/((math.pi**2)*diffti))*(-(math.pi*math.pi/3)*diffFi \
                                            - (2*math.pi)*( np.sqrt(1-(math.pi/3)*Fi[1:]) \
                                                - np.sqrt(1 - (math.pi/3)*Fi[0:-1]) ))

        # Fechtig and Kalbitzer Equation 5c, for cumulative gas fractions greater than 90%
        DR2_c[1:] = (1/(math.pi*math.pi*diffti))*(np.log((1-Fi[0:-1])/(1-Fi[1:])))

        # Decide which equation to use based on the cumulative gas fractions from each step
        use_a = (Fi<= 0.1) & (Fi> 0.00000001)
        use_b = (Fi > 0.1) & (Fi<= 0.9)
        use_c = (Fi > 0.9) & (Fi<= 1.0)

        # Compute the final values
        DR2 = use_a*DR2_a + np.nan_to_num(use_b*DR2_b) + use_c*DR2_c

    elif geometry == "plane sheet":

        DR2_a = np.zeros([nstep])
        DR2_b = np.zeros([nstep])


        #Fechtig and Kalbitzer Equation 5a
        DR2_a[0] = ((((Fi[0]**2) - 0**2))*math.pi)/(4*tsec[0])
        DR2_a[1:] = ((((Fi[1:]**2)-(Fi[0:-1])**2))*math.pi)/(4*tsec[1:])
        DR2_b[1:] = (4/((math.pi**2)*tsec[1:]))*np.log((1-Fi[0:-1])/(1-Fi[1:]))
        usea = (Fi > 0) & (Fi < 0.6)
        useb = (Fi >= 0.6) & (Fi <= 1)

        DR2 = usea*DR2_a + useb*DR2_b

    # Compute uncertainties in diffusivity using a Monte Carlo simulation
    # Generates simulated step degassing datasets, such that each step of the 
    # experiment has a Gaussian distribution centered at M and with 1s.d. of
    # delM across the simulated datasets.Then recomputes diffusivities for each 
    # simulated dataset and uses the range of diffusivities for each step across
    # all simulated datasets to estimate uncertainty. 
    # make vector with correct diffusivites for each step

    n_sim = 30000 #number of simulations in the monte carlo
    MCsim = np.zeros([nstep,n_sim])#initialize matrix for simulated measurements


    
    
    for i in range(nstep):
        #Generate the simulated measurements
        MCsim[i,:] = np.random.randn(1,n_sim)*delM[i] + M[i]

    #compute cumulative gas release fraction for each simulation
    MCSi = np.cumsum(MCsim,0)
    MCS = np.amax(MCSi,0)
    MCFi = np.zeros([nstep,n_sim])
    delMCFi = np.zeros([nstep,1])
    MCFimean = np.zeros([nstep,1])



    for i in range(n_sim):
        MCFi[:,i] = MCSi[:,i]/np.amax(MCSi[:,i])
    for i in range(nstep):
        #delMCFi[i] = (np.amax(MCFi[i,:],0) - np.amin(MCFi[i,:],0))/2
        MCFimean[i] = np.mean(MCFi[i,:],0)
    
    #Initialize vectors
    MCDR2_a = np.zeros([nstep,n_sim])
    MCDR2_b = np.zeros([nstep,n_sim])
    MCdiffFi = np.zeros([nstep,n_sim])



    for m in range(1,nstep): #For step of each experiment...
        for n in range(n_sim):
            MCdiffFi[m,n] = MCFi[m,n] - MCFi[m-1,n] #calculate the fraction released at each step
            MCdiffFi[0,n] = MCFi[0,n]
    for m in range(0,nstep):
        delMCFi[m] = np.std(MCdiffFi[m,:])
    
    
    if geometry == "spherical":
        MCDR2_c = np.zeros([nstep,n_sim])
        for n in range(n_sim): #For each first step of an experiment, insert 0 for previous amount released
            MCDR2_a[0,n] = ((MCFi[m,n])**2 - (MCFi[m-1,n])**2 )*math.pi/(36*(diffti[m-1]))
        for m in range(1,nstep): #Calculate fechtig and kalbitzer equations for each fraction
            for n in range(n_sim):
                MCDR2_a[m,n] = ( (MCFi[m,n])**2 - (MCFi[m-1,n])**2 )*math.pi/(36*(diffti[m-1]));
                MCDR2_b[m,n] = (1/((math.pi**2)*diffti[m-1]))*( -(math.pi*math.pi/3)* MCdiffFi[m,n] \
                                - (2*math.pi)*( np.sqrt(1-(math.pi/3)*MCFi[m,n]) \
                                -np.sqrt(1 - (math.pi/3)*MCFi[m-1,n]) ))
                MCDR2_c[m,n] = (1/(math.pi*math.pi*diffti[m-1]))*(np.log((1-MCFi[m-1,n])/(1-MCFi[m,n])));
        MCdiffFiFinal = np.zeros([nstep])
        for m in range(0,nstep):
            MCdiffFiFinal[m] = np.mean(MCdiffFi[m,:])

        use_a_MC = (MCFi<= 0.1) & (MCFi> 0.00000001)
        use_b_MC = (MCFi > 0.1) & (MCFi<= 0.9)
        use_c_MC = (MCFi > 0.9) & (MCFi<= 1.0) 


        MCDR2 = use_a_MC*MCDR2_a + np.nan_to_num(use_b_MC*MCDR2_b) + use_c_MC*MCDR2_c

        MCDR2_uncert = np.zeros([nstep,1])
        for i in range(nstep):
            MCDR2_uncert[i,0] = np.std(MCDR2[i,:])

    elif geometry == "plane sheet":
        for i in range(n_sim):
            MCDR2_a[0,i] = ((MCFi[1,i])**2 - 0**2 )*math.pi/(4*(tsec[0]))
        for i in range(1,nstep):
            for j in range(n_sim):
                MCDR2_a[i,j] = ((MCFi[i,j])**2 - (MCFi[i-1,j])**2 )*math.pi/(4*(tsec[i-1]))
                MCDR2_b[i,j] = (4/((math.pi**2)*diffti[i-1]))*np.log((1-MCFi[i-1,j])/(1-MCFi[i,j]))
            usea_MC = (MCFi < 0.6) & (MCFi > 0)
            useb_MC = (MCFi >= 0.6) & (MCFi <= 1)
            MCDR2 = usea_MC*MCDR2_a + useb_MC * MCDR2_b


    MCDR2_uncert = np.zeros([nstep,1])
    MCDR24Uncert = MCDR2.copy()
    #For each row in the MCDR, which corresponds to every monte carlo step of the heating schedule..
    #Replace the Nan/inf/-inf values with the mean so that we can more easily calculate the standard deviation...
    #To estimate the uncertainty below. 
    for i in range(MCDR24Uncert.shape[0]-1):
        index = (MCDR24Uncert[i,:] != -np.inf) & (MCDR24Uncert[i,:] != np.inf)
        mean = np.mean(MCDR24Uncert[i,:][index])
        MCDR24Uncert[i,:][~index] = mean


    for i in range(nstep):
        MCDR2_uncert[i,0] = np.std(MCDR24Uncert[i,:])


    return pd.DataFrame({"Tplot": Tplot,"Fi": MCFimean.ravel(),"Fi uncertainty": \
                            delMCFi.ravel(), "Daa": DR2,"Daa uncertainty": MCDR2_uncert.ravel(), \
                            "ln(D/a^2)": np.log(DR2),"ln(D/a^2)-del": np.log(DR2-MCDR2_uncert.ravel()), \
                            "ln(D/a^2)+del": np.log(DR2+MCDR2_uncert.ravel()) })
