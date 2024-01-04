import math
import numpy as np
import pandas as pd
import math as math
import warnings


def D0calc_MonteCarloErrors(expdata,geometry:str):
    # Function for calculating D0 and D/a^2 from experimental data. Input should be a
    # Pandas DataFrame with columns "TC", "thr",
    # M, and, and delM, which correspond to heating temperature (deg C), 
    # heating step duration (time in hours),
    # M (measured concentration in cps, atoms, or moles), delM (same units)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
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

            
            
            #DR2_a[0] = ( (Fi[0]**2 - 0.**2 )*math.pi/(36*(cumtsec[0])))

            # Equation 5a for all other steps
            # Updated textbook equations from reiners page 90
            #Fi is cumulative F at each step i, cumtsec is cumulative t in seconds, diffFi is differential Fi at each step, so length is len(Fi)-1, diffti is analagously length n-1
            DR2_a[0] = 1/((math.pi**2)*cumtsec[0])*(2*math.pi - (math.pi**2/3)*Fi[0] - 2*math.pi*np.sqrt(1-(math.pi/3)*Fi[0]))
            DR2_a[1:] = (1/(math.pi**2*diffti)) * (-math.pi**2/3 * diffFi - 2*math.pi*( np.sqrt(1-(math.pi/3)*Fi[1:]) - np.sqrt(1-(math.pi/3)*Fi[0:-1])))
            DR2_b[0] = -1/(math.pi**2* cumtsec[0]) * np.log((1-Fi[0]) * (math.pi**2/6))
            DR2_b[1:] = -1/(math.pi**2*diffti) * np.log( (1-Fi[1:])/(1-Fi[0:-1]))


            # Decide which equation to use based on the cumulative gas fractions from each step
            use_a = (Fi<= 0.85) & (Fi> 0.00000001)
            use_b = Fi > 0.85
            
            # Compute the final values
            DR2 = np.nan_to_num(use_a*DR2_a) + use_b*DR2_b
            uncert_a = np.zeros(len(DR2))
            uncert_b = np.zeros(len(DR2))
            for i in range(1,len(M)):
                
                # Equation XXXX from Ginster (2018)
                uncert_a[i] = 1/(3*(cumtsec[i]-cumtsec[i-1])*np.sum(M))*np.sqrt(
                        ((1-Fi[i])*((1/(np.sqrt(1-(math.pi/3)*Fi[i])))-1) + (1-Fi[i-1])*(1-(1/(np.sqrt(1-(math.pi/3)*Fi[i-1])))))**2*np.sum(delM[0:-1]**2) + 
                        ((1-Fi[i])*(1/np.sqrt(1-(math.pi/3)*Fi[i])-1) - Fi[i-1]*(1 - 1/np.sqrt(1-(math.pi/3)*Fi[i-1])))**2 * delM[i]**2 + 
                        (Fi[i]* (1- (1/np.sqrt(1-(math.pi/3)*Fi[i]))) - Fi[i-1]*(1 - 1/(np.sqrt(1-(math.pi/3)*Fi[i-1])))) **2 * np.sum(delM[i+1:]**2)
                )

                # Equation XXXX from Ginster (2018)
                uncert_b[i] = (1/(math.pi**2*(cumtsec[i]-cumtsec[i-1])*np.sum(M))) * np.sqrt((1+(Fi[i-1]/(1-Fi[i-1])))**2 * 
                            delM[i]**2 + (Fi[i-1]/(1-Fi[i-1]) - Fi[i]/(1-Fi[i]))**2 * np.sum(delM[i+1:]**2))
                
            # Combine the equations based on which gas fractions they're intended to be used for
            DR2_uncert = np.nan_to_num(uncert_a*use_a) + uncert_b*use_b
            
            # Special equations for the first step (Ginster, 2018)
            if Fi[0] <= 0.85:          
                DR2_uncert[0] = (1/(3*cumtsec[0]*np.sum(M)))* ((1/(np.sqrt(1-(math.pi/3)*Fi[0])))-1) * np.sqrt(((1-Fi[0])*delM[0])**2 + Fi[0]**2 * np.sum(delM[1:]**2))
            
            # If first step is < 0.85 (Ginster, 2018) 
            else:
                DR2_uncert[0] = (1/(math.pi**2*cumtsec[0]*np.sum(M))) * np.sqrt(delM[0]**2 + (Fi[0]/(1-Fi[0]))**2 * np.sum(delM[1:]**2))

                
        elif geometry == "plane sheet":
            # Initialize D/r^2
            DR2_a = np.zeros([nstep])
            DR2_b = np.zeros([nstep])

            
            #Fechtig and Kalbitzer Equation 5a
            DR2_a[0] = ((((Fi[0]**2) - 0**2))*math.pi)/(4*tsec[0])
            DR2_a[1:] = ((((Fi[1:]**2)-(Fi[0:-1])**2))*math.pi)/(4*tsec[1:])
            DR2_b[1:] = (4/((math.pi**2)*tsec[1:]))*np.log((1-Fi[0:-1])/(1-Fi[1:]))

            # Determine where gas fractions are specific value ranges
            usea = (Fi > 0) & (Fi < 0.6)
            useb = (Fi >= 0.6) & (Fi <= 1)

            # Calculate final vector of D/r2, using the correct equations where appropriate
            DR2 = usea*DR2_a + useb*DR2_b

            # Initialize uncertainties
    
            uncert_b = np.zeros(len(DR2))
            uncert_c = np.zeros(len(DR2))
    
            ## Compute uncertainties using the equations of Ginster 2018       
            for i in range(1,len(M)):
                uncert_b[i] =(math.pi/(2*(cumtsec[i]-cumtsec[i-1])*np.sum(M)) * np.sqrt(
                    ((Fi[i]*(1-Fi[i])) - Fi[i-1]*(1-Fi[i-1]))**2 * np.sum(delM[0:i]**2) + 
                    (Fi[i]*(1-Fi[i]) + Fi[i-1]**2)**2 * delM[i]**2 + 
                    (Fi[i-1]**2-Fi[i]**2)**2 * np.sum(delM[i+1:]**2)))

                # Equation XXX from Ginster 2018
                uncert_c[i] = (4/(math.pi**2*(cumtsec[i]-cumtsec[i-1])*np.sum(M))) * np.sqrt((1+Fi[i-1]/(1-Fi[i-1]))**2 * delM[i]**2 + ((Fi[i-1]/(1-Fi[i-1])) - (Fi[i]/(1-Fi[i])))**2 * np.sum(delM[i+1:]**2))

                

            # Put the equations together as necessary
            DR2_uncert = np.nan_to_num(usea*uncert_b)+ np.nan_to_num(useb*uncert_c)
            

            if Fi[0] < 0.6: # If the first release is < 60% of total gas
                DR2_uncert[0] = math.pi/(2*diffti[0]) *(diffFi[0]/np.sum(np.sum(M)))*np.sqrt(((1-diffFi[0])*delM[0])**2 + diffFi[0]**2 * np.sum(delM[1:]**2))
            
            else: # If first release is > 60% of gas
                DR2_uncert[0] = (4/(math.pi^2*diffti[0]*np.sum(M)))*np.sqrt(delM[0]**2+ (diffFi[0]/(1-diffFi[0]))**2 * (np.sum(delM[2:]**2)))
            

        return pd.DataFrame({"Tplot": Tplot,"Fi": Fi.ravel(),"Daa": DR2,"Daa uncertainty": DR2_uncert.ravel(), \
                                "ln(D/a^2)": np.log(DR2),"ln(D/a^2)-del": DR2_uncert.ravel()/DR2, \
                                "ln(D/a^2)+del": DR2_uncert.ravel()/DR2 })