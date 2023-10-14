import torch as torch
import numpy as np

def conHe_Param(X:torch.Tensor): # Input is Ea, lnd0aa_x, Frac_x-1
    X = X[1:]
    # CONSTRAING 1: FRACTIONS MUST ADD TO 1
    # Determine number of domains
    if len(X) <= 3:
            ndom = 1
    else:
        ndom = (len(X))//2
    temp = X[1:]
    fracstemp = temp[ndom:] #fracstemp has numDomain-1 entries 1-sum(Fracstemp)
    frac_id = 1-sum(fracstemp)

    # CONSTRAINT 2: lnD0aa1 > lnD0aa2.... > lnD0aa_n
    
    lnD0aa = temp[0:ndom] # Has shape (1,num_domains)
    

    lnD0aa_diff = lnD0aa[:-1] - lnD0aa[1:]
    lnD0aa_check = np.sum(np.where(lnD0aa_diff <= 0, lnD0aa_diff, np.array(0.0)),axis=0)

    return np.array([frac_id,lnD0aa_check])

    #constraint 1 must be >0
    #Constraint 2 must be between negative infinity and zero
    
#1.13402072e+01 1.26753159e+01 1.56747530e+01 1.65452306e+01 2.11751999e+01 2.42503575e+01 
# 
# 6.83008279e-02
#9.64097846e-03 2.72293434e-01 3.69831141e-01 2.78933714e-01]