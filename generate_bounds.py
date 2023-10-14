def generate_bounds(ndom:int, moles_bound, mineral_name:str,stat = "chisq",Ea_bounds:tuple =  (1,500), lnd0aa_bounds:tuple = (-10,50)):
        if stat.lower() == "chisq" or stat.lower() == "l2_moles" or stat.lower() == "l1_moles":
            moles = True
        else:
            moles = False

        
        frac_bounds = (0,1)

        if ndom == 1:
            if moles == True:
                return [moles_bound,Ea_bounds,lnd0aa_bounds]
            else:
                return [Ea_bounds,lnd0aa_bounds]
        elif ndom >1:
            if moles == True:
                return [moles_bound,Ea_bounds]+ ndom*[lnd0aa_bounds]+ (ndom-1)*[frac_bounds]
            else:
                return [Ea_bounds]+ ndom*[lnd0aa_bounds]+ (ndom-1)*[frac_bounds]
    
    



