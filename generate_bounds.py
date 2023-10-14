def generate_bounds(ndom:int, moles_bound, mineral_name:str,stat = "chisq"):


    if mineral_name.lower() == "quartz":
        if stat.lower() == "chisq" or stat.lower() == "l2_moles" or stat.lower() == "l1_moles":
            moles = True
        else:
            moles = False

        Ea_bounds = (50,250)
        lnd0aa_bounds = (-10,40)
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
    
    elif mineral_name.lower() == "kspar":
        if stat.lower() == "chisq" or stat.lower() == "l2_moles" or stat.lower() == "l1_moles":
            moles = True
        else:
            moles = False

        Ea_bounds = (50,400)
        lnd0aa_bounds = (-5,50)
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
            
        
    elif mineral_name.lower() == "pyroxene":
        if stat.lower() == "chisq" or stat.lower() == "l2_moles" or stat.lower() == "l1_moles":
            moles = True
        else:
            moles = False

        Ea_bounds = (1,500)
        lnd0aa_bounds = (-50,50)
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
    
    elif mineral_name.lower() == "plag":
        if stat.lower() == "chisq" or stat.lower() == "l2_moles" or stat.lower() == "l1_moles":
            moles = True
        else:
            moles = False

        Ea_bounds = (1,500)
        lnd0aa_bounds = (-50,50)
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



