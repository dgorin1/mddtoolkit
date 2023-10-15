def generate_bounds(
    ndom: int,
    moles_bound,
    mineral_name: str,
    stat="chisq",
    Ea_bounds: tuple = (1, 500),
    lnd0aa_bounds: tuple = (-10, 50),
):
    """
    Generate bounds for the optimization problem. The bounds are based on the number
    of domains, the type of optimization, and the mineral name.

    Args:
        ndom (int): the number of domains in the optimization problem.
        moles_bound (tuple): the bounds for the moles of gas released.
        mineral_name (str): the name of the mineral.
        stat (str, optional): the type of optimization. Defaults to "chisq".
        Ea_bounds (tuple, optional): the bounds for the activation energy. Defaults to (1,500).
        lnd0aa_bounds (tuple, optional): the bounds for the ln(D0/a^2). Defaults to (-10,50).

    Returns:
        list: a list of bounds for the optimization problem.
    """
    if (
        stat.lower() == "chisq"
        or stat.lower() == "l2_moles"
        or stat.lower() == "l1_moles"
    ):
        moles = True
    else:
        moles = False

    frac_bounds = (0, 1)

    if ndom == 1:
        if moles == True:
            return [moles_bound, Ea_bounds, lnd0aa_bounds]
        else:
            return [Ea_bounds, lnd0aa_bounds]
    elif ndom > 1:
        if moles == True:
            return (
                [moles_bound, Ea_bounds]
                + ndom * [lnd0aa_bounds]
                + (ndom - 1) * [frac_bounds]
            )
        else:
            return [Ea_bounds] + ndom * [lnd0aa_bounds] + (ndom - 1) * [frac_bounds]
