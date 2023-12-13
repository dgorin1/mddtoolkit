def generate_bounds(
    ndom: int,
    moles_bound1,
    stat="chisq",
    Ea_bounds: tuple = (1, 500),
    lnd0aa_bounds: tuple = (-10, 35),
):
    """
    Generate bounds for the optimization problem. The bounds are based on the number
    of domains, and the type of optimization.

    Args:
        ndom (int): the number of domains in the optimization problem.
        moles_bound (tuple): the bounds for the moles of gas released.
        stat (str, optional): the type of optimization. Defaults to "chisq".
        Ea_bounds (tuple, optional): the bounds for the activation energy in kJ/mol. Defaults to (1,500).
        lnd0aa_bounds (tuple, optional): the bounds for the ln(D0/a^2). Defaults to (-10,35).

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
            return [moles_bound1, Ea_bounds, lnd0aa_bounds]
        else:
            return [Ea_bounds, lnd0aa_bounds]
    elif ndom > 1:
        if moles == True:
            return (
                [moles_bound1, Ea_bounds]
                + ndom * [lnd0aa_bounds]
                + (ndom - 1) * [frac_bounds]
            )
        else:
            return [Ea_bounds] + ndom * [lnd0aa_bounds] + (ndom - 1) * [frac_bounds]
