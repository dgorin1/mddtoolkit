import numpy as np


def organize_x(x, chop_fracs=True):
    """Re-pack and sort the optimizer parameter vector into a canonical form.

    Domains are sorted from most to least retentive (highest lnD0/a² first),
    with their corresponding gas fractions kept in sync.

    Args:
        x (np.ndarray): Raw parameter vector from the optimizer:
            ``[total_moles (optional), Ea, lnD0/a²_1, ..., frac_1, ...]``.
        chop_fracs (bool): If ``True`` (default), drop the last fraction because
            it is implicit (``1 - sum(remaining fracs)``).

    Returns:
        np.ndarray: Re-packed parameter vector in canonical order.
    """
    params = x.copy()

    if len(params) % 2 != 0:
        moles = params[0]
        params = params[1:]
    else:
        moles = np.nan

    n_dom = len(params) // 2
    ea = params[0]
    ln_d0aa = params[1 : 1 + n_dom]
    fracs = params[1 + n_dom :]
    fracs = np.append(fracs, 1 - np.sum(fracs))

    # Sort domains from most to least retentive (highest lnD0/a² first),
    # keeping fracs synchronised with their corresponding domain.
    sort_idx = np.argsort(ln_d0aa, kind="stable")[::-1]
    ln_d0aa = ln_d0aa[sort_idx]
    fracs = fracs[sort_idx]

    if not np.isnan(moles):
        output = np.append(moles, ea)
    else:
        output = ea
    output = np.append(output, ln_d0aa)

    if chop_fracs:
        output = np.append(output, fracs[:-1])
    else:
        output = np.append(output, fracs)

    return output
