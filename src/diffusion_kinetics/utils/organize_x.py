import numpy as np


def organize_x(x, chop_fracs=True):
    """Organize the parameters from the optimization problem into a single array and ensure that the 
    domains are listed in order from least to most retentive.

    Args:
        x (torch.Tensor): the parameters from the optimization problem.
        chop_fracs (bool, optional): whether or not to chop the last fraction. Defaults to True.

    Returns:
        np.array: the parameters from the optimization problem.
    """
    x1 = x.copy()
    if len(x1) % 2 != 0:
        moles = x1[0]
        x1 = x1[1:]
    else:
        moles = np.NaN

    ndom =  (len(x1))//2
    Ea = x1[0]
    lnd0aa = x1[1 : 1 + ndom]
    fracs = x1[1 + ndom :]
    fracs = np.append(fracs, 1 - np.sum(fracs))

    n = len(fracs)
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Traverse the array from 0 to n-i-1
            # Swap if the element found is greater than the next element
            if lnd0aa[j] < lnd0aa[j + 1]:
                lnd0aa[j], lnd0aa[j + 1] = lnd0aa[j + 1], lnd0aa[j]
                fracs[j], fracs[j + 1] = fracs[j + 1], fracs[j]

    if ~np.isnan(moles):
        output = np.append(moles, Ea)
    else:
        output = Ea
    output = np.append(output, lnd0aa)
    if chop_fracs == True:
        output = np.append(output, fracs[0:-1])
    else:
        output = np.append(output, fracs)

    return output