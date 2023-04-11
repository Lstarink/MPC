import numpy as np


def Saturate(U, lower, upper):
    U_saturated = np.zeros(U.shape)

    for n, un in enumerate(U):
        if un < lower[n]:
            un = lower[n]
        elif un > upper:
            un = upper
        U_saturated[n] = un

    return U_saturated
