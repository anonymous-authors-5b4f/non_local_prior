import numpy as np
from scipy.special import gamma


def pdf(theta: float, theta_0: float, nu: float, tau: float, k: int):

    assert k >= 1, "Required parameters: k >= 1"

    # Guard against zero division error in the negative power term
    # if theta == theta_0
    if theta == theta_0:
        return 0
    dist = (theta - theta_0) ** 2

    return(
        k * tau ** (nu / 2.0) /
        gamma(nu / (2 * k)) *
        np.power(dist, -(nu+1)/2) *
        np.exp(-1.0 * (dist / tau) ** (-k))
    )
