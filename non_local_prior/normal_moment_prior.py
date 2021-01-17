from scipy.stats import norm
from scipy.special import factorial2


def _normal_central_moment(n: int, sigma: float) -> float:
    """
    Return the nth central moment of a normal distribution
    https://math.stackexchange.com/questions/92648/calculation-of-the-n-th-central-moment-of-the-normal-distribution-mathcaln
    """
    if n % 2 == 1:
        return 0.0
    else:
        return factorial2(n-1, exact=True) * sigma ** n


def pdf(theta: float, theta_0: float, sigma: float, k: int) -> float:
    assert sigma > 0.0 and k >= 1, \
        "Required parameters: sigma > 0 and k >= 1"

    return(
        (theta - theta_0) ** (2 * k) /
        _normal_central_moment(n=(2*k), sigma=sigma) *
        norm.pdf(theta, theta_0, sigma)
    )
