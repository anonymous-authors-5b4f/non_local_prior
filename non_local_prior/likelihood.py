import numpy as np
from scipy.stats import norm


def null(delta: float, mu_0: float = 0, N_E: float = 100., **kwargs):
    """
    Likelihood function f(delta|mu) for the null hypothesis where mu=mu_0.
    Recap: delta~N(mu, 1/N_E), mu=mu_0
    """
    assert N_E > 0, "Parameter requirement: N_E > 0"

    return norm.pdf(delta, mu_0, np.sqrt(1/N_E))


def alt_normal(delta: float, mu_0: float = 0, V: float = 1.,
               N_E: float = 100., **kwargs):
    """
    Likelihood function f(delta|mu) where mu is the normal (local) prior.
    Recap: delta~N(mu,1/N_E), mu~N(mu_0, V^2)
    """
    assert N_E > 0, "Parameter requirement: N_E > 0"

    return norm.pdf(delta, mu_0, np.sqrt(V**2 + 1/N_E))


def alt_normal_moment1(delta: float, mu_0: float = 0, V: float = 1.,
                       N_E: float = 100., **kwargs):
    """
    Likelihood function f(delta|mu) where mu follows the normal moment prior
    with k=1.
    Recap: delta~N(mu,1/N_E), mu~NMP(theta_0=mu_0, sigma^2=V^2, k=1)
    """
    assert N_E > 0, "Parameter requirement: N_E > 0"

    delta_var = V**2 + 1/N_E
    return(
        norm.pdf(delta, mu_0, np.sqrt(delta_var)) / delta_var *
        ((delta - mu_0) ** 2 * V ** 2 / delta_var + 1 / N_E)
    )
