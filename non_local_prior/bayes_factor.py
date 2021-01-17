from non_local_prior import likelihood


# A Bayes factor of B_{xy} (implemented in code as x_y) denotes the evidence
# in favour of model/hypothesis x on top of competing model/hypothesis y

def null_normal(delta: float, mu_0: float, V: float, N_E: float) -> float:
    """
    Computing the Bayes factor in favour of null B_{0 1L}, where
    0: Null hypothesis - mu = mu_0
    1L: Alternate hypothesis with local normal prior - mu ~ N(mu_0, V^2)
    """
    return(
        likelihood.null(delta=delta, mu_0=mu_0, N_E=N_E) /
        likelihood.alt_normal(delta=delta, mu_0=mu_0, V=V, N_E=N_E)
    )


def null_NMP1(delta: float, mu_0: float, V: float, N_E: float) -> float:
    """
    Computing the Bayes factor in favour of null B_{0 1M}, where
    0: Null hypothesis - mu = mu_0
    1M: Alternate hypothesis with normal moment prior with k=1 - 
        mu ~ NMP(theta_0=mu_0, sigma^2=V^2, k=1)
    """
    return(
        likelihood.null(delta=delta, mu_0=mu_0, N_E=N_E) /
        likelihood.alt_normal_moment1(delta=delta, mu_0=mu_0, V=V, N_E=N_E)
    )


def NMP1_null(delta: float, mu_0: float, V: float, N_E: float) -> float:
    """
    Computing the Bayes factor in favour of alternate (NMP1) B_{1M 0}, where
    1M: Alternate hypothesis with normal moment prior with k=1 - 
        mu ~ NMP(theta_0=mu_0, sigma^2=V^2, k=1)
    0: Null hypothesis - mu = mu_0
    """
    return(
        likelihood.alt_normal_moment1(delta=delta, mu_0=mu_0, V=V, N_E=N_E) /
        likelihood.null(delta=delta, mu_0=mu_0, N_E=N_E)
    )


def normal_null(delta: float, mu_0: float, V: float, N_E: float) -> float:
    """
    Computing the Bayes factor in favour of alternate (local normal) B_{1L 0}, 
    where
    1L: Alternate hypothesis with local normal prior - mu ~ N(mu_0, V^2)
    0: Null hypothesis - mu = mu_0
    """
    return(
        likelihood.alt_normal(delta=delta, mu_0=mu_0, V=V, N_E=N_E) /
        likelihood.null(delta=delta, mu_0=mu_0, N_E=N_E)
    )
