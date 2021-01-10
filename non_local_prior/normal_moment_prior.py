from scipy.special import factorial2


def normal_central_moment(n: int, sigma: float):
    """
    Return the nth central moment of a normal distribution
    https://math.stackexchange.com/questions/92648/calculation-of-the-n-th-central-moment-of-the-normal-distribution-mathcaln
    """
    if n % 2 == 1:
        return 0
    else:
        return factorial2(n-1, exact=True) * sigma ** n


class NMP:
    def pdf(self):
        pass
