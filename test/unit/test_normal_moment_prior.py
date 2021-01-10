# import pytest
from non_local_prior.normal_moment_prior import normal_central_moment


class TestNormalCentralMoment:
    def test_odd_central_moment_equal_zero(self):
        """
        All odd central moments of a normal distribution, regardless of its
        parameters, are equal to zero
        :return:
        """
        n = 1
        sigma = 1
        assert normal_central_moment(n=n, sigma=sigma) == 0.0

        n = 1
        sigma = 0.16728
        assert normal_central_moment(n=n, sigma=sigma) == 0.0

        n = 3
        sigma = 1
        assert normal_central_moment(n=n, sigma=sigma) == 0.0

        n = 3
        sigma = 0.419819
        assert normal_central_moment(n=n, sigma=sigma) == 0.0

        n = 5
        sigma = 1
        assert normal_central_moment(n=n, sigma=sigma) == 0.0

        n = 5
        sigma = 0.912182
        assert normal_central_moment(n=n, sigma=sigma) == 0.0

    def test_second_central_moment_is_sigma_squared(self):
        # 2nd central moment of normal is sigma^2
        n = 2
        sigma = 1
        assert normal_central_moment(n=n, sigma=sigma) == sigma ** 2

        n = 2
        sigma = 0.35815
        assert normal_central_moment(n=n, sigma=sigma) == sigma ** 2

    def test_fourth_central_moment(self):
        # 4th central moment of normal is 3*sigma^4
        n = 4
        sigma = 1
        assert (normal_central_moment(n=n, sigma=sigma) == 3 * sigma ** 4)

        n = 4
        sigma = 0.35815
        assert (normal_central_moment(n=n, sigma=sigma) == 3 * sigma ** 4)

    def test_sixth_central_moment(self):
        # 6th central moment of normal is 15*sigma^6
        n = 6
        sigma = 1
        assert (normal_central_moment(n=n, sigma=sigma) == 15 * sigma ** 6)

        n = 6
        sigma = 0.332281
        assert (normal_central_moment(n=n, sigma=sigma) == 15 * sigma ** 6)

