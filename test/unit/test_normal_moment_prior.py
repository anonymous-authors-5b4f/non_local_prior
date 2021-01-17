import pytest
import non_local_prior.normal_moment_prior as nmp


class TestNormalCentralMoment:
    def test_odd_central_moment_equal_zero(self):
        """
        All odd central moments of a normal distribution, regardless of its
        parameters, are equal to zero
        :return:
        """
        n = 1
        sigma = 1
        assert nmp._normal_central_moment(n=n, sigma=sigma) == 0.0

        n = 1
        sigma = 0.16728
        assert nmp._normal_central_moment(n=n, sigma=sigma) == 0.0

        n = 3
        sigma = 1
        assert nmp._normal_central_moment(n=n, sigma=sigma) == 0.0

        n = 3
        sigma = 0.419819
        assert nmp._normal_central_moment(n=n, sigma=sigma) == 0.0

        n = 5
        sigma = 1
        assert nmp._normal_central_moment(n=n, sigma=sigma) == 0.0

        n = 5
        sigma = 0.912182
        assert nmp._normal_central_moment(n=n, sigma=sigma) == 0.0

    def test_second_central_moment_is_sigma_squared_1(self):
        # 2nd central moment of normal is sigma^2
        n = 2
        sigma = 1
        assert nmp._normal_central_moment(n=n, sigma=sigma) == sigma ** 2

    def test_second_central_moment_is_sigma_squared_2(self):
        n = 2
        sigma = 0.35815
        assert nmp._normal_central_moment(n=n, sigma=sigma) == sigma ** 2

    def test_fourth_central_moment_1(self):
        # 4th central moment of normal is 3*sigma^4
        n = 4
        sigma = 1
        assert nmp._normal_central_moment(n=n, sigma=sigma) == 3 * sigma ** 4

    def test_fourth_central_moment_2(self):
        n = 4
        sigma = 0.35815
        assert nmp._normal_central_moment(n=n, sigma=sigma) == 3 * sigma ** 4

    def test_sixth_central_moment_1(self):
        # 6th central moment of normal is 15*sigma^6
        n = 6
        sigma = 1
        assert nmp._normal_central_moment(n=n, sigma=sigma) == 15 * sigma ** 6

    def test_sixth_central_moment_2(self):
        n = 6
        sigma = 0.332281
        assert nmp._normal_central_moment(n=n, sigma=sigma) == 15 * sigma ** 6


class TestPdf:
    def test_pdf_is_zero_when_theta_equals_theta_0(self):
        theta = 0.0
        theta_0 = 0.0
        sigma = 1.0
        k = 1

        assert nmp.pdf(theta, theta_0, sigma, k) == 0

    def test_error_when_sigma_is_non_positive(self):
        theta = 0.0
        theta_0 = 0.0
        sigma = -0.5
        k = 1

        with pytest.raises(AssertionError):
            nmp.pdf(theta, theta_0, sigma, k)

    def test_error_when_k_is_non_positive(self):
        theta = 0.0
        theta_0 = 0.0
        sigma = 1.0
        k = 0

        with pytest.raises(AssertionError):
            nmp.pdf(theta, theta_0, sigma, k)
