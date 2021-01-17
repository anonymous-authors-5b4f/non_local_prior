import pytest
from non_local_prior import bayes_factor as bf


class TestNMP1Null:
    def test_is_inverse_of_null_NMP1(self):
        delta = 0.015
        mu_0 = 0
        V = 0.005
        N_E = 18283

        # We are good if the inverse value is up to machine precision
        assert (
            bf.NMP1_null(delta=delta, mu_0=mu_0, V=V, N_E=N_E) ==
            pytest.approx(
                1.0 / bf.null_NMP1(delta=delta, mu_0=mu_0, V=V, N_E=N_E))
        )


class TestNormalNull:
    def test_is_inverse_of_null_normal(self):
        delta = 1.1248
        mu_0 = -0.248
        V = 0.123
        N_E = 12124

        # We are good if the inverse value is up to machine precision
        assert (
            bf.normal_null(delta=delta, mu_0=mu_0, V=V, N_E=N_E) ==
            pytest.approx(
                1.0 / bf.null_normal(delta=delta, mu_0=mu_0, V=V, N_E=N_E))
        )
