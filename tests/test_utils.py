import numpy as np
import pytest

from crystalytics_py.utils import (
    shortest_collinear_vector_with_integer_components,
)


def test_integer_vector_is_reduced_to_coprime_components():

    vector, deviation = (
        shortest_collinear_vector_with_integer_components(
            np.array([2, 4, 2], dtype=int)
        )
    )

    np.testing.assert_array_equal(
        vector[0],
        np.array([1, 2, 1]),
    )

    assert deviation[0] == pytest.approx(0.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known Stage-1 bug: floating-vector "
        "cosine-deviation normalization is incorrect."
    ),
)
def test_float_vector_exactly_collinear_has_zero_deviation():

    vector, deviation = (
        shortest_collinear_vector_with_integer_components(
            np.array(
                [0.5, 1.0, 0.5],
                dtype=float,
            )
        )
    )

    np.testing.assert_array_equal(
        vector[0],
        np.array([1, 2, 1]),
    )

    assert deviation[0] == pytest.approx(
        0.0,
        abs=1e-12,
    )