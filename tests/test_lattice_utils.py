import numpy as np

from crystalytics_py.lattice_utils import (
    gauss_reduce,
    reduce_and_transform_lattice_vecs,
)


def test_gauss_reduction_preserves_lattice():

    basis = np.array(
        [
            [2.0, 0.0],
            [1.0, 1.0],
        ]
    )

    reduced, transform = gauss_reduce(basis)

    np.testing.assert_allclose(
        reduced,
        transform @ basis,
    )

    assert abs(
        round(np.linalg.det(transform))
    ) == 1


def test_reduce_and_transform_2d_lattice_embedded_in_3d():

    basis = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )

    reduced, intrinsic, Q, transform = (
        reduce_and_transform_lattice_vecs(basis)
    )

    np.testing.assert_allclose(
        reduced,
        transform @ basis,
    )

    np.testing.assert_allclose(
        Q.T @ Q,
        np.eye(3),
        atol=1e-12,
    )

    assert intrinsic.shape == (2, 2)

    assert abs(
        round(np.linalg.det(transform))
    ) == 1