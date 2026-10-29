import numpy as np
import pytest

from crystalytics_py.crystal_structure import CrystalStructure
from crystalytics_py.direction import Direction, LatticeDirection

def test_direction_alias_is_backward_compatible():
    assert Direction is LatticeDirection

def test_sc_111_is_lattice_and_plane_commensurate(
    sc,
    direction_factory,
):
    d = direction_factory(
        sc,
        [1, 1, 1],
    )

    assert d.is_lattice_direction[0]
    assert d.is_plane_commensurate[0]

    np.testing.assert_array_equal(
        d.normal_plane_indices[0],
        [1, 1, 1],
    )

def test_non_cubic_direction_maps_to_different_normal_plane_indices():

    crystal = CrystalStructure(
        np.diag(
            [
                1.0,
                np.sqrt(2.0),
                1.0,
            ]
        )
    )

    d = LatticeDirection(
        crystal,
        np.array(
            [[1, 1, 0]],
            dtype=int,
        ),
        basis_directions="primitive_vector",
    )

    d.compute("lattice_spacing")

    assert d.is_lattice_direction[0]
    assert d.is_plane_commensurate[0]

    np.testing.assert_array_equal(
        d.normal_plane_indices[0],
        [1, 2, 0],
    )

    d.compute("lattice_interplanar_spacing")

    assert d.multiplicity[0] == 3

    assert (
        d.lattice_interplanar_spacings[0]
        == pytest.approx(
            1.0 / np.sqrt(3.0)
        )
    )

def test_lattice_direction_can_lack_rational_normal_plane():

    crystal = CrystalStructure(
        np.diag(
            [
                1.0,
                2.0**0.25,
                1.0,
            ]
        )
    )

    d = LatticeDirection(
        crystal,
        np.array(
            [[1, 1, 0]],
            dtype=int,
        ),
        basis_directions="primitive_vector",
    )

    d.compute("lattice_spacing")

    assert d.is_lattice_direction[0]

    assert not d.is_plane_commensurate[0]

    with pytest.raises(
        RuntimeError,
        match="not normal to a rational",
    ):
        d.compute("lattice_interplanar_spacing")

    with pytest.raises(
        RuntimeError,
        match="plane-commensurate",
    ):
        _ = d.normal_plane_indices

def test_arbitrary_cartesian_direction_can_be_flagged_as_not_lattice_direction(
    sc,
):
    requested = np.array(
        [[1.0, np.sqrt(2.0), 0.0]]
    )

    d = LatticeDirection(
        sc,
        requested,
        basis_directions="global_orthonormal",
    )

    d.compute("lattice_spacing")

    assert not d.is_lattice_direction[0]

    assert (
        d.lattice_direction_cosine_deviations[0]
        > 0.0
    )

