import numpy as np
import pytest

from crystalytics_py.direction import (
    LatticeDirection,
)


def test_compute_lattice_spacing_populates_requested_result(
    sc,
):
    """
    A direct lattice-spacing request should populate its public outputs
    without adding unnecessary computations to the pipeline.
    """

    direction = LatticeDirection(
        sc,
        np.array(
            [[1, 1, 0]],
            dtype=int,
        ),
        basis_directions="primitive_vector",
    )

    direction.compute(
        "lattice_spacing"
    )

    assert direction._compute_pipeline == [
        "lattice_spacing"
    ]

    assert (
        direction.lattice_spacings[0]
        == pytest.approx(
            np.sqrt(2.0)
        )
    )

    np.testing.assert_array_equal(
        direction.shortest_lattice_vectors[
            0
        ],
        [1, 1, 0],
    )


def test_compute_interplanar_spacing_inserts_dependencies(
    sc,
):
    """
    Requesting interplanar spacing should automatically insert lattice
    spacing and plane-commensurability calculations beforehand.
    """

    direction = LatticeDirection(
        sc,
        np.array(
            [[1, 1, 1]],
            dtype=int,
        ),
        basis_directions="primitive_vector",
    )

    direction.compute(
        "lattice_interplanar_spacing"
    )

    assert direction._compute_pipeline == [
        "lattice_spacing",
        "plane_commensurability",
        "lattice_interplanar_spacing",
    ]

    assert (
        direction.is_plane_commensurate[0]
    )

    assert (
        direction.multiplicity[0]
        == 3
    )

    assert (
        direction
        .lattice_interplanar_spacings[0]
        == pytest.approx(
            1.0 / np.sqrt(3.0)
        )
    )


def test_compute_motif_plane_locations_inserts_full_dependency_chain(
    sc,
):
    """
    Requesting motif-plane locations should automatically compute every
    crystallographic prerequisite needed by that quantity.
    """

    direction = LatticeDirection(
        sc,
        np.array(
            [[1, 0, 0]],
            dtype=int,
        ),
        basis_directions="primitive_vector",
    )

    direction.compute(
        "motif_plane_locations"
    )

    assert direction._compute_pipeline == [
        "lattice_spacing",
        "plane_commensurability",
        "lattice_interplanar_spacing",
        "motif_plane_locations",
    ]

    np.testing.assert_allclose(
        direction.motif_plane_locations[
            0
        ],
        [0.0],
    )


def test_compute_multiple_requests_deduplicates_shared_dependencies(
    sc,
):
    """
    Multiple requested outputs should share prerequisites rather than
    placing the same computation in the pipeline more than once.
    """

    direction = LatticeDirection(
        sc,
        np.array(
            [[1, 0, 0]],
            dtype=int,
        ),
        basis_directions="primitive_vector",
    )

    direction.compute(
        [
            "motif_plane_locations",
            "lattice_interplanar_spacing",
        ]
    )

    assert direction._compute_pipeline == [
        "lattice_spacing",
        "plane_commensurability",
        "lattice_interplanar_spacing",
        "motif_plane_locations",
    ]


def test_compute_all_populates_every_registered_computation(
    sc,
):
    """
    The 'all' selector should execute every computation currently
    registered in the public computation pipeline.
    """

    direction = LatticeDirection(
        sc,
        np.array(
            [[1, 0, 0]],
            dtype=int,
        ),
        basis_directions="primitive_vector",
    )

    direction.compute("all")

    assert direction._compute_pipeline == [
        "lattice_spacing",
        "plane_commensurability",
        "lattice_interplanar_spacing",
        "motif_plane_locations",
        "relative_shift_of_lattice_planes",
    ]

    assert hasattr(
        direction,
        "_shortest_relative_shift_vectors",
    )


def test_compute_rejects_unknown_computation(
    sc,
):
    """
    An unknown computation name should fail before any scientific
    calculation begins and report that the request is unsupported.
    """

    direction = LatticeDirection(
        sc,
        np.array(
            [[1, 0, 0]],
            dtype=int,
        ),
        basis_directions="primitive_vector",
    )

    with pytest.raises(
        ValueError,
        match="Unknown computation",
    ):
        direction.compute(
            "not_a_real_computation"
        )


def test_compute_rejects_empty_request(
    sc,
):
    """
    An empty computation list should be rejected instead of silently
    executing an empty pipeline.
    """

    direction = LatticeDirection(
        sc,
        np.array(
            [[1, 0, 0]],
            dtype=int,
        ),
        basis_directions="primitive_vector",
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        direction.compute([])