import numpy as np
import pytest


@pytest.mark.parametrize(
    "uvw, expected_d, expected_m",
    [
        ([1, 0, 0], 1.0, 1),
        ([1, 1, 0], 1.0 / np.sqrt(2.0), 2),
        ([1, 2, 3], 1.0 / np.sqrt(14.0), 14),
    ],
)
def test_sc_interplanar_spacing_known_good(
    sc,
    direction_with_planes_factory,
    uvw,
    expected_d,
    expected_m,
):
    d = direction_with_planes_factory(sc, uvw)

    assert d.lattice_interplanar_spacings[0] == pytest.approx(
        expected_d
    )

    assert d.multiplicity[0] == expected_m

    assert d.lattice_spacings[0] == pytest.approx(
        expected_m * expected_d
    )


def test_sc_111_interplanar_spacing(
    sc,
    direction_with_planes_factory,
):
    d = direction_with_planes_factory(sc, [1, 1, 1])

    assert d.multiplicity[0] == 3

    assert d.lattice_interplanar_spacings[0] == pytest.approx(
        1.0 / np.sqrt(3.0)
    )


def test_fcc_110_interplanar_spacing(
    fcc,
    direction_with_planes_factory,
):
    d = direction_with_planes_factory(fcc, [1, 1, 0])

    assert d.multiplicity[0] == 2

    assert d.lattice_interplanar_spacings[0] == pytest.approx(
        1.0 / np.sqrt(8.0)
    )


def test_fcc_111_interplanar_spacing(
    fcc,
    direction_with_planes_factory,
):
    d = direction_with_planes_factory(fcc, [1, 1, 1])

    assert d.multiplicity[0] == 3

    assert d.lattice_interplanar_spacings[0] == pytest.approx(
        1.0 / np.sqrt(3.0)
    )


def test_fcc_121_interplanar_spacing(
    fcc,
    direction_with_planes_factory,
):
    d = direction_with_planes_factory(fcc, [1, 2, 1])

    assert d.multiplicity[0] == 6

    assert d.lattice_interplanar_spacings[0] == pytest.approx(
        1.0 / np.sqrt(24.0)
    )


def test_bcc_111_interplanar_spacing(
    bcc,
    direction_with_planes_factory,
):
    d = direction_with_planes_factory(bcc, [1, 1, 1])

    assert d.multiplicity[0] == 3

    assert d.lattice_interplanar_spacings[0] == pytest.approx(
        1.0 / np.sqrt(12.0)
    )


def test_hexagonal_c_axis_plane_spacing(
    hcp_lattice,
    direction_with_planes_factory,
):
    d = direction_with_planes_factory(
        hcp_lattice,
        [0, 0, 1],
    )

    expected_c = np.sqrt(8.0 / 3.0)

    assert d.multiplicity[0] == 1

    assert d.lattice_interplanar_spacings[0] == pytest.approx(
        expected_c
    )

def test_fcc_111_relative_plane_shifts(
    fcc,
    direction_with_planes_factory,
):
    """
    FCC (111) planes repeat with the three
    Shockley-type registry shifts.
    """

    d = direction_with_planes_factory(
        fcc,
        [1, 1, 1],
    )

    d.compute("relative_shift_of_lattice_planes")

    shifts = d.shortest_relative_shift_vectors[0]

    expected = np.array(
        [
            [1.0, -2.0, 1.0],
            [-2.0, 1.0, 1.0],
            [1.0, 1.0, -2.0],
        ]
    ) / 6.0

    assert len(shifts) == 3

    # The three symmetry-equivalent shifts need not
    # be returned in a particular ordering.
    for expected_shift in expected:
        assert np.any(
            np.all(
                np.isclose(
                    shifts,
                    expected_shift,
                    atol=1e-12,
                ),
                axis=1,
            )
        )