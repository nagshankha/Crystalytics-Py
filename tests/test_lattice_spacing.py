import numpy as np
import pytest


@pytest.mark.parametrize(
    "uvw, expected_length",
    [
        ([1, 0, 0], 1.0),
        ([1, 1, 0], np.sqrt(2.0)),
        ([1, 1, 1], np.sqrt(3.0)),
        ([1, 2, 3], np.sqrt(14.0)),
    ],
)
def test_sc_shortest_lattice_spacing(
    sc,
    direction_factory,
    uvw,
    expected_length,
):
    d = direction_factory(sc, uvw)

    assert d.lattice_spacings[0] == pytest.approx(expected_length)


@pytest.mark.parametrize(
    "uvw, expected_length",
    [
        ([1, 0, 0], 1.0),
        ([1, 1, 0], 1.0 / np.sqrt(2.0)),
        ([1, 2, 1], np.sqrt(6.0) / 2.0),
        ([1, 1, 1], np.sqrt(3.0)),
    ],
)
def test_fcc_shortest_lattice_spacing(
    fcc,
    direction_factory,
    uvw,
    expected_length,
):
    d = direction_factory(fcc, uvw)

    assert d.lattice_spacings[0] == pytest.approx(expected_length)


def test_fcc_121_shortest_translation_is_half_121(
    fcc,
    direction_factory,
):
    d = direction_factory(fcc, [1, 2, 1])

    cartesian = d.convert_primitive_to_orthonormal_basis(
        d.shortest_lattice_vectors
    )[0]

    np.testing.assert_allclose(
        cartesian,
        0.5 * np.array([1.0, 2.0, 1.0]),
    )


@pytest.mark.parametrize(
    "uvw, expected_length",
    [
        ([1, 0, 0], 1.0),
        ([1, 1, 0], np.sqrt(2.0)),
        ([1, 1, 1], np.sqrt(3.0) / 2.0),
    ],
)
def test_bcc_shortest_lattice_spacing(
    bcc,
    direction_factory,
    uvw,
    expected_length,
):
    d = direction_factory(bcc, uvw)

    assert d.lattice_spacings[0] == pytest.approx(expected_length)


def test_hexagonal_c_axis_repeat_length(
    hcp_lattice,
    direction_factory,
):
    d = direction_factory(hcp_lattice, [0, 0, 1])

    assert d.lattice_spacings[0] == pytest.approx(
        np.sqrt(8.0 / 3.0)
    )