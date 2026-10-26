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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known Stage-1 bug: multiplicity near 3 is "
        "truncated instead of rounded."
    ),
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known Stage-1 bug: FCC [111] multiplicity "
        "is truncated from ~3 to 2."
    ),
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known Stage-1 bug: FCC [121] multiplicity "
        "is truncated from ~6 to 5."
    ),
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known Stage-1 bug: BCC [111] multiplicity "
        "is truncated from ~3 to 2."
    ),
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