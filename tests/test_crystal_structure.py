import numpy as np
import pytest

from crystalytics_py.crystal_structure import CrystalStructure


def test_default_single_motif_is_at_origin():

    crystal = CrystalStructure(
        np.eye(3, dtype=float)
    )

    assert len(crystal.motifs) == 1

    np.testing.assert_allclose(
        crystal.motifs["pos"][0],
        np.zeros(3),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known Stage-1 bug: default motif_types length "
        "is based on dimension, not number of motifs."
    ),
)
def test_default_motif_types_preserve_all_motifs():

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.50, 0.50, 0.50],
            [0.75, 0.75, 0.75],
        ],
        dtype=float,
    )

    crystal = CrystalStructure(
        np.eye(3, dtype=float),
        motif_positions=positions,
    )

    assert len(crystal.motifs) == 4

    np.testing.assert_allclose(
        crystal.motifs["pos"],
        positions,
    )

    np.testing.assert_array_equal(
        crystal.motifs["type"],
        np.ones(4, dtype=int),
    )


def test_explicit_motif_types_preserve_all_motifs():

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.50, 0.50, 0.50],
            [0.75, 0.75, 0.75],
        ],
        dtype=float,
    )

    types = ["A", "B", "A", "B"]

    crystal = CrystalStructure(
        np.eye(3, dtype=float),
        motif_positions=positions,
        motif_types=types,
    )

    assert len(crystal.motifs) == 4

    np.testing.assert_array_equal(
        crystal.motifs["type"],
        types,
    )