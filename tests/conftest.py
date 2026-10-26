import numpy as np
import pytest

from crystalytics_py.crystal_structure import CrystalStructure
from crystalytics_py.direction import Direction


@pytest.fixture
def sc():
    """Simple-cubic Bravais lattice with conventional lattice parameter a = 1."""
    return CrystalStructure(np.eye(3, dtype=float))


@pytest.fixture
def fcc():
    """FCC primitive lattice with conventional cubic lattice parameter a = 1."""
    primitive_vectors = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ],
        dtype=float,
    )
    return CrystalStructure(primitive_vectors)


@pytest.fixture
def bcc():
    """BCC primitive lattice with conventional cubic lattice parameter a = 1."""
    primitive_vectors = 0.5 * np.array(
        [
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
        ],
        dtype=float,
    )
    return CrystalStructure(primitive_vectors)


@pytest.fixture
def hcp_lattice():
    """
    Hexagonal Bravais lattice with a = 1 and ideal c/a = sqrt(8/3).

    This fixture tests non-orthogonal primitive vectors only. The two-atom HCP
    motif is intentionally left for the motif/layer tests introduced later.
    """
    c_over_a = np.sqrt(8.0 / 3.0)

    primitive_vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, c_over_a],
        ],
        dtype=float,
    )

    return CrystalStructure(primitive_vectors)


@pytest.fixture
def direction_factory():
    """
    Factory for constructing a Direction and computing lattice spacing.

    Stage 0 deliberately calls the current private computation routine because
    Direction.compute() does not yet populate/use its computation pipeline.

    No production API is changed merely to make the tests convenient.
    """

    def _make(crystal_structure, direction):

        direction = np.asarray(direction, dtype=int).reshape(1, -1)

        obj = Direction(
            crystal_structure,
            direction,
            basis_directions="global_orthonormal",
        )

        obj._compute_lattice_spacing()

        return obj

    return _make


@pytest.fixture
def direction_with_planes_factory(direction_factory):
    """Factory that additionally computes interplanar spacing/multiplicity."""

    def _make(crystal_structure, direction):

        obj = direction_factory(crystal_structure, direction)

        obj._compute_lattice_interplanar_spacing()

        return obj

    return _make