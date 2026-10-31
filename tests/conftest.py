"""Shared fixtures and tolerances for the cpv test suite."""

import numpy as np
import pytest

# Statements that hold exactly in exact arithmetic (commutators, unitarity,
# trace identities) are checked at a few hundred times machine epsilon.
EXACT = 1e-12

# Statements that involve a discretisation are checked against a tolerance
# derived from the measurement recorded in the relevant test.
pytest_plugins: list[str] = []


@pytest.fixture
def rng():
    return np.random.default_rng(20260401)
