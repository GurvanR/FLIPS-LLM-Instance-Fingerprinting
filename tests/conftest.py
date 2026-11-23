# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

# This file is automatically imported by all tests.
# Add your global fixtures here

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """
    Remove errors when no tests where collected.

    See https://github.com/pytest-dev/pytest/issues/2393#issuecomment-452634365.
    """
    if exitstatus == 5:
        session.exitstatus = 0


FIXTURES = __import__("pathlib").Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_grid():
    """Feature grid of the in-repo fixture (tests/fixtures/answers.parquet), computed once per session."""
    import polars as pl

    from audit_llm.features import build_grid

    return build_grid(pl.read_parquet(FIXTURES / "answers.parquet"), n_jobs=4)


@pytest.fixture(scope="session")
def closed_reference():
    import numpy as np

    return np.load(FIXTURES / "reference" / "closed_set.npz")
