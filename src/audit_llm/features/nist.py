# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""NIST SP 800-22 statistics of a bit sequence and its length (columns 0 to 28 of the 57 features).

Every test of ``Bits_Generation.NIST_Tests_lib`` returns (p-value, pass flag, test statistic). The feature set uses
the test statistic of 28 tests, which are 28 of the 31 NIST statistics of the paper (block frequency with blocks of
30 and 100 bits, the 8 non-overlapping and the 8 overlapping templates of 2 and 3 bits with blocks of 75 bits,
cumulative sums forward and backward, random excursions, monobit, runs, longest run of ones, binary matrix rank,
spectral, linear complexity, approximate entropy with m = 10), and the sequence length. The 3 other NIST statistics
(the share of ones, the approximate entropy with m = 3 and 4) are computed in ``features.statistics``. Tests run only
on sequences longer than ``MIN_TEST_LENGTH`` bits; a test that raises gives NaN.
"""

import logging
import warnings
from multiprocessing import Pool
from typing import List, Sequence

import numpy as np

MIN_TEST_LENGTH = 10

_STATE = {}


def nist_test_parameters() -> dict:
    """Test name -> list of parameter dicts (empty list: the test has no parameter), in column order."""
    from audit_llm.Bits_Generation.NIST_Tests_lib.Full_Nist_Testsuite import make_template_pattern

    templates = [t for k in (2, 3) for t in make_template_pattern(size=k, all=True)]
    tests = {
        "block frequency": [{"block_size": 30}, {"block_size": 100}],
        "non overlapping": [{"template_pattern": t, "block": 75} for t in templates],
        "overlapping patterns": [{"template_pattern": t, "block_size": 75} for t in templates],
    }
    for name in (
        "cumulative sums 1s", "cumulative sums 2s", "random excursion", "monobit", "run", "longest one block",
        "binary matrix rank", "spectral", "linear complexity", "approximate entropy",
    ):
        tests[name] = []
    return tests


def nist_column_names() -> List[str]:
    """Names of the columns of ``nist_columns``: (test statistic ``_ts``, p-value ``_pv``) per test (56), then ``seq_length``."""
    from audit_llm.Bits_Generation.NIST_Tests_lib.Full_Nist_Testsuite import Nist_index

    index = Nist_index(nist_test_parameters())
    names = [None] * len(index)
    for name, position in index.items():
        names[position] = name
    return names + ["seq_length"]


def nist_feature_names() -> List[str]:
    """Columns 0 to 28 of the 57 features: the 28 test statistics, then ``seq_length``."""
    return [n for n in nist_column_names() if n.endswith("_ts")] + ["seq_length"]


def _init_worker():
    from audit_llm.Bits_Generation.NIST_Tests_lib.Full_Nist_Testsuite import NIST_TESTS

    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    _STATE["tests"] = NIST_TESTS
    _STATE["parameters"] = nist_test_parameters()


def _as_float(v) -> float:
    return float(v) if isinstance(v, (int, float, bool, np.bool_)) else np.nan


def nist_columns_one(bit_string: str) -> List[float]:
    """57 values for one bit string (``'0'``/``'1'`` characters): (statistic, p-value) per test, then the length."""
    if "tests" not in _STATE:
        _init_worker()
    out = []
    if len(bit_string) > MIN_TEST_LENGTH:
        for name, parameters in _STATE["parameters"].items():
            for p in parameters or [None]:
                try:
                    test = _STATE["tests"][name](p) if p is not None else _STATE["tests"][name]
                    result = test(bit_string)
                    if len(result) != 3:
                        raise ValueError(result)
                    out += [_as_float(result[2]), _as_float(result[0])]
                except Exception:  # a test that cannot be computed on this sequence
                    out += [np.nan, np.nan]
    else:
        out = [np.nan] * 56
    return out + [float(len(bit_string))]


def nist_columns(bit_strings: Sequence[str], n_jobs: int = 1) -> np.ndarray:
    """(n, 57) float64 NIST columns of every bit string (see ``nist_column_names``)."""
    if n_jobs > 1:
        with Pool(n_jobs, initializer=_init_worker) as pool:
            rows = pool.map(nist_columns_one, list(bit_strings), chunksize=256)
    else:
        rows = [nist_columns_one(s) for s in bit_strings]
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), 57)
