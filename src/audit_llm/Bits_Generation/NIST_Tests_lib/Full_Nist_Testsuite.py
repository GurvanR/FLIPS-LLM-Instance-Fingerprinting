"""The NIST SP 800-22 tests of the feature set: registry of the tests and naming of their columns.

Each test returns ``(p_value, pass_flag, test_statistic)``; ``audit_llm.features.nist`` keeps the statistic and the
p-value of the tests listed in ``nist_test_parameters``.
"""

import random
from itertools import product
from typing import Any, Dict, List, Sequence

from audit_llm.Bits_Generation.NIST_Tests_lib.ApproximateEntropy import ApproximateEntropy as aet
from audit_llm.Bits_Generation.NIST_Tests_lib.Complexity import ComplexityTest as ct
from audit_llm.Bits_Generation.NIST_Tests_lib.CumulativeSum import CumulativeSums as cst
from audit_llm.Bits_Generation.NIST_Tests_lib.FrequencyTest import FrequencyTest as ft
from audit_llm.Bits_Generation.NIST_Tests_lib.Matrix import Matrix as mt
from audit_llm.Bits_Generation.NIST_Tests_lib.RandomExcursions import RandomExcursions as ret
from audit_llm.Bits_Generation.NIST_Tests_lib.RunTest import RunTest as rt
from audit_llm.Bits_Generation.NIST_Tests_lib.Spectral import SpectralTest as st
from audit_llm.Bits_Generation.NIST_Tests_lib.TemplateMatching import TemplateMatching as tm


def block_frequency_test_maker(parameter: Dict):
    return lambda binary_data: ft.block_frequency(binary_data, **parameter)  # parameter: block_size M


def non_overlapping_test_maker(parameter: Dict):
    return lambda binary_data: tm.non_overlapping_test(binary_data, **parameter)  # template_pattern, block


def overlapping_patterns_test_maker(parameter: Dict):
    return lambda binary_data: tm.overlapping_patterns(binary_data, **parameter)  # template_pattern, block_size


# Parameters of the template-matching tests (set in ``audit_llm.features.nist.nist_test_parameters``): templates of
# 2 and 3 bits, and 75 for ``block`` (non-overlapping test) and ``block_size`` (overlapping test). NIST SP 800-22
# requires N > 5 / min(pi_i) blocks and n > M * N.
NIST_TESTS = {
    "block frequency": block_frequency_test_maker,  # NIST recommendation: n > 100, M > 20
    "non overlapping": non_overlapping_test_maker,  # NIST recommendation: m = 9 or 10, N <= 100
    "overlapping patterns": overlapping_patterns_test_maker,  # m should be chosen so that m ~ log2(M)
    "monobit": ft.monobit_test,
    "run": rt.run_test,
    "longest one block": rt.longest_one_block_test,
    "binary matrix rank": mt.binary_matrix_rank_test,  # the matrix sizes are fixed: the class probabilities depend on them
    "spectral": st.spectral_test,
    "cumulative sums 1s": lambda binary_data: cst.cumulative_sums_test(binary_data, terms='one'),
    "cumulative sums 2s": lambda binary_data: cst.cumulative_sums_test(binary_data, terms='two'),
    "linear complexity": ct.linear_complexity_test,  # NIST recommends n >= 1e6; the test statistic is used as computed
    "approximate entropy": aet.approximate_entropy_test,  # block length m = 10 (the default of the implementation)
    "random excursion": ret.random_excursions_test,
}

def make_nist_feat_name_from_info(test_name, info):
    return test_name +'_'+'_'.join([str(item) for item in info.values()])

def is_primitive(seq: Sequence[Any]) -> bool:
    """
    Return True if `seq` is aperiodic (primitive), i.e. it cannot be
    formed by repeating a smaller subsequence.
    """
    n = len(seq)
    for d in range(1, n):
        if n % d == 0:
            if seq == seq[:d] * (n // d):  # type: ignore
                return False
    return True

def make_template_pattern(
    size: int,
    alphabet: Sequence[Any] = ('0', '1'),
    all: bool = False,
    nb_of_patterns: int = 1,
    random_seed: int = 42
) -> List[str]:
    """
    Generate up to `nb_of_patterns` distinct aperiodic (primitive)
    sequences of length `size` over the given `alphabet`.  

    Parameters
    ----------
    size : int
        Length of each sequence; must be >= 1.
    alphabet : Sequence[Any]
        The set of symbols to use (default ('0','1')).
    all : bool
        If True, ignore `nb_of_patterns` and return every possible
        primitive sequence.
    nb_of_patterns : int
        How many random primitives to return (if `all` is False).
    random_seed : int
        Seed for reproducible sampling.

    Returns
    -------
    List[str]
        A list of primitive sequences (each a string like '0001').
    """
    if size < 1:
        raise ValueError("Size must be at least 1")

    # Generate all sequences of length `size`
    all_seqs = list(product(alphabet, repeat=size))

    # Filter to primitives and convert each to string
    primitive_seqs = ["".join(seq) for seq in all_seqs if is_primitive(seq)]

    if all or nb_of_patterns >= len(primitive_seqs):
        return primitive_seqs
    else:
        random.seed(random_seed)
        return random.sample(primitive_seqs, nb_of_patterns)


def Nist_index(rng_tests_dict: Dict[str, List[Dict]]):
    """Column position of the test statistic (``_ts``) and the p-value (``_pv``) of every test, in the order of the dict."""
    test_scores_idx = {}

    tests = []
    for test, parameters in rng_tests_dict.items():
        if parameters:
            tests.extend(make_nist_feat_name_from_info(test, parameter) for parameter in parameters)
        else:
            tests.append(test)
    i = 0
    for test in tests:
        test_scores_idx[f"{test}_ts"] = i
        test_scores_idx[f"{test}_pv"] = i + 1
        i += 2

    return test_scores_idx
