# NIST Randomness Testsuite

This is a Python implementation of NIST's A Statistical Test Suite for Random and Pseudorandom Number Generators for
Cryptographic Applications (NIST SP 800-22 Rev. 1a), vendored here for the NIST statistics of the FLIPS feature set
(`audit_llm.features.nist`). Each test returns `(p_value, pass_flag, test_statistic)`.

The tests used are registered in `Full_Nist_Testsuite.NIST_TESTS`: block frequency, non-overlapping and overlapping
template matching, monobit, runs, longest run of ones, binary matrix rank, spectral, cumulative sums (forward and
backward), linear complexity, approximate entropy, random excursions.

## Prerequisites
Python 3.11 and above, NumPy and SciPy.

## Using the tests
```python
from audit_llm.Bits_Generation.NIST_Tests_lib.FrequencyTest import FrequencyTest
from audit_llm.Bits_Generation.NIST_Tests_lib.RunTest import RunTest
from audit_llm.Bits_Generation.NIST_Tests_lib.Spectral import SpectralTest

binary_data = "1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000"
print(FrequencyTest.monobit_test(binary_data))
print(RunTest.run_test(binary_data))
print(SpectralTest.spectral_test(binary_data))
```
