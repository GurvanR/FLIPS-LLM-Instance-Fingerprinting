# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Statistics computed directly from the bits of a sequence (columns 29 to 56 of the 57 features).

They are the 25 complementary statistics of the paper and 3 of its 31 NIST statistics, and are functions of the
bits only (no fitted parameter). The 19 statistics of the dependence on past bits:

- ``longest_run0``, ``longest_run1``: longest run of 0s and of 1s;
- ``switch_rate``: share of positions where the bit differs from the previous one;
- ``p1_k1_c0`` ... ``p1_k3_c7``: P(bit = 1 | previous k bits = context c) for k = 1..3 (14 values, add-1/2 smoothing);
- ``haz0_l1``, ``haz1_l1``: probability that a run of 0s (of 1s) stops after one bit; the last run, cut by the
  end of the sequence, is not counted (add-1/2 smoothing).

The 3 NIST statistics:

- ``p1_k0``: share of ones (second statistic of the Monobit test; add-1/2 smoothing);
- ``apen_m3``, ``apen_m4``: approximate entropy with m = 3 and 4 (overlapping, non-cyclic m-grams).

The 6 statistics of the opening of the sequence:

- ``op_run0_sym``, ``op_run0_len``: symbol and length of the first run;
- ``op_first8``: the first 8 bits read as an integer;
- ``op_w0_ones``: share of ones among the first 16 bits;
- ``op_quart0_ones``, ``op_quart3_ones``: share of ones in the first and in the last quarter of the sequence.

Sequences shorter than 2 bits get NaN. The kernel is compiled with numba when it is installed (recommended for
the full corpus); otherwise the same code runs in plain Python.
"""

import os

import numpy as np

try:
    import numba
    from numba import njit, prange

    numba.set_num_threads(max(1, min(int(os.environ.get("FLIPS_NUM_THREADS", "4")), numba.config.NUMBA_NUM_THREADS)))
except ImportError:  # plain-Python fallback (same arithmetic, much slower)

    def njit(*args, **kwargs):
        return (lambda f: f) if not (len(args) == 1 and callable(args[0])) else args[0]

    prange = range

STATISTIC_NAMES = (
    "longest_run0 longest_run1 switch_rate p1_k0 p1_k1_c0 p1_k1_c1 p1_k2_c0 p1_k2_c1 p1_k2_c2 p1_k2_c3 p1_k3_c0 p1_k3_c1 "
    "p1_k3_c2 p1_k3_c3 p1_k3_c4 p1_k3_c5 p1_k3_c6 p1_k3_c7 haz0_l1 haz1_l1 apen_m3 apen_m4 op_run0_sym op_run0_len "
    "op_first8 op_w0_ones op_quart0_ones op_quart3_ones"
).split()
N_STATISTICS = len(STATISTIC_NAMES)  # 28


@njit(cache=False)
def _sequence_statistics(b, n, out):
    """Write the statistics of the bit sequence ``b[:n]`` (n >= 2) into ``out``, in STATISTIC_NAMES order."""
    # conditional probabilities P(1 | previous k bits), k = 0..3 -> out[3:18]
    pos = 3
    for k in range(4):
        n_contexts = 1 << k
        ones = np.zeros(n_contexts)
        seen = np.zeros(n_contexts)
        for t in range(k, n):
            ctx = 0
            for j in range(k):
                ctx = ctx * 2 + int(b[t - k + j])
            seen[ctx] += 1.0
            ones[ctx] += b[t]
        for c in range(n_contexts):
            out[pos] = (ones[c] + 0.5) / (seen[c] + 1.0)
            pos += 1
    # switch rate -> out[2]
    switches = 0
    for t in range(1, n):
        if b[t] != b[t - 1]:
            switches += 1
    out[2] = switches / max(n - 1, 1)
    # runs: longest run of each symbol -> out[0:2]; stop-after-one-bit hazards (completed runs only) -> out[18:20]
    stopped_at_1 = np.zeros(2)
    completed = np.zeros(2)
    longest = np.zeros(2)
    run = 1
    for t in range(1, n + 1):
        if t < n and b[t] == b[t - 1]:
            run += 1
        else:
            sym = b[t - 1]
            if run > longest[sym]:
                longest[sym] = run
            if t < n:  # the run really ended (it was not cut by the end of the sequence)
                completed[sym] += 1.0
                if run == 1:
                    stopped_at_1[sym] += 1.0
            run = 1
    out[0] = longest[0]
    out[1] = longest[1]
    out[18] = (stopped_at_1[0] + 0.5) / (completed[0] + 1.0)
    out[19] = (stopped_at_1[1] + 0.5) / (completed[1] + 1.0)
    # approximate entropy ApEn(m) = Phi_m - Phi_{m+1}, m = 3, 4 -> out[20:22]
    phi = np.zeros(6)
    for m in range(3, 6):
        counts = np.zeros(1 << m)
        total = n - m + 1
        if total < 1:
            continue
        for t in range(total):
            code = 0
            for j in range(m):
                code = code * 2 + int(b[t + j])
            counts[code] += 1.0
        acc = 0.0
        for c in range(1 << m):
            if counts[c] > 0:
                p = counts[c] / total
                acc += p * np.log(p)
        phi[m] = acc
    out[20] = phi[3] - phi[4]
    out[21] = phi[4] - phi[5]
    # opening: first run (symbol, length) -> out[22:24]
    r = 1
    while r < n and b[r] == b[0]:
        r += 1
    out[22] = b[0]
    out[23] = r
    # first 8 bits as an integer -> out[24]
    first8 = 0
    for j in range(min(8, n)):
        first8 = first8 * 2 + int(b[j])
    out[24] = first8
    # share of ones among the first 16 bits -> out[25]
    e = min(16, n)
    ones16 = 0.0
    for t in range(e):
        ones16 += b[t]
    out[25] = ones16 / e if e > 0 else 0.5
    # share of ones in the first and last quarters -> out[26], out[27]
    for q, slot in ((0, 26), (3, 27)):
        a = (n * q) // 4
        e = (n * (q + 1)) // 4
        s = 0.0
        for t in range(a, e):
            s += b[t]
        out[slot] = s / max(e - a, 1)


@njit(parallel=True, cache=False)
def _all_sequences(bits, lens, out):
    for i in prange(bits.shape[0]):
        n = lens[i]
        if n >= 2:
            _sequence_statistics(bits[i], n, out[i])
        else:
            for j in range(out.shape[1]):
                out[i, j] = np.nan


def sequence_statistics(bits: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """The statistics of every sequence (``STATISTIC_NAMES``).

    Args:
        bits: uint8 array ``(..., L)`` of 0/1 values (positions past the sequence length are ignored).
        lens: int array ``bits.shape[:-1]``, the length of every sequence.

    Returns:
        float64 array ``(..., 28)`` in ``STATISTIC_NAMES`` order (NaN for sequences shorter than 2 bits).
    """
    bits = np.asarray(bits)
    lead = bits.shape[:-1]
    flat_bits = np.ascontiguousarray(bits.reshape(-1, bits.shape[-1]))
    flat_lens = np.ascontiguousarray(np.asarray(lens).reshape(-1)).astype(np.int64)
    out = np.zeros((flat_bits.shape[0], N_STATISTICS))
    _all_sequences(flat_bits, flat_lens, out)
    return out.reshape(*lead, N_STATISTICS)
