import logging
logger = logging.getLogger(__name__)

from math import isnan as isnan
from numpy import abs as abs
from numpy import append as append
from numpy import array as array
from numpy import clip as clip
from numpy import cumsum as cumsum
from numpy import ones as ones
from numpy import sqrt as sqrt
from numpy import sum as sum
from numpy import transpose as transpose
from numpy import where as where
from numpy import zeros as zeros
from scipy.special import erfc as erfc
from scipy.special import gammaincc as gammaincc

import numpy as np
from scipy.special import gammaincc
from math import nan

class RandomExcursions:
    
    @staticmethod
    def random_excursions_test(binary_data: str,
                            verbose: bool = False,
                            state: int = 1):
        """
        NIST SP800-22 Random Excursions Test for a single state.
        
        :param binary_data:  the input bit‐string (e.g. '010101...')
        :param verbose:      if True, prints detailed debug info
        :param state:        the state to test (must be one of [-4,-3,-2,-1,1,2,3,4])
        :return:             (p_value, passed, t_stat)
        """
        # Quick safety checks
        n = len(binary_data)
        if n == 0 or binary_data.count('0') == 0 or binary_data.count('1') == 0:
            return nan, False, nan

        # map bits → ±1
        seq = np.array([1 if b=='1' else -1 for b in binary_data], dtype=float)

        # cumulative sums S1…Sn, with padding zeros at both ends
        S = np.concatenate(([0], np.cumsum(seq), [0]))

        # our states of interest and index lookup
        x_values = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        try:
            idx = list(x_values).index(state)
        except ValueError:
            raise ValueError(f"Invalid state {state}; must be one of {x_values.tolist()}")

        # find zero‐crossing positions → extract cycles
        zero_pos = np.where(S == 0)[0]
        if len(zero_pos) < 2:
            return nan, False, nan

        cycles = [ S[zero_pos[i]:zero_pos[i+1]+1] for i in range(len(zero_pos)-1) ]
        num_cycles = len(cycles)
        if num_cycles == 0:
            return nan, False, nan

        # count visits to each state in each cycle, clipped at 5
        visits = np.array([
            np.clip([np.count_nonzero(cycle == x) for x in x_values], 0, 5)
            for cycle in cycles
        ]).T    # shape = (8 states) × (num_cycles)

        # su[k,j] = # of cycles with exactly k visits to state j, for k=0…5
        # → shape (6 visit‐counts) × (8 states) → transpose → (8 states) × 6
        su = np.array([
            [(visits[j] == k).sum() for k in range(6)]
            for j in range(len(x_values))
        ])

        # expected probabilities π(k) for each (k=0…5) and each state
        pi = np.array([
            [ RandomExcursions.get_pi_value(k, x) for k in range(6) ]
            for x in x_values
        ])

        # compute χ² for each state
        inner = num_cycles * pi
        # guard against zero‐division
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2 = ((su - inner)**2 / inner).sum(axis=1)

        # p‐values via the incomplete gamma
        p_vals = gammaincc(2.5, chi2/2.0)

        # pick out our requested state
        t_stat  = float(chi2[idx]) if np.isfinite(chi2[idx]) else nan
        p_value = float(p_vals[idx]) if np.isfinite(p_vals[idx]) else nan
        passed  = bool(p_value >= 0.01)

        if verbose:
            logger.debug("Random Excursions Test DEBUG:")
            logger.debug(f"  Input length = {n}, cycles = {num_cycles}")
            logger.debug("  State   χ²-stat    p-value    pass?")
            for i, x in enumerate(x_values):
                χ2i = chi2[i] if np.isfinite(chi2[i]) else nan
                pi_val = p_vals[i] if np.isfinite(p_vals[i]) else nan
                logger.debug(f"    {x:>2}    {χ2i:8.4f}    {pi_val:8.4f}    {pi_val>=0.01}")
            logger.debug("-- end DEBUG --")

        return p_value, passed, t_stat


    @staticmethod
    def get_pi_value(k, x):
        """
        This method is used by the random_excursions method to get expected probabilities
        """
        if k == 0:
            out = 1 - 1.0 / (2 * abs(x))
        elif k >= 5:
            out = (1.0 / (2 * abs(x))) * (1 - 1.0 / (2 * abs(x))) ** 4
        else:
            out = (1.0 / (4 * x * x)) * (1 - 1.0 / (2 * abs(x))) ** (k - 1)
        return out
