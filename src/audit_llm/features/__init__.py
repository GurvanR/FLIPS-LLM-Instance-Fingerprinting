# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Features of a generated bit sequence and the per-token-pair grid.

The 57 features are the 31 test statistics of the NIST suite, the sequence length and 25 complementary statistics
(19 on the dependence on past bits, 6 on the opening of the sequence). ``features.nist`` computes 28 of the NIST
statistics and the sequence length; ``features.statistics`` computes the complementary statistics and the 3 other
NIST statistics.
"""

from audit_llm.features.grid import Grid, TokenPairGrid, build_grid, feature_names
from audit_llm.features.nist import nist_feature_names
from audit_llm.features.statistics import STATISTIC_NAMES, sequence_statistics

__all__ = ["Grid", "TokenPairGrid", "build_grid", "feature_names", "nist_feature_names", "STATISTIC_NAMES", "sequence_statistics"]
