"""The set of 31 token pairs of the paper.

``['0', '1']`` plus 30 pairs drawn with seed 70 from the intersection of the vocabularies of the tokenizers
(``Productions/Intersection_vocab/``). The draw reads the intersection file at import time.
"""

from typing import Dict, List

from audit_llm.Bits_Generation.parsing_bits_tools import token_pair_to_string
from audit_llm.Tokens_analysis.token_sampling import get_random_token_pairs_from_intersection

FLIPS_SET_OF_TOKEN_PAIRS = [["0", "1"]] + get_random_token_pairs_from_intersection(nb_of_uplets=30, seed=70, monochar=False)

TOKEN_PAIRS_SETS_DICT: Dict[str, List[str]] = {
    "FLIPS_SET_OF_TOKEN_PAIRS": [token_pair_to_string(token_pair) for token_pair in FLIPS_SET_OF_TOKEN_PAIRS],
}
