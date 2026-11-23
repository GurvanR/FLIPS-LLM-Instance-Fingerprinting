"""paper/expected/ is complete and clean (no archive needed)."""
import os

from paper import check

EXP = check.EXPECTED
MACRO_FILES = {"numbers.tex": 62, "figures/attr_numbers.tex": 2, "figures/distance2_numbers.tex": 2, "figures/discard_macros.tex": 5}


def test_macro_files_define_every_macro_once():
    for rel, n in MACRO_FILES.items():
        names = [name for name, _ in check.macros(open(os.path.join(EXP, rel)).read())]
        assert len(names) == len(set(names)) == n, rel
    names = {name for rel in MACRO_FILES for name, _ in check.macros(open(os.path.join(EXP, rel)).read())}
    assert {"nClosedThirtyTwo", "nAucThirtyTwo", "nLLMmapBest", "nImpShareNist", "fgAttrClasses", "nDtwAucDtwoThirtyTwo", "rDiscardAll"} <= names


def test_expected_has_the_sidecars_the_numbers_read():
    for rel in ("closed/numbers.json", "figures/tab_openset_op.json", "figures/fig10_feature_importance.json", "figures/fig12_closed_heatmap.json",
                "figures/fig04_zero_one_vs_pairs.json",
                "figures/absorption.json", "figures/attr_miss.json", "figures/fig06_unseen_pr.json", "figures/distance2_numbers.tex", "figures/attr_numbers.tex"):
        assert os.path.exists(os.path.join(EXP, rel)), rel
