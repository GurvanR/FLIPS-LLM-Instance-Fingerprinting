# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Features: the statistics of ``features.statistics`` on hand-checked sequences, the column order and the grid layout."""

import numpy as np
import polars as pl

from audit_llm.features import STATISTIC_NAMES, build_grid, feature_names, nist_feature_names, sequence_statistics


def stats_of(bit_string):
    bits = np.array([[int(c) for c in bit_string]], np.uint8)
    return dict(zip(STATISTIC_NAMES, sequence_statistics(bits, np.array([len(bit_string)]))[0]))


def test_names_and_order():
    assert len(STATISTIC_NAMES) == 28 and len(nist_feature_names()) == 29 and len(feature_names()) == 57
    assert feature_names()[28] == "seq_length" and feature_names()[29] == "longest_run0"


def test_statistics_of_an_alternating_sequence():
    s = stats_of("01" * 50)
    assert s["switch_rate"] == 1.0 and s["longest_run0"] == 1 and s["longest_run1"] == 1
    assert s["p1_k1_c0"] == (50 + 0.5) / (50 + 1.0)  # after a 0 always comes a 1
    assert s["p1_k1_c1"] == 0.5 / (49 + 1.0)  # after a 1 never comes a 1 (49 transitions)
    assert s["op_run0_sym"] == 0 and s["op_run0_len"] == 1 and s["op_first8"] == int("01010101", 2)
    assert s["op_w0_ones"] == 0.5 and s["op_quart0_ones"] == 0.48 and s["haz0_l1"] == (50 + 0.5) / (50 + 1.0)


def test_statistics_of_runs_and_short_sequences():
    s = stats_of("0001111111")
    assert s["longest_run0"] == 3 and s["longest_run1"] == 7 and s["switch_rate"] == 1 / 9
    assert s["haz0_l1"] == 0.5 / 2.0  # one completed run of 0s, of length 3; the last run (1s) is cut by the end
    out = sequence_statistics(np.zeros((1, 5), np.uint8), np.array([1]))
    assert np.isnan(out).all()


def test_grid_layout_on_a_tiny_table():
    rows = []
    for model in ("m/a", "m/b"):
        for q in range(3):
            rows.append(dict(Token_pair="tp", Model=model, **{"Dataset_Question Index": q}, Answer="01101" * 30 if q else "0110",
                             temperature=1.0, system_prompt_idx=-1))
    g = build_grid(pl.DataFrame(rows))
    assert g.class_names == ["m/a_temp-1.0_sp--1", "m/b_temp-1.0_sp--1"]
    F = g.pair("tp").features
    assert F.shape == (3, 2, 57)
    assert np.isnan(F[0]).all()  # 4-bit answer < min_seq_length -> undefined
    assert F[1, 0, 28] == 150 and np.isfinite(F[1, 0, 29:]).all()
    assert (g.pair("tp").lens[:, 0] == [4, 150, 150]).all()
