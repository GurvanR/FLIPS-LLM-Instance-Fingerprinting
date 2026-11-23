# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Hierarchical instance classifier: LLM grouping, routing and the per-LLM decision."""

import numpy as np

from audit_llm.Classification.hierarchical import HierarchicalClassifier, decide_within, llm_blocks, llm_groups, llm_name

NAMES = ["org/a_temp-1.0_sp--1", "org/a@@fp8_temp-1.0_sp--1", "org/a_temp-0.6_sp--1", "org/b_temp-1.0_sp--1", "org/c-abliterated_temp-1.0_sp--1"]


def test_llm_of_an_instance():
    assert llm_name(NAMES[1]) == "org/a" and llm_name(NAMES[4]) == "org/c-abliterated"
    blocks, llms = llm_blocks(NAMES)
    assert llms == ["org/a", "org/b", "org/c-abliterated"] and blocks.tolist() == [0, 0, 0, 1, 2]


def test_decision_stays_inside_the_selected_llm_and_singletons_are_returned():
    blocks, _ = llm_blocks(NAMES)
    groups = llm_groups(blocks)
    L2 = np.zeros((4, 5))
    L2[:, 2] = -0.1
    L2[:, [0, 1]] = -3.0
    batches = np.array([[0, 1], [2, 3]])
    assert decide_within(L2, batches, groups, np.array([0, 2])).tolist() == [2, 4]


def test_global_decision_selects_the_llm_and_the_per_llm_scores_decide():
    clf = HierarchicalClassifier(NAMES)
    L = np.log(np.array([[0.30, 0.05, 0.05, 0.55, 0.05], [0.40, 0.05, 0.05, 0.45, 0.05]], np.float32))
    L2 = np.zeros((2, 5))
    L2[:, :3] = np.log([0.1, 0.1, 0.8])
    d = clf.decide(L, L2, np.array([[0, 1]]))
    assert d["global_pred"].tolist() == [3] and d["pred"].tolist() == [3]  # org/b has one instance
    L[:, 3] = np.log(0.05)
    d = clf.decide(L, L2, np.array([[0, 1]]))
    assert d["llm"].tolist() == [0] and d["pred"].tolist() == [2]  # routed to org/a, its classifier picks instance 2
    assert 0 < d["confidence"][0] <= 1
