# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Experiment driver: answers -> feature grid -> closed-set or open-set evaluation, per (token pair, split) cell.

Command line::

    # build the feature grid of a generation run once (NIST tests: CPU-hours on the full corpus)
    python -m audit_llm.pipeline build-grid --answers <answers.parquet | dir of parquet files> --out <grid dir> [--n-jobs 8]
    # run an experiment from a grid (or directly from the answers)
    python -m audit_llm.pipeline run --config XP_configs/closed_set.yaml --grid <grid dir> --out <results dir>
    python -m audit_llm.pipeline run --config XP_configs/smoke.yaml --answers tests/fixtures/answers.parquet --out /tmp/smoke

Every cell writes ``<out>/cells/<token pair>_s<split>[_d<draw>].json`` (per-sample labels, predictions and
confidences at every N_t); ``<out>/summary.json`` holds the accuracies per N_t (mean over cells, and per cell).
XGBoost ``hist`` results depend on the thread count (``classifier.n_jobs``; the default is 4).
"""

import argparse
import glob
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

from audit_llm.Classification.hierarchical import PER_LLM_STREAM, HierarchicalClassifier
from audit_llm.Classification.protocol import make_splits, pair_index, prepare_cell, sample_batches, select_rows
from audit_llm.experiment_config_schema import load_experiment_config
from audit_llm.features.grid import Grid, build_grid

logger = logging.getLogger(__name__)


def read_answers(path: str):
    """Answers table from a parquet file or a directory of parquet files."""
    import polars as pl

    files = sorted(glob.glob(os.path.join(path, "*.parquet"))) if os.path.isdir(path) else [path]
    if not files:
        raise FileNotFoundError(f"no parquet file in {path}")
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal")


def scenario_instances(scenario: str) -> list:
    """Instance names selected by a scenario (a name under ``config/scenarios/`` or a path to its YAML)."""
    from audit_llm.features.grid import class_name
    from audit_llm.path_utils import get_repository_level_path
    from audit_llm.scenarios.enumerator import build_instances
    from audit_llm.scenarios.loader import load_scenario

    path = scenario if os.path.exists(scenario) else os.path.join(get_repository_level_path(), "config", "scenarios", f"{scenario}.yaml")
    return [class_name(i.storage_name, i.temperature, i.system_prompt_idx) for i in build_instances(load_scenario(path))]


def _cell_data(grid: Grid, token_pair: str, cfg: dict):
    p = cfg["protocol"]
    X, y, flat = select_rows(grid.pair(token_pair).features, grid.feature_names)
    used = p["splits"] if cfg["experiment_fun"] == "closed_set" else [s for s, _ in cfg["open_set"]["cells"]]
    splits = make_splits(y, max(used) + 1, p["test_per_class"])
    return X, y, flat, splits


def closed_set_cell(grid: Grid, token_pair: str, split: int, cfg: dict) -> dict:
    """Train the hierarchical instance classifier of one cell and score its test samples at every N_t."""
    t0 = time.time()
    p, c = cfg["protocol"], cfg["classifier"]
    X, y, _, splits = _cell_data(grid, token_pair, cfg)
    cell = prepare_cell(X, y, splits, token_pair, split, p["n_train"], p["inner_val"], p["seed"])
    clf = HierarchicalClassifier(grid.class_names, c["global_rounds"], c["n_jobs"], c["round_selection_patience"], c["round_selection_max"])
    clf.fit(cell.X_train, cell.y_train, (PER_LLM_STREAM, split, pair_index(token_pair)))
    L, L2 = clf.global_log_probs(cell.X_test), clf.per_llm_log_probs(cell.X_test)
    out = dict(token_pair=token_pair, split=split, n_classes=grid.n_classes, n_train_per_class_min=int(min(cell.rows_per_class.values())),
               n_test=int(len(cell.y_test)),
               per_llm_rounds={clf.llm_names[g]: m["rounds"] for g, m in clf.per_llm.items()}, by_n_t={})
    for n_t in cfg["batch_sizes"]:
        batches, labels = sample_batches(cell.y_test, n_t)
        if not len(labels):
            continue
        d = clf.decide(L, L2, batches)
        out["by_n_t"][str(n_t)] = dict(
            n_samples=int(len(labels)), accuracy=float((d["pred"] == labels).mean()), accuracy_global=float((d["global_pred"] == labels).mean()),
            llm_accuracy=float((d["llm"] == clf.blocks[labels]).mean()),
            labels=labels.tolist(), predictions=d["pred"].tolist(), global_predictions=d["global_pred"].tolist(),
            confidence=d["confidence"].round(8).tolist())
    out["seconds"] = round(time.time() - t0, 1)
    return out


def open_set_cell(grid: Grid, token_pair: str, split: int, draw: int, cfg: dict) -> dict:
    """Open-set evaluation of one (token pair, split, draw) cell: Unseen instances drawn with ``draw``, instance classifier
    and Unseen detector built from the Known training sequences, test samples of Known and Unseen instances at every N_t."""
    from audit_llm.Classification.unseen_detector import evaluate_open_set_cell

    t0 = time.time()
    p, c, o = cfg["protocol"], cfg["classifier"], cfg["open_set"]
    X, y, _, splits = _cell_data(grid, token_pair, cfg)
    fitted, by_n_t = evaluate_open_set_cell(
        X, y, splits, grid.class_names, token_pair, split, draw, batch_sizes=cfg["batch_sizes"], target_recall=o["target_recall"],
        n_train=p["n_train"], inner_val=p["inner_val"], seed=p["seed"], n_unseen=o["n_unseen"], n_folds=o["n_folds"],
        held_out_per_class=o["held_out_per_class"], unseen_rows_per_instance=o["unseen_rows_per_instance"],
        known_batches_per_instance=o["known_batches_per_instance"], unseen_batches_per_instance=o["unseen_batches_per_instance"],
        min_rows_per_instance=o["min_rows_per_instance"], unseen_weight_share=o["unseen_weight_share"], C=o["regularization_c"],
        global_rounds=c["global_rounds"], round_patience=c["round_selection_patience"], round_max=c["round_selection_max"], n_jobs=c["n_jobs"])
    out = dict(token_pair=token_pair, split=split, draw=draw, n_classes=grid.n_classes, unseen=[grid.class_names[i] for i in fitted.unseen],
               n_test=int(len(fitted.cell.y_test)), by_n_t={})
    nan_to_none = lambda v: None if v != v else float(v)
    for n_t, r in by_n_t.items():
        if not len(r["labels"]):
            continue
        out["by_n_t"][str(n_t)] = dict(
            n_samples=int(len(r["labels"])), accuracy=r["accuracy"], unseen_recall=nan_to_none(r["unseen_recall"]),
            unseen_precision=nan_to_none(r["unseen_precision"]), accuracy_global=None, known_accuracy_closed=r["known_accuracy_closed"],
            threshold=float(r["threshold"]), labels=r["labels"].tolist(), is_unseen=r["is_unseen"].astype(int).tolist(),
            predictions=r["pred"].tolist(), hierarchical_predictions=r["hierarchical_pred"].tolist(), unseen_scores=np.round(r["scores"], 8).tolist())
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _run_cell(args):
    grid_root, grid, token_pair, split, draw, cfg = args
    if grid is None:
        grid = Grid.load(grid_root)
    if cfg["experiment_fun"] == "closed_set":
        return closed_set_cell(grid, token_pair, split, cfg)
    return open_set_cell(grid, token_pair, split, draw, cfg)


def _summary(cells: list, cfg: dict) -> dict:
    keys = ["accuracy", "accuracy_global"] + (["unseen_recall", "unseen_precision"] if cfg["experiment_fun"] == "open_set" else [])
    by_n_t = {}
    for n_t in map(str, cfg["batch_sizes"]):
        vals = {k: [c["by_n_t"][n_t][k] for c in cells if n_t in c["by_n_t"] and c["by_n_t"][n_t].get(k) is not None] for k in keys}
        by_n_t[n_t] = {f"mean_{k}": (float(np.mean(v)) if v else None) for k, v in vals.items()} | dict(n_cells=len(vals["accuracy"]))
    per_cell = [{k: v for k, v in c.items() if k != "by_n_t"} | {"accuracy": {n: r["accuracy"] for n, r in c["by_n_t"].items()}} for c in cells]
    return dict(experiment_fun=cfg["experiment_fun"], xp_name=cfg.get("xp_name"), by_n_t=by_n_t, cells=per_cell)


def run_experiment(cfg: dict, grid: Grid, out_dir: str) -> dict:
    """Run every cell of the experiment ``cfg`` (validated dict) on ``grid``; writes the cell files and ``summary.json``."""
    if cfg.get("scenario"):
        grid = grid.select_classes(scenario_instances(cfg["scenario"]))
    pairs = cfg["token_pairs"] or grid.token_pairs
    missing = [t for t in pairs if t not in grid.token_pairs]
    if missing:
        raise KeyError(f"token pairs not in the grid: {missing}")
    if cfg["experiment_fun"] == "closed_set":
        cells = [(tp, s, None) for tp in pairs for s in cfg["protocol"]["splits"]]
    else:
        cells = [(tp, s, d) for tp in pairs for s, d in cfg["open_set"]["cells"]]
    os.makedirs(os.path.join(out_dir, "cells"), exist_ok=True)
    shared = grid if grid.root is None else None  # an in-memory grid is passed to the workers, a saved one is reopened
    tasks = [(grid.root, shared, tp, s, d, cfg) for tp, s, d in cells]
    if cfg["n_workers"] > 1:
        with ProcessPoolExecutor(cfg["n_workers"]) as pool:
            results = list(pool.map(_run_cell, tasks))
    else:
        results = [_run_cell(t) for t in tasks]
    for r in results:
        name = f"{r['token_pair']}_s{r['split']}" + (f"_d{r['draw']}" if "draw" in r else "")
        with open(os.path.join(out_dir, "cells", name + ".json"), "w") as f:
            json.dump(r, f)
    summary = _summary(results, cfg)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-grid", help="compute the feature grid of an answers table")
    b.add_argument("--answers", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--token-pairs", default=None, help="comma list (default: all)")
    b.add_argument("--min-seq-length", type=int, default=100)
    b.add_argument("--n-jobs", type=int, default=os.cpu_count())
    r = sub.add_parser("run", help="run an experiment config")
    r.add_argument("--config", required=True)
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--grid", help="grid directory written by build-grid")
    src.add_argument("--answers", help="answers parquet file or directory (the grid is computed in memory)")
    r.add_argument("--out", required=True)
    r.add_argument("--n-jobs", type=int, default=os.cpu_count(), help="processes for the NIST tests (with --answers)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.command == "build-grid":
        pairs = a.token_pairs.split(",") if a.token_pairs else None
        grid = build_grid(read_answers(a.answers), pairs, a.min_seq_length, a.n_jobs)
        grid.save(a.out)
        logger.info("grid of %d instances x %d token pairs written to %s", grid.n_classes, len(grid.token_pairs), a.out)
        return
    cfg = load_experiment_config(a.config)
    cfg["xp_name"] = cfg.get("xp_name") or os.path.splitext(os.path.basename(a.config))[0]
    if a.grid:
        grid = Grid.load(a.grid)
    else:
        grid = build_grid(read_answers(a.answers), cfg["token_pairs"] or None, cfg["min_seq_length"], a.n_jobs)
    t0 = time.time()
    summary = run_experiment(cfg, grid, a.out)
    for n_t, v in summary["by_n_t"].items():
        logger.info("N_t=%s: %s", n_t, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
    logger.info("done in %.0f s -> %s", time.time() - t0, a.out)


if __name__ == "__main__":
    main()
