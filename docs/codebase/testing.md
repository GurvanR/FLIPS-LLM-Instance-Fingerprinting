# Testing

Tier 0 of the reproduction ([`../reproduction.md`](../reproduction.md)): everything runs on CPU against the
in-repo fixture, with no download.

## Test suite

```bash
make test           # python -m pytest tests -q
```

| Directory / file | What it covers |
|---|---|
| `tests/test_features.py` | the 57 statistics and the feature grid |
| `tests/test_protocol.py`, `test_imputation.py`, `test_hierarchical.py`, `test_log_sum_aggregation.py`, `test_unseen_detector.py` | the protocol (splits, draws, samples), the imputer, the classifier, the aggregation and the detector |
| `tests/test_closed_set_reference.py`, `test_open_set_reference.py` | the closed-set and open-set pipelines against stored reference outputs on the fixture |
| `tests/test_pipeline.py` | the experiment driver: configs, cells, summary files |
| `tests/test_experiment_config_schema.py` | every YAML under `XP_configs/` validates |
| `tests/test_main_scenario_237.py`, `test_variation_resolver.py` | the 237-instance scenario and the instance labels |
| `tests/generation/` | GPU-free parts of the generation path: chat-template rendering, token-id prompts and stop tokens, request configs (needs the `generation` extra for some tests; those skip without it) |
| `tests/paper/` | the analysis tier: data access, the driver, the comparison with `paper/expected/`, the stored-baseline curve, `fetch_data.py`; the end-to-end test needs the results archive |
| `tests/smoke/test_import.py` | the package imports |

`tests/fixtures/` holds the fixture (`answers.parquet`: 22 instances, 2 token pairs, 200 sequences each) and the
reference outputs (`reference/`) described in `tests/fixtures/README.md`.

### Markers

| Marker | Meaning |
|---|---|
| `requires_data` | the test needs the results archive unpacked under `$FLIPS_DATA` (default `data/`); skipped when it is absent |

Markers are declared in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Gates

| Script (target) | What it checks |
|---|---|
| `scripts/gates/gate_a.py` (`make gate-a`) | runs `XP_configs/smoke.yaml` on the fixture and checks that every cell is present and that accuracy is above chance at every `N_t` and higher at the largest `N_t` than with one query |
| `scripts/gates/gate_golden.py` (`make gate-golden`) | runs the same config and compares every cell's accuracy at every `N_t` with `tests/fixtures/reference/smoke_summary.json` (exact for the pinned environment; `--tolerance` allows a band) |

## Data integrity

```bash
make fetch-check    # python scripts/fetch_data.py --check-only
```

Classifies each entry of `data/manifest.sha256`: PASS (present, hash matches), FAIL (present, hash differs),
missing (not fetched yet; not a failure). The exit status is non-zero only on a FAIL.

## Lint

```bash
flake8 src/audit_llm/
```

Configured in `setup.cfg` (maximum line length 120). Advisory: CI reports findings without failing. `.pre-commit-config.yaml` runs `reuse lint` on push (`poetry run pre-commit install`).

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. `poetry install` (core and dev groups, no extras)
2. flake8 on `src/audit_llm/` (advisory)
3. import smoke: `python -c "import audit_llm"`
4. `python -m pytest tests -q`
5. `python scripts/gates/gate_a.py`
6. `reuse lint`
