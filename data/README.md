<!--
SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
SPDX-License-Identifier: CC-BY-4.0
-->
# `data/` — released data, manifests and fetch wiring

This directory documents the released FLIPS data: what is **fetched from an external
archive**, how it is **verified**, and where the analysis code reads it. The data license
(and the per-model output constraints layered on top) is in [`LICENSE`](LICENSE).

## Files

- [`manifest.sha256`](manifest.sha256) — SHA-256 of every released file, one
  `<sha256>  <relative/path>` line per file (sha256sum convention). Entries whose hash is
  **64 zeros** are externally hosted and are skipped as external by the verifier
  (never PASS/FAIL).
- [`archives.sha256`](archives.sha256) — SHA-256 of each downloadable archive (same
  convention, archive file name instead of a path). `fetch_data.py` checks the archive
  before unpacking it, then every unpacked file against `manifest.sha256`.
- [`LICENSE`](LICENSE) — CC-BY-4.0 for our data and derived results, plus the Cohere (CC-BY-NC)
  and Gemma (ToU) per-model output constraints, and the no-weight-redistribution note.

## Archives

| Archive | Unpacks to | Size | Role |
|---|---|---|---|
| `flips_results.zip` | `data/results/` (340 files) | 4.4 MB unpacked, 2.2 MB zipped | cached results: the paper's computed figures and tables and the numbers its text quotes through macros regenerate from it (`paper/make_paper.py`, about 20 seconds on a CPU), except Figure 3 and the LLMmap bars of Figure 1 (see `paper/README.md`) |
| `flips_release_data.tar.gz` (`--with-generations`) | `Productions/FLiPS_ICML_run/` (76 files), `Productions/LLMmap_ICML_run/` (600 files) | 4.6 GB zipped, 6.4 GB unpacked | the generated answers (`Analysis/Graph_Numeric/answers/`) and token ids (`Analysis/Graph_Numeric/token_ids/`) of the 237 instances, one parquet file per model, input of the feature grid (tiers 2 and 3); the run data of the LLMmap baseline (configurations, trained classifiers, logs, evaluations), provided as they are; the code that trains LLMmap is not in this repository |

Both are files of one Zenodo record (open access). Each archive also contains `README.md`,
`LICENSE`, `MANIFEST.tsv` (path, bytes, sha256 and description of the other files) and `SHA256SUMS`;
`fetch_data.py` unpacks only the files listed in `manifest.sha256` (`results/` below `data/`, and
`Productions/…` at the repository root).

The analysis code reads **one data root**: `$FLIPS_DATA` (default `data/`), below which it
expects `results/closed_set`, `results/open_set`, `results/nist`, `results/corpus`,
`results/meta` and `results/llmmap`.

## Fetching + verifying (`scripts/fetch_data.py`)

`scripts/fetch_data.py` (stdlib only) downloads the archive(s) of the record, unpacks
them (`results/` below `data/`, `Productions/` at the repository root) and verifies **every** unpacked file's SHA-256 against
`manifest.sha256` — aborting and removing the files of the failed run on any mismatch.

The data URL is resolved by priority: `--url` > `$FLIPS_DATA_URL` > the
`ZENODO_RECORD_URL` constant in the script. Use the `…/records/<id>` form of a Zenodo record
(the resolver reads the record's file list through the public Zenodo API), a direct
archive URL (`https://…/flips_results.zip`, `file:///…`), or a local directory holding the archives.

| Flag | Effect |
|---|---|
| `--check-only` | Verify present files against the manifest; **zero network**. Externally hosted entries are skipped; files not fetched yet are reported as missing (not a failure). Exit 1 only on a hash mismatch. |
| `--dry-run` | Print the fetch plan (records, archives, files, target paths); no download, no writes. |
| `--with-generations` | Also fetch the release data archive into `Productions/` (4.6 GB download; about 11 GB of free disk while it unpacks). |
| `--manifest`, `--archives`, `--dest`, `--url` | Manifest path, archive-hash list, repository root to unpack into, source URL override. |

- **Fetch the results:** `python scripts/fetch_data.py`, then
  `FLIPS_DATA=data python paper/make_paper.py && python paper/check.py`.
- **Fetch the generations as well:** `python scripts/fetch_data.py --with-generations`, then `make grid`
  (reads `Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/answers`).
- **Verify what is on disk (no network):**
  `python scripts/fetch_data.py --check-only --manifest data/manifest.sha256`.
- **Rebuild the hashes** after (re)building the archive: the archive's own `SHA256SUMS`
  lists `results/…` paths; `manifest.sha256` is the same list with a `data/` prefix, plus the
  `Productions/…` rows of the release data archive's `SHA256SUMS`, and `archives.sha256` carries the hash
  of each archive itself.

## No weight redistribution

The generated answers and token ids, the result artefacts and the LLMmap baseline's trained classifiers are
released here; the weights of the audited models are not. They are obtained from their original sources under
their own licenses.
