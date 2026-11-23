#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: CC-BY-4.0
"""Fetch + verify the FLIPS released data archives (stdlib only).

Downloads the archive(s) of a Zenodo record (or a direct archive URL, or a local directory holding the archives), checks each archive
against ``data/archives.sha256``, unpacks it (``results/`` below ``data/``, ``Productions/`` at the repository
root) and verifies **every** unpacked file's SHA-256 against ``data/manifest.sha256``. A hash mismatch aborts
and removes the files this run placed; files that are already present and valid are skipped (idempotent re-run).

URL resolution (priority): ``--url`` > ``$FLIPS_DATA_URL`` > the module constant
``ZENODO_RECORD_URL``. ``--dry-run`` / ``--check-only`` never touch the network.

Records (two archives of the same Zenodo record, each fetched at most once per run):

* ``results``     (always): ``flips_results.zip`` -> ``data/results/``: the cached results from which
  ``paper/make_paper.py`` regenerates the paper's computed figures, tables and quoted numbers (except Figure 3
  and the LLMmap bars of Figure 1, see ``paper/README.md``).
* ``generations`` (``--with-generations``): ``flips_release_data.tar.gz`` -> ``Productions/``: the
  generated answers of the 237 instances (``Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/``,
  ``answers/`` and ``token_ids/``, one parquet file per model), input of the feature grid of tiers 2
  and 3 (``make grid``), and the raw run of the LLMmap baseline (``Productions/LLMmap_ICML_run/``).

Members of the results archive are named relative to ``data/`` (``results/...``), those of the release
data archive relative to the repository root (``Productions/...``); anything of an archive that is not
listed in the manifest (its README, LICENSE, MANIFEST.tsv, SHA256SUMS) is not unpacked.

Examples
--------
# Verify what is on disk (zero network):
python scripts/fetch_data.py --check-only --manifest data/manifest.sha256

# Show what a fetch would do:
python scripts/fetch_data.py --dry-run

# Real fetch (override the record with --url or $FLIPS_DATA_URL):
python scripts/fetch_data.py

# Also fetch the generations and the LLMmap run (4.6 GB download, 6.4 GB unpacked; about 11 GB free disk while unpacking):
python scripts/fetch_data.py --with-generations

# From a local directory holding the archive(s):
python scripts/fetch_data.py --with-generations --url /path/to/dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

# --- Constants ---------------------------------------------------------------

# Zenodo record holding the archives below. Override per run with --url / $FLIPS_DATA_URL.
# Must stay the …/records/<id> form: the resolver reads the record's file list through the
# public Zenodo API (a DOI or tokened preview URL will not work here).
ZENODO_RECORD_URL = "https://zenodo.org/records/24461716"

PLACEHOLDER_HASH = "0" * 64  # externally hosted manifest entries.
DATA_DIR = "data"  # top-level dir the archives unpack into (manifest paths start with it).
CHUNK = 1 << 20  # 1 MiB streaming reads — never load a whole archive/file into RAM.


@dataclass(frozen=True)
class Record:
    """A separately fetchable archive and the manifest paths it provides."""

    name: str
    tier: str
    archive: str  # file name of the archive on the record
    prefix: str | tuple[str, ...]  # repo-relative manifest prefix(es) of the files it provides
    gated_by: str | None = None  # CLI flag that must be set to select this record.


RECORDS: tuple[Record, ...] = (
    Record(name="results", tier="analysis", archive="flips_results.zip", prefix="data/results/"),
    Record(
        name="generations",
        tier="recompute",
        archive="flips_release_data.tar.gz",
        prefix=("Productions/FLiPS_ICML_run/", "Productions/LLMmap_ICML_run/"),
        gated_by="--with-generations",
    ),
)


# --- Manifests ---------------------------------------------------------------


@dataclass
class Entry:
    sha256: str
    relpath: str  # repo-root-relative, forward slashes (manifest convention).

    @property
    def is_external(self) -> bool:
        return self.sha256 == PLACEHOLDER_HASH

    @property
    def member(self) -> str:
        """First two path components below the repo root, e.g. ``data/results``."""
        return "/".join(self.relpath.split("/")[:2])


def _parse_sums(path: Path) -> list[tuple[str, str]]:
    """Parse a sha256sum-style file, skipping ``#`` comments and blanks -> [(sha256, name)]."""
    rows: list[tuple[str, str]] = []
    for raw in path.read_text().splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # Two-space separator per sha256sum convention; tolerate extra spaces in paths.
        sha, sep, rel = line.partition("  ")
        if not sep:
            raise ValueError(f"malformed hash line (no '  ' separator): {line!r}")
        rows.append((sha.strip(), rel.strip()))
    return rows


def parse_manifest(path: Path) -> list[Entry]:
    """Parse ``data/manifest.sha256``."""
    return [Entry(sha256=sha, relpath=rel) for sha, rel in _parse_sums(path)]


def parse_archives(path: Path) -> dict[str, str]:
    """Parse ``data/archives.sha256`` -> {archive file name: sha256} (missing file: empty)."""
    if not path.is_file():
        return {}
    return {name: sha for sha, name in _parse_sums(path)}


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    return _hash_file(path, "sha256")


# --- check-only --------------------------------------------------------------


def check_only(entries: list[Entry], dest: Path) -> int:
    """Verify every present manifest file against its hash. Zero network.

    Classification per entry: EXTERNAL (placeholder hash, skipped), MISSING (not on
    disk — not yet fetched), PASS (present + hash matches), FAIL (present + differs).
    Returns process exit code: 1 iff any FAIL, else 0.
    """
    counts = {"pass": 0, "fail": 0, "missing": 0, "external": 0}
    per_member: dict[str, dict[str, int]] = {}
    failures: list[str] = []

    for e in entries:
        bucket = per_member.setdefault(e.member, {"pass": 0, "fail": 0, "missing": 0, "external": 0})
        if e.is_external:
            counts["external"] += 1
            bucket["external"] += 1
            continue
        target = dest / e.relpath
        if not target.is_file():
            counts["missing"] += 1
            bucket["missing"] += 1
            continue
        if sha256_file(target) == e.sha256:
            counts["pass"] += 1
            bucket["pass"] += 1
        else:
            counts["fail"] += 1
            bucket["fail"] += 1
            failures.append(e.relpath)

    print("check-only — verifying present files against the manifest (no network)\n")
    for member in sorted(per_member):
        b = per_member[member]
        flags = []
        if b["pass"]:
            flags.append(f"{b['pass']} pass")
        if b["fail"]:
            flags.append(f"{b['fail']} FAIL")
        if b["missing"]:
            flags.append(f"{b['missing']} missing")
        if b["external"]:
            flags.append(f"{b['external']} external")
        status = "OK" if not b["fail"] else "FAILED"
        if b["pass"] == 0 and b["fail"] == 0 and b["external"] == 0:
            status = "not fetched"
        print(f"  [{status:>11}] {member:<28} {', '.join(flags)}")

    if failures:
        print("\nHASH MISMATCHES (data is corrupt or stale):")
        for rel in failures:
            print(f"  FAIL  {rel}")

    print(
        f"\nTotals: {counts['pass']} pass, {counts['fail']} fail, "
        f"{counts['missing']} missing (not fetched), {counts['external']} external (skipped)."
    )
    if counts["fail"]:
        print("RESULT: FAIL — at least one present file does not match its manifest hash.")
        return 1
    print("RESULT: OK — every present file matches its manifest hash.")
    return 0


# --- Record selection / planning --------------------------------------------


def selected_records(with_generations: bool) -> list[Record]:
    flags = {"--with-generations": with_generations}
    return [rec for rec in RECORDS if rec.gated_by is None or flags.get(rec.gated_by, False)]


def entries_for_record(entries: list[Entry], rec: Record) -> list[Entry]:
    """Manifest entries a record provides (externally hosted rows excluded)."""
    return [e for e in entries if e.relpath.startswith(rec.prefix) and not e.is_external]


def _prefixes(rec: Record) -> str:
    return ", ".join(rec.prefix) if isinstance(rec.prefix, tuple) else rec.prefix


def is_placeholder(url: str) -> bool:
    return "PLACEHOLDER" in url


def dry_run(entries: list[Entry], records: list[Record], url: str, dest: Path) -> int:
    print("dry-run — fetch plan (no download, no writes)\n")
    print(f"  source URL : {url}")
    if is_placeholder(url):
        print("               ^ PLACEHOLDER — a real fetch will refuse. Set --url, $FLIPS_DATA_URL,")
        print("                 or edit ZENODO_RECORD_URL (see data/README.md).")
    print(f"  manifest   : {len(entries)} entries\n")

    for rec in records:
        print(f"  record '{rec.name}' [{rec.tier}] archive {rec.archive}"
              + (f"  (selected by {rec.gated_by})" if rec.gated_by else "  (always)"))
        rec_entries = entries_for_record(entries, rec)
        if not rec_entries:
            print(f"      (no manifest rows under {_prefixes(rec)}: a fetch refuses)")
            continue
        present = sum(1 for e in rec_entries if (dest / e.relpath).is_file())
        print(f"      {len(rec_entries)} files below {_prefixes(rec)}, {present} already present.")
    print("\nWould verify the archive and each unpacked file's SHA-256 and remove this run's files on mismatch.")
    return 0


# --- Download + unpack -------------------------------------------------------


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _safe_extract(archive: Path, into: Path) -> None:
    """Extract a tar/zip archive, refusing any member that escapes *into* (zip-slip)."""
    into.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for n in zf.namelist():
                if not _is_within(into, into / n):
                    raise RuntimeError(f"refusing path-traversal archive member: {n!r}")
            zf.extractall(into)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                if not _is_within(into, into / m.name):
                    raise RuntimeError(f"refusing path-traversal archive member: {m.name!r}")
            # filter="data": regular files and directories only, no absolute or escaping links, no special modes
            # (``data_filter`` exists from Python 3.11.4; older 3.11 patch releases have only the checks above).
            tf.extractall(into, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    else:
        raise RuntimeError(f"unrecognised archive format (not tar/zip): {archive}")


def resolve_archives(url: str) -> dict[str, tuple[str, str | None]]:
    """Resolve the data URL to ``{archive file name: (download URL, md5 or None)}``.

    A Zenodo ``…/records/<id>`` web URL is resolved through the public records API
    (``…/api/records/<id>``): every ``files[]`` entry carries its ``key`` (file name),
    ``links.self`` (direct URL) and, when present, ``checksum`` (``md5:<hex>``). A local directory
    (a path or a ``file://`` URL) provides every archive file it holds. Any other URL is taken as
    one direct archive and named after the last path component.
    """
    local = Path(unquote(urlparse(url).path)) if url.startswith("file://") else Path(url)
    if "://" not in url or url.startswith("file://"):
        if local.is_dir():
            return {f.name: (f.resolve().as_uri(), None) for f in sorted(local.iterdir())
                    if f.is_file() and f.name.endswith((".zip", ".tar.gz"))}
        if "://" not in url:
            url = local.resolve().as_uri()
    if "zenodo.org/records/" not in url:
        name = unquote(Path(urlparse(url).path).name)
        return {name: (url, None)}
    rid = url.rstrip("/").rsplit("/", 1)[-1]
    api = f"https://zenodo.org/api/records/{rid}"
    with urllib.request.urlopen(api) as resp:  # noqa: S310 — https Zenodo API only.
        meta = json.load(resp)
    out: dict[str, tuple[str, str | None]] = {}
    for f in meta.get("files", []):
        link = f.get("links", {}).get("self")
        if not link:
            continue
        key = f.get("key") or unquote(link.rstrip("/").split("/")[-2])
        chk = f.get("checksum", "")
        out[key] = (link, chk.split(":", 1)[1] if chk.startswith("md5:") else None)
    if not out:
        raise RuntimeError(f"Zenodo record {rid} exposes no downloadable files")
    return out


def _download(url: str, dest_file: Path) -> None:
    with urllib.request.urlopen(url) as resp, open(dest_file, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, CHUNK)


def _place_and_verify(
    src_root: Path, rec: Record, rec_entries: list[Entry], dest: Path, placed: list[Path]
) -> None:
    """Move a record's files from the extracted archive into the repo, verifying each one.

    Archive members are named relative to ``data/``. Raises on any hash mismatch (the caller
    removes ``placed``). Idempotent: a destination file already present and hash-valid is
    left untouched.
    """
    strip = DATA_DIR + "/"
    for e in rec_entries:
        target = dest / e.relpath
        if target.is_file() and sha256_file(target) == e.sha256:
            print(f"      skip (present+valid): {e.relpath}")
            continue
        source = src_root / (e.relpath[len(strip):] if e.relpath.startswith(strip) else e.relpath)
        if not source.is_file():
            raise RuntimeError(f"archive '{rec.archive}' misses expected file: {e.relpath}")
        digest = sha256_file(source)
        if digest != e.sha256:
            raise RuntimeError(f"SHA-256 mismatch for {e.relpath}: expected {e.sha256}, got {digest}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, target)
        placed.append(target)


def fetch(entries: list[Entry], records: list[Record], archives: dict[str, str], url: str, dest: Path) -> int:
    if is_placeholder(url):
        print(
            "error: the data URL is a PLACEHOLDER. Refusing to download.\n"
            "       Set --url / $FLIPS_DATA_URL to the Zenodo record of the release, or edit\n"
            "       ZENODO_RECORD_URL in scripts/fetch_data.py. See data/README.md.",
            file=sys.stderr,
        )
        return 2

    todo = [(rec, entries_for_record(entries, rec)) for rec in records]
    empty = [rec for rec, ents in todo if not ents]
    if empty:
        names = ", ".join(f"'{r.name}' ({_prefixes(r)})" for r in empty)
        print(f"error: no manifest rows for record {names}: nothing to fetch.", file=sys.stderr)
        return 2

    available = resolve_archives(url)
    print(f"resolved {len(available)} archive(s) from {url}")

    # Unpack next to the destination (same file system: files are moved, not copied, and the
    # release data archive does not need room in the system temp directory).
    (dest / DATA_DIR).mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".fetch_", dir=dest / DATA_DIR))
    placed: list[Path] = []
    done: dict[str, Path] = {}  # archive name -> unpacked dir: an archive is downloaded at most once per run
    try:
        for rec, rec_entries in todo:
            print(f"record '{rec.name}' [{rec.tier}] <- {rec.archive}")
            if rec.archive not in done:
                if rec.archive not in available:
                    raise RuntimeError(f"archive '{rec.archive}' is not on the record (has: {', '.join(sorted(available))})")
                au, md5 = available[rec.archive]
                archive = tmp / rec.archive
                print(f"  downloading {au}")
                _download(au, archive)
                expected = archives.get(rec.archive)
                if expected and expected != PLACEHOLDER_HASH:
                    got = sha256_file(archive)
                    if got != expected:
                        raise RuntimeError(f"archive SHA-256 mismatch for {rec.archive}: expected {expected}, got {got}")
                    print("  archive sha256 ok")
                else:
                    print(f"  warning: no archive hash for {rec.archive} in data/archives.sha256; per-file hashes still apply")
                if md5 and _hash_file(archive, "md5") != md5:
                    raise RuntimeError(f"archive md5 mismatch for {rec.archive} (record says {md5})")
                extract_dir = tmp / f"{rec.name}_unpacked"
                _safe_extract(archive, extract_dir)
                archive.unlink(missing_ok=True)
                done[rec.archive] = extract_dir
            _place_and_verify(done[rec.archive], rec, rec_entries, dest, placed)
            print(f"  placed/verified {len(rec_entries)} files")
    except Exception as exc:  # noqa: BLE001 — abort + remove what this run placed.
        print(f"\nFETCH ABORTED: {exc}", file=sys.stderr)
        for f in placed:
            f.unlink(missing_ok=True)
        if placed:
            print(f"  removed {len(placed)} file(s) placed by this run", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nfetch complete — all unpacked files verified against the manifest.")
    return 0


# --- CLI ---------------------------------------------------------------------


def resolve_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url
    env = os.environ.get("FLIPS_DATA_URL")
    if env:
        return env
    return ZENODO_RECORD_URL


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=None,
                    help="Zenodo record URL, direct archive URL or local directory of archives (overrides env/constant).")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.sha256"),
                    help="SHA-256 manifest to verify against (default: data/manifest.sha256).")
    ap.add_argument("--archives", type=Path, default=None,
                    help="Archive hash list (default: archives.sha256 next to the manifest).")
    ap.add_argument("--dest", type=Path, default=Path("."),
                    help="Repository root to unpack/verify into (default: current dir).")
    ap.add_argument("--dry-run", action="store_true", help="Print the fetch plan; no download, no writes.")
    ap.add_argument("--check-only", action="store_true",
                    help="Verify present files against the manifest; no network.")
    ap.add_argument("--with-generations", action="store_true",
                    help="Also fetch the release data archive: the generations (input of tiers 2 and 3) "
                         "and the LLMmap run, into Productions/.")
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    entries = parse_manifest(args.manifest)

    if args.check_only:
        return check_only(entries, args.dest)

    records = selected_records(args.with_generations)
    url = resolve_url(args.url)

    if args.dry_run:
        return dry_run(entries, records, url, args.dest)

    archives = parse_archives(args.archives or args.manifest.with_name("archives.sha256"))
    return fetch(entries, records, archives, url, args.dest)


if __name__ == "__main__":
    raise SystemExit(main())
