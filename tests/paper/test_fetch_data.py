"""Offline tests of scripts/fetch_data.py: file:// archives, hashes, refusal paths, single download per archive,
results by default and generations on request from one record (local directory or a mocked Zenodo record)."""
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import zipfile

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "fetch_data.py")
spec = importlib.util.spec_from_file_location("fetch_data", SCRIPT)
fd = importlib.util.module_from_spec(spec)
sys.modules["fetch_data"] = fd          # dataclasses resolve string annotations through sys.modules
spec.loader.exec_module(fd)

FILES = {"results/closed_set/a.json": b'{"a": 1}\n', "results/open_set/b.csv": b"x,y\n1,2\n"}
GEN_FILES = {"Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/answers/org__model.parquet": b"answers",
             "Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/token_ids/org__model.parquet": b"token ids",
             "Productions/LLMmap_ICML_run/LLMmap_classification/a/model.pt": b"weights"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def make_release(tmp_path, with_readme=True):
    """a miniature archive + manifest + archive-hash list in a fake repo root"""
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    zpath = tmp_path / "flips_results.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for rel, b in FILES.items():
            zf.writestr(rel, b)
        if with_readme:
            zf.writestr("README.md", "not part of the manifest\n")
    manifest = "# test manifest\n" + "".join(f"{sha(b)}  data/{rel}\n" for rel, b in sorted(FILES.items()))
    (root / "data" / "manifest.sha256").write_text(manifest)
    (root / "data" / "archives.sha256").write_text(f"{sha(zpath.read_bytes())}  flips_results.zip\n")
    return root, zpath


def add_generations(tmp_path, root):
    """a miniature release data tar.gz beside the results zip, with its manifest rows and archive hash"""
    tpath = tmp_path / "flips_release_data.tar.gz"
    with tarfile.open(tpath, "w:gz") as tf:
        for rel, b in {**GEN_FILES, "README.md": b"not part of the manifest\n"}.items():
            ti = tarfile.TarInfo(rel)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    with open(root / "data" / "manifest.sha256", "a") as f:
        f.write("".join(f"{sha(b)}  {rel}\n" for rel, b in sorted(GEN_FILES.items())))
    with open(root / "data" / "archives.sha256", "a") as f:
        f.write(f"{sha(tpath.read_bytes())}  flips_release_data.tar.gz\n")
    return tpath


def run(root, *args):
    return fd.main(["--manifest", str(root / "data" / "manifest.sha256"), "--dest", str(root), *args])


def test_parse_manifest_and_archives(tmp_path):
    root, zpath = make_release(tmp_path)
    ents = fd.parse_manifest(root / "data" / "manifest.sha256")
    assert [e.relpath for e in ents] == ["data/results/closed_set/a.json", "data/results/open_set/b.csv"]
    assert fd.parse_archives(root / "data" / "archives.sha256") == {"flips_results.zip": sha(zpath.read_bytes())}
    assert fd.parse_archives(root / "data" / "nope") == {}


def test_placeholder_refuses_real_fetch_but_not_dry_run(tmp_path, capsys, monkeypatch):
    root, _ = make_release(tmp_path)
    # the shipped constant may hold the published record id: test the refusal on a placeholder URL
    monkeypatch.setattr(fd, "ZENODO_RECORD_URL", "https://zenodo.org/records/PLACEHOLDER")
    monkeypatch.delenv("FLIPS_DATA_URL", raising=False)
    assert run(root) == 2                                   # real fetch against the placeholder: refused
    assert run(root, "--dry-run") == 0
    assert "PLACEHOLDER" in capsys.readouterr().out
    assert not (root / "data" / "results").exists()


def test_fetch_from_file_url_places_only_manifest_files(tmp_path):
    root, zpath = make_release(tmp_path)
    assert run(root, "--url", zpath.as_uri()) == 0
    assert (root / "data/results/closed_set/a.json").read_bytes() == FILES["results/closed_set/a.json"]
    assert not (root / "data" / "README.md").exists()       # archive docs are not unpacked
    assert run(root, "--check-only") == 0
    assert run(root, "--url", zpath.as_uri()) == 0           # idempotent


def test_check_only_detects_corruption_and_missing_is_not_failure(tmp_path):
    root, zpath = make_release(tmp_path)
    assert run(root, "--check-only") == 0                    # nothing fetched yet: missing, not failed
    assert run(root, "--url", zpath.as_uri()) == 0
    (root / "data/results/open_set/b.csv").write_bytes(b"corrupt")
    assert run(root, "--check-only") == 1


def test_bad_archive_hash_aborts_and_removes_what_it_placed(tmp_path):
    root, zpath = make_release(tmp_path)
    (root / "data" / "archives.sha256").write_text(f"{'1' * 64}  flips_results.zip\n")
    assert run(root, "--url", zpath.as_uri()) == 1
    assert not any((root / "data" / "results").rglob("*.*")) if (root / "data" / "results").exists() else True


def test_bad_member_hash_rolls_back(tmp_path):
    root, zpath = make_release(tmp_path)
    lines = (root / "data" / "manifest.sha256").read_text().splitlines()
    lines[-1] = f"{'2' * 64}  data/results/open_set/b.csv"      # the second file cannot verify
    (root / "data" / "manifest.sha256").write_text("\n".join(lines) + "\n")
    assert run(root, "--url", zpath.as_uri()) == 1
    assert not (root / "data/results/closed_set/a.json").exists()    # the first, already placed, is removed again


def test_one_download_per_archive(tmp_path, monkeypatch):
    root, zpath = make_release(tmp_path)
    calls = []
    real = fd._download
    monkeypatch.setattr(fd, "_download", lambda url, dest: (calls.append(url), real(url, dest))[1])
    assert run(root, "--url", zpath.as_uri()) == 0
    assert calls == [zpath.as_uri()]


def test_generations_refused_without_manifest_rows(tmp_path):
    root, zpath = make_release(tmp_path)
    assert run(root, "--url", zpath.as_uri(), "--with-generations") == 2


def test_local_directory_record_results_by_default_generations_on_request(tmp_path):
    root, zpath = make_release(tmp_path)
    add_generations(tmp_path, root)
    assert run(root, "--url", str(tmp_path)) == 0                       # default: results only
    assert (root / "data/results/open_set/b.csv").is_file()
    assert not (root / "Productions").exists()
    assert run(root, "--url", tmp_path.as_uri(), "--with-generations") == 0
    for rel, b in GEN_FILES.items():
        assert (root / rel).read_bytes() == b
    assert not (root / "README.md").exists() and not (root / "data" / "README.md").exists()
    assert sorted(p.name for p in (root / "data").iterdir()) == ["archives.sha256", "manifest.sha256", "results"]
    assert run(root, "--check-only") == 0


def test_zenodo_record_with_two_files(tmp_path, monkeypatch):
    """one mocked Zenodo record holding both archives: the API lists them, each is downloaded once and md5-checked"""
    root, zpath = make_release(tmp_path)
    tpath = add_generations(tmp_path, root)
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
    listing = {"files": [{"key": p.name, "links": {"self": p.as_uri()}, "checksum": f"md5:{md5(p)}"} for p in (zpath, tpath)]}
    real_urlopen, calls = fd.urllib.request.urlopen, []

    def urlopen(url, *a, **k):
        calls.append(url)
        if url == "https://zenodo.org/api/records/4242":
            return io.BytesIO(json.dumps(listing).encode())
        return real_urlopen(url, *a, **k)

    monkeypatch.setattr(fd.urllib.request, "urlopen", urlopen)
    assert run(root, "--url", "https://zenodo.org/records/4242", "--with-generations") == 0
    assert calls == ["https://zenodo.org/api/records/4242", zpath.as_uri(), tpath.as_uri()]
    assert (root / "Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/token_ids/org__model.parquet").read_bytes() == b"token ids"
    listing["files"][1]["checksum"] = "md5:" + "0" * 32                  # record and download disagree: abort, roll back
    for rel in GEN_FILES:
        (root / rel).unlink()
    assert run(root, "--url", "https://zenodo.org/records/4242", "--with-generations") == 1
    assert not any(p.is_file() for p in (root / "Productions").rglob("*"))


def test_record_table():
    assert [r.archive for r in fd.RECORDS] == ["flips_results.zip", "flips_release_data.tar.gz"]
    assert fd.RECORDS[0].gated_by is None and fd.RECORDS[1].gated_by == "--with-generations"
    assert [r.name for r in fd.selected_records(False)] == ["results"]
    assert [r.name for r in fd.selected_records(True)] == ["results", "generations"]


def test_shipped_manifest_lists_both_archives():
    """the repository's data/ manifests carry the rows and archive hashes of both archives"""
    data = os.path.join(os.path.dirname(SCRIPT), "..", "data")
    ents = fd.parse_manifest(fd.Path(data) / "manifest.sha256")
    arcs = fd.parse_archives(fd.Path(data) / "archives.sha256")
    assert set(arcs) == {"flips_results.zip", "flips_release_data.tar.gz"}
    gen = [e.relpath for e in ents if e.relpath.startswith("Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/")]
    assert len(gen) == 76 and {p.split("/")[4] for p in gen} == {"answers", "token_ids"}
    assert sum(e.relpath.startswith("Productions/LLMmap_ICML_run/") for e in ents) == 600


@pytest.mark.parametrize("name", ["../evil.json", "/abs/evil.json"])
def test_zip_slip_refused(tmp_path, name):
    zpath = tmp_path / "bad.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(name, "x")
    with pytest.raises(RuntimeError):
        fd._safe_extract(zpath, tmp_path / "into")
