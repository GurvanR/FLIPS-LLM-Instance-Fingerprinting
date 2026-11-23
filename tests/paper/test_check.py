"""Fast tests of paper/check.py on a tiny sample (no data, no figures)."""
import json
import os

from paper import check


def _write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def _sample(root, nums="% generated\n\\newcommand{\\nA}{1.0\\%}\n\\newcommand{\\nB}{2}\n", val=0.5):
    _write(root, "numbers.tex", nums)
    _write(root, "closed/numbers.json", json.dumps({"fig15": {"mean": val, "n": 3}, "xs": [1, 2, 3]}))
    _write(root, "figures/attr_numbers.tex", "\\newcommand{\\fgAttrNt}{32}\n")
    _write(root, "figures/table3_pairs.tex", "\\begin{tabular}{lr}\n\\end{tabular}\n")
    _write(root, "figures/fig05_accuracy_vs_nt.png", "png-bytes")


def test_macros_parse_nested_braces_and_skip_comments():
    text = "% \\newcommand{\\nX}{0}\n\\newcommand{\\nA}{95\\% interval {a} to {b}}   % note\n\\newcommand{\\nB}{x}\n"
    assert check.macros(text) == [("nA", "95\\% interval {a} to {b}"), ("nB", "x")]


def test_json_diff_tolerance_and_structure():
    assert check.json_diff({"a": [1, 2.0]}, {"a": [1, 2.0 + 1e-12]}) == []
    assert check.json_diff({"a": 1.0}, {"a": 1.1})
    assert check.json_diff({"a": 1}, {"b": 1})
    assert check.json_diff({"a": [1, 2]}, {"a": [1, 2, 3]})
    assert check.json_diff({"a": float("nan")}, {"a": float("nan")}) == []
    assert check.json_diff({"a": True}, {"a": 1})  # bool is not a number here


def test_compare_equal_and_update_roundtrip(tmp_path):
    out, exp = str(tmp_path / "out"), str(tmp_path / "expected")
    _sample(out)
    _write(exp, "figures/stale.json", "{}")
    assert check.update(out, exp) == 4                           # the PNG is not a reference file
    assert not os.path.exists(os.path.join(exp, "figures", "stale.json")) and not os.path.exists(os.path.join(exp, "figures", "fig05_accuracy_vs_nt.png"))
    fails, n_text, n_mac, n_json = check.compare(out, exp)
    assert fails == [] and (n_text, n_mac, n_json) == (1, 3, 1)


def test_comments_and_order_of_macro_files_do_not_matter(tmp_path):
    out, exp = str(tmp_path / "out"), str(tmp_path / "expected")
    _sample(exp)
    _sample(out, nums="\\newcommand{\\nB}{2}\n\\newcommand{\\nA}{1.0\\%}   % other comment\n")
    assert check.compare(out, exp)[0] == []


def test_compare_detects_missing_extra_and_different_macros(tmp_path):
    out, exp = str(tmp_path / "out"), str(tmp_path / "expected")
    _sample(exp)
    _sample(out, nums="\\newcommand{\\nA}{1.1\\%}\n\\newcommand{\\nC}{3}\n\\newcommand{\\nC}{3}\n", val=0.6)
    _write(out, "figures/other_numbers.tex", "\\newcommand{\\nD}{4}\n")
    fails, *_ = check.compare(out, exp)
    assert "MACROS numbers.tex: missing macro \\nB" in fails
    assert "MACROS numbers.tex: extra macro \\nC" in fails
    assert "MACROS numbers.tex: duplicate macro \\nC in output" in fails
    assert any(f.startswith("MACROS numbers.tex: macro \\nA:") for f in fails)
    assert any(f.startswith("EXTRA macro file figures/other_numbers.tex") for f in fails)
    assert any("JSON differs closed/numbers.json" in f for f in fails)


def test_missing_output_and_text_bytes(tmp_path):
    out, exp = str(tmp_path / "out"), str(tmp_path / "expected")
    _sample(out)
    check.update(out, exp)
    _write(out, "figures/fig05_accuracy_vs_nt.png", "other-bytes")        # figures are not compared
    _write(out, "figures/table3_pairs.tex", "\\begin{tabular}{ll}\n\\end{tabular}\n")
    os.remove(os.path.join(out, "figures", "attr_numbers.tex"))
    fails, *_ = check.compare(out, exp)
    assert fails == ["MISSING output figures/attr_numbers.tex", "BYTES differ figures/table3_pairs.tex"]


def test_main_exit_codes(tmp_path, capsys):
    out, exp = str(tmp_path / "out"), str(tmp_path / "expected")
    _sample(out)
    assert check.main(["--out", out, "--expected", exp, "--update"]) == 0
    assert check.main(["--out", out, "--expected", exp]) == 0
    _write(out, "numbers.tex", "changed\n")
    assert check.main(["--out", out, "--expected", exp]) == 1
    assert check.main(["--out", str(tmp_path / "nowhere"), "--expected", exp]) == 2
    capsys.readouterr()
