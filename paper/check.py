#!/usr/bin/env python3
"""check.py - compare the output of make_paper.py with paper/expected/.

    python paper/check.py [--out DIR] [--expected DIR] [--rtol 1e-9] [--update]

Compared (every file of expected/, and every macro file of the output):
  * macro files (numbers.tex and the figures/*.tex that define \\newcommand macros): macro by macro; a macro that is missing, extra or has a different
    value is a failure, and so is an output macro file that expected/ does not have;
  * the other text files (the token-pair table, figures/*.txt): byte-equal;
  * JSON sidecars (closed/numbers.json, figures/**/*.json): equal as parsed JSON, floats within --rtol (default 1e-9: same-platform runs are bit-equal).
The figures themselves (PDF, PNG) are not compared: their bytes depend on the matplotlib / freetype build; their plotted data are the JSON sidecars.
Exit 0 iff every compared file matches. `--update` rewrites expected/ from the output (maintainers; the expected files are the paper's numbers).
"""
import argparse, json, math, os, re, shutil, sys

PKG = os.path.dirname(os.path.abspath(__file__))
EXPECTED = os.path.join(PKG, "expected")
NEWCOMMAND = re.compile(r"\\newcommand\{\\(\w+)\}\{")


def out_root():
    return os.environ.get("FLIPS_OUT") or os.path.join(PKG, "out")


def _walk(base):
    for dp, _, fns in os.walk(base):
        for fn in sorted(fns):
            yield os.path.relpath(os.path.join(dp, fn), base).replace(os.sep, "/")


def macros(text):
    """[(name, value)] of the \\newcommand{\\name}{value} definitions of a TeX text (brace matching; comments outside the values ignored)"""
    text = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in text.split("\n"))
    out = []
    for m in NEWCOMMAND.finditer(text):
        i, depth = m.end(), 1
        while depth and i < len(text):
            depth += (text[i] == "{") - (text[i] == "}"); i += 1
        out.append((m.group(1), text[m.end():i - 1]))
    return out


def macro_diff(o, e):
    """failures of the macros of output text o against expected text e"""
    mo, me = macros(o), macros(e)
    fails = [f"duplicate macro \\{n} in {w}" for w, m in (("output", mo), ("expected", me)) for n in sorted({n for n, _ in m if [x for x, _ in m].count(n) > 1})]
    do, de = dict(mo), dict(me)
    fails += [f"missing macro \\{n}" for n in sorted(set(de) - set(do))]
    fails += [f"extra macro \\{n}" for n in sorted(set(do) - set(de))]
    fails += [f"macro \\{n}: {do[n]!r} vs expected {de[n]!r}" for n in sorted(set(do) & set(de)) if do[n] != de[n]]
    return fails


def json_diff(a, b, path="", rtol=1e-9, out=None):
    """paths where parsed JSON a and b differ (floats within rtol; NaN == NaN)"""
    out = [] if out is None else out
    if len(out) >= 20:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'expected' if k in b else 'output'}")
            else:
                json_diff(a[k], b[k], f"{path}/{k}", rtol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                json_diff(x, y, f"{path}[{i}]", rtol, out)
    elif isinstance(a, bool) or isinstance(b, bool) or not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if type(a) is not type(b) and (isinstance(a, bool) or isinstance(b, bool)) or a != b:
            out.append(f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}")
    elif not (a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
              or abs(a - b) <= rtol * max(abs(a), abs(b))):
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def _read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def compare(out, expected, rtol=1e-9):
    """-> (failures [str], n_text byte-equal files, n_macros compared, n_json sidecars)"""
    fails, n_text, n_mac, n_json = [], 0, 0, 0
    exp_files = set(_walk(expected))
    for rel in sorted(exp_files):
        e, o = os.path.join(expected, rel), os.path.join(out, rel)
        if not os.path.exists(o):
            fails.append(f"MISSING output {rel}")
        elif rel.endswith(".json"):
            n_json += 1
            d = json_diff(json.load(open(o)), json.load(open(e)), rtol=rtol)
            if d:
                fails.append(f"JSON differs {rel}: " + "; ".join(d[:4]))
        elif rel.endswith(".tex") and macros(_read(e)):
            n_mac += len(macros(_read(e)))
            fails += [f"MACROS {rel}: {d}" for d in macro_diff(_read(o), _read(e))]
        else:
            n_text += 1
            if open(o, "rb").read() != open(e, "rb").read():
                fails.append(f"BYTES differ {rel}")
    if os.path.isdir(out):
        for rel in sorted(_walk(out)):
            if rel.endswith(".tex") and rel not in exp_files and macros(_read(os.path.join(out, rel))):
                fails.append(f"EXTRA macro file {rel} (not in expected/): " + ", ".join(f"\\{n}" for n, _ in macros(_read(os.path.join(out, rel)))))
    return fails, n_text, n_mac, n_json


def update(out, expected):
    """make expected/ the checked files of the output (numbers.tex, closed/ and figures/ json|tex|txt); files of expected/ that the output does not have are removed"""
    keep = [rel for rel in sorted(_walk(out))
            if rel == "numbers.tex" or (rel.endswith((".json", ".tex", ".txt")) and rel.split("/")[0] in ("closed", "figures"))]
    if os.path.isdir(expected):
        for rel in sorted(_walk(expected)):
            if rel not in keep:
                os.remove(os.path.join(expected, rel))
    for rel in keep:
        dst = os.path.join(expected, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(out, rel), dst)
    return len(keep)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output root of make_paper.py (default: $FLIPS_OUT or paper/out)")
    ap.add_argument("--expected", default=EXPECTED)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--update", action="store_true", help="rewrite expected/ from the output (maintainers)")
    a = ap.parse_args(argv)
    out = a.out or out_root()
    if not os.path.isdir(out):
        print(f"no output at {out}: run paper/make_paper.py first", file=sys.stderr)
        return 2
    if a.update:
        print(f"expected/ updated: {update(out, a.expected)} files")
        return 0
    fails, n_text, n_mac, n_json = compare(out, a.expected, a.rtol)
    for f in fails:
        print("FAIL", f)
    print(f"{n_mac} macros checked one by one, {n_text} text files byte-equal checked, {n_json} JSON sidecars checked, {len(fails)} failure(s)")
    print("RESULT:", "all equal" if not fails else "DIFFERENCES")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
