"""absorption.py - where the method's class decision sends the Unseen test batches, N_t 32, 155 cells. Descriptive, no tuning.
For every Unseen test batch (source u, class decision p = a Known instance of the cell, taken regardless of the detector flag):
  (i)  share with attribute distance d(u, p) = 1 (instances differ in exactly one of base LLM / quantisation / temperature / system prompt; parser dataio.attrs; an abliterated
       checkpoint is its own base LLM) vs the share expected if p were a uniformly drawn Known instance of that cell (|{k in Known_cell : d(u, k) = 1}| / |Known_cell|,
       averaged over the same batches); full distance distribution d = 1..4 too;
  (ii) which attribute differs, among the d = 1 decisions (and the mix expected under the uniform draw);
  (iii) share flagged Unseen by the pre-test 50 %-target cut among batches decided at d = 1 vs d >= 2.
Uncertainty: pair-clustered percentile bootstrap (31 token pairs resampled, B = 2000, seed 0) of the batch-pooled ratios.
Input: results/open_set/per_instance/test_rows.npz (N_t 32, all batches); output <out>/absorption.json.
  python -m paper.stats.absorption [--out DIR]"""
import argparse, json, os
import numpy as np
from ..dataio import R, attrs, fig_dir, r4

ATT = ("base LLM", "quantisation", "temperature", "system prompt"); B, SEED = 2000, 0


def analyse(cell, src, pred, flag, known, A, pair_of_cell):
    D = (A[src] != A[pred]).sum(1)                                                                  # attribute distance of each Unseen batch's decision
    Dall = (A[:, None, :] != A[None, :, :]).sum(2)                                                  # (237, 237)
    KD = np.stack([(known[cell] & (Dall[src] == d)).sum(1) for d in range(5)], 1) / known[cell].sum(1, keepdims=True)    # expected share per distance, per batch
    one = D == 1; diffat = np.argmax(A[src] != A[pred], 1)
    expatt = np.stack([(known[cell] & (Dall[src] == 1) & (A[src][:, None, a] != A[None, :, a])).sum(1) for a in range(4)], 1) / known[cell].sum(1, keepdims=True)
    pair = pair_of_cell[cell]; P = np.unique(pair); pix = np.searchsorted(P, pair)
    def stat(w):                                                                                    # w = per-pair multiplicities (len P) -> dict of ratios
        m = w[pix]; tot = m.sum(); o1 = (m * one).sum()
        r = dict(obs_d1=o1 / tot, exp_d1=(m * KD[:, 1]).sum() / tot); r["obs_minus_exp_d1"] = r["obs_d1"] - r["exp_d1"]
        r["flag_d1"] = (m * (one & flag)).sum() / max(o1, 1e-12); r["flag_other"] = (m * (~one & flag)).sum() / max((m * ~one).sum(), 1e-12); r["flag_d1_minus_other"] = r["flag_d1"] - r["flag_other"]
        return r
    est = stat(np.ones(len(P))); rng = np.random.default_rng(SEED); bs = [stat(np.bincount(rng.integers(0, len(P), len(P)), minlength=len(P)).astype(float)) for _ in range(B)]
    ci = {k: [r4(np.percentile([b[k] for b in bs], 2.5)), r4(np.percentile([b[k] for b in bs], 97.5))] for k in est}
    out = dict(n_unseen_batches=int(len(src)), n_cells=int(len(np.unique(cell))), n_pairs=int(len(P)),
               i_distance1=dict(observed=r4(est["obs_d1"]), expected_uniform_known=r4(est["exp_d1"]), observed_minus_expected=r4(est["obs_minus_exp_d1"]),
                                ci95=dict(observed=ci["obs_d1"], expected=ci["exp_d1"], observed_minus_expected=ci["obs_minus_exp_d1"]),
                                distance_distribution={str(d): dict(observed=r4((D == d).mean()), expected=r4(KD[:, d].mean())) for d in range(1, 5)}),
               ii_attribute_at_distance1=dict(n_batches_d1=int(one.sum()),
                                              observed_share={ATT[a]: r4((diffat[one] == a).mean()) for a in range(4)},
                                              expected_share_uniform={ATT[a]: r4(expatt[:, a].sum() / expatt.sum()) for a in range(4)},
                                              observed_over_unseen_batches={ATT[a]: r4((one & (diffat == a)).mean()) for a in range(4)}),
               iii_flagged_c50=dict(d1=r4(est["flag_d1"]), d_ge2=r4(est["flag_other"]), d1_minus_dge2=r4(est["flag_d1_minus_other"]),
                                    ci95=dict(d1=ci["flag_d1"], d_ge2=ci["flag_other"], d1_minus_dge2=ci["flag_d1_minus_other"]), overall=r4(flag.mean())))
    out["iii_flagged_c50_by_attribute"] = {ATT[a]: dict(n=int((one & (diffat == a)).sum()), flagged=r4(flag[one & (diffat == a)].mean()) if (one & (diffat == a)).any() else None) for a in range(4)}
    return out, D, one, diffat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--out", default=None, help="output directory (default: <FLIPS_OUT>/figures)"); a = ap.parse_args(argv)
    OUT = a.out or fig_dir(); os.makedirs(OUT, exist_ok=True); DATA = R("open_set", "per_instance")
    T = np.load(os.path.join(DATA, "test_rows.npz")); names = [str(x) for x in T["names"]]
    assert len(names) == 237
    A = np.array([attrs(n) for n in names]); known = T["known"].astype(bool); assert known.shape == (155, 237) and (known.sum(1) == 214).all()
    pair_of_cell = np.array([str(c).split("_s")[0] for c in T["cells"]])
    y, cell, s = T["y"].astype(int), T["cell"].astype(int), T["s"]; U = y == 1
    c32, src32, pred32 = cell[U], T["src"][U].astype(int), T["pred"][U].astype(int); f32 = s[U] >= T["tau50"][c32]
    assert not known[c32, src32].any() and known[c32, pred32].all(), "Unseen source must be outside, decision inside the cell's Known set"
    r32, dist32, one32, att32 = analyse(c32, src32, pred32, f32, known, A, pair_of_cell)
    ab = np.array(["abliterated" in n.lower() for n in names]); abl_pairs = int((one32 & (att32 == 0) & (ab[src32] | ab[pred32])).sum())
    out = dict(description="Unseen test batches of the method, 155 cells: attribute distance of the class decision (taken regardless of the detector flag) to the Unseen source; (i) d = 1 share vs uniform-Known expectation, (ii) differing attribute, (iii) flagged share at the 50 %-target cut by d.", attribute_parser="paper/dataio.py attrs(name) -> (model, quant, T, SP); quant '-' = none; abliterated checkpoint = distinct base LLM",
               nt32=r32,
               notes=dict(nt32_base_llm_d1_involving_abliterated=abl_pairs,
                          known_per_cell=214, unseen_per_cell=23, operating_point="flag = stored s_test >= stored tau50 (pre-test 50 %-target cut)",
                          weighting="every Unseen test batch weight 1, pooled over the 155 cells; CIs resample the 31 token pairs"),
               sources=dict(nt32="results/open_set/per_instance/test_rows.npz: class decision `pred`, test scores `s` and pre-test cut `tau50` of every cell (N_t 32)"))
    path = os.path.join(OUT, "absorption.json"); json.dump(out, open(path, "w"), indent=1); print(f"[ok] {path}"); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
