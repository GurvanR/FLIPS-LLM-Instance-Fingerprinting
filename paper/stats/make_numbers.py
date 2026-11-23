"""make_numbers.py -> <out>/numbers.tex : the \\n... macros of the paper text, each read from a result file of the earlier steps; no hand-typed result.
Sources: results/closed_set/hierarchical/summary.json (closed set), <out>/figures/{tab_openset_op,fig06_unseen_pr,attr_miss,absorption,fig04_zero_one_vs_pairs,
fig12_closed_heatmap,fig10_feature_importance}.json, <out>/closed/numbers.json (showcase, closed-set precision / recall, pair accuracies),
results/closed_set/truncation/summary.json, and the stored LLMmap-IF curve (stats/llmmap_curve.py).
The other macros of the paper are written by figures/attribute_errors.py (attr_numbers.tex), figures/distance2.py (distance2_numbers.tex) and figures/corpus.py (discard_macros.tex);
constants/attr_caption.tex holds the hand-written caption of the attribute figure."""
import argparse, json, os
from ..dataio import R, CLOSED, TRUNC, closed_dir, fig_dir, out_dir
from .llmmap_curve import curve as llmmap_curve

W = {8: 'Eight', 32: 'ThirtyTwo', 64: 'SixtyFour'}
pct = lambda x, n=1: ('%.*f' % (n, 100 * x)) + '\\%'
J = lambda p: json.load(open(p))


def build():
    """the macro definitions (list of '\\newcommand{\\n<Name>}{<value>}' lines), in the order of the paper"""
    out = []
    def d(name, val): out.append('\\newcommand{\\n%s}{%s}' % (name, val))
    F = lambda name: os.path.join(fig_dir(), name)                       # sidecars of the figure steps
    # ---- design
    sm = J(R(*CLOSED, 'summary.json')); os_ = J(F('tab_openset_op.json'))
    d('NPairs', sm['pairs_complete'])
    # ---- closed set (hierarchical)
    for nt in (8, 32, 64): d('Closed' + W[nt], pct(sm['per_nt'][str(nt)]['all']['acc_hier']))
    # ---- LLMmap (stored curve)
    ll = llmmap_curve()                                                  # {N_t: filtered mean accuracy}, N_t 1..8
    d('LLMmapFour', pct(ll[4]))
    bn = max(ll, key=ll.get); d('LLMmapBest', pct(ll[bn])); d('LLMmapBestNt', str(bn))   # best stored LLMmap-IF value over N_t 1..8 (filtered mean)
    # ---- open set
    r = os_['rows']['32']
    d('AucThirtyTwo', '%.3f' % r['auc']['est'])
    d('CfiftyAccThirtyTwo', pct(r['c50_acc']['est']))
    d('RecCfiftyThirtyTwo', pct(r['c50_unseen_recall']['est'])); d('PrecCfiftyThirtyTwo', pct(r['c50_unseen_precision']['est']))
    # ---- attribute errors (attribute missed, relative to an instance drawn at random): attr_miss.json of the attribute step
    am = J(F('attr_miss.json'))['attributes']
    for k, w in (('model', 'Model'), ('quant', 'Quant'), ('T', 'T'), ('SP', 'SP')): d('AttrRel' + w, pct(am[k]['ratio'], 0))
    d('AttrMissSP', pct(am['SP']['flips_miss'])); d('AttrRandSP', pct(am['SP']['random_miss']))
    f6 = J(F('fig06_unseen_pr.json')); d('UnseenPrev', pct(f6['unseen_prevalence']))
    # ---- abstract: integer percentages
    d('AbsClosedThirtyTwo', pct(sm['per_nt']['32']['all']['acc_hier'], 0)); d('AbsClosedEight', pct(sm['per_nt']['8']['all']['acc_hier'], 0)); d('AbsLLMmapBest', pct(ll[bn], 0))
    d('AbsRecCfifty', pct(r['c50_unseen_recall']['est'], 0)); d('AbsPrecCfifty', pct(r['c50_unseen_precision']['est'], 0))
    d('PrecLiftCfifty', '%.1f' % (r['c50_unseen_precision']['est'] / f6['unseen_prevalence']))   # Unseen precision at the 50 % target over the Unseen prevalence (precision of a random flag)
    # ---- showcase
    nums = J(os.path.join(closed_dir(), 'numbers.json'))                 # the closed-set build's numbers (showcase, closed-set precision / recall, pair accuracies)
    sc = nums['showcase']; lm = list(sc['llmmap_pct'].values()); fl = list(sc['flips_attributed_to_original_pct'].values())
    d('ShowLlmmapLo', '%d\\%%' % round(min(lm))); d('ShowLlmmapHi', '%d\\%%' % round(max(lm))); d('ShowFlipsLo', '%d\\%%' % round(min(fl))); d('ShowFlipsHi', '%d\\%%' % round(max(fl)))
    # ---- token pair 0-1 against the other pairs: NIST score and closed-set recall at N_t 32
    f4 = J(F('fig04_zero_one_vs_pairs.json'))
    d('FourNistOthers', '%.2f' % f4['nist_mean_others_mean']); d('FourNistZeroOne', '%.2f' % f4['nist_mean_pair01'])
    f4r = f4['recall']; assert f4r['nt'] == 32
    d('FourRecallOthers', pct(f4r['recall_mean_others'])); d('FourRecallZeroOne', pct(f4r['recall_mean_pair01']))
    d('FourShareBeatsAll', pct(f4['share_instances_pair01_beats_all_30_others'], 0))
    # ---- heatmap numbers (all 237 instances)
    h = J(F('fig12_closed_heatmap.json')); va = h['variation_average']; short = lambda s: s.split('/')[-1]
    d('HeatGrand', pct(h['grand'])); d('HeatLlmLow', pct(h['llm_lowest'][1])); d('HeatLlmLowName', short(h['llm_lowest'][0])); d('HeatLlmHigh', pct(h['llm_highest'][1])); d('HeatLlmHighName', short(h['llm_highest'][0]))
    plain = [v for k, v in va.items() if k.startswith('temp') or k.startswith('sp')]
    d('HeatVarLow', pct(min(plain))); d('HeatVarHigh', pct(max(plain)))
    d('HeatSpTwo', pct(va['sp 2 (temp 1.0)'])); d('HeatSpThree', pct(va['sp 3 (temp 1.0)'])); d('HeatSpFour', pct(va['sp 4 (temp 1.0)']))
    d('HeatTempZeroFour', pct(va['temp 0.4 (no sp)'])); d('HeatTempOneZero', pct(va['temp 1.0 (no sp)'])); d('HeatFpEight', pct(va['fp8 temp 1.0 (no sp)'])); d('HeatIntFour', pct(va['int4 temp 1.0 (no sp)'])); d('HeatAbli', pct(va['abliterated']))
    d('HeatQwenSeventyTwoSpTwo', pct(h['cells']['Qwen/Qwen2-72B-Instruct | sp 2 (temp 1.0)'], 0)); d('HeatQwenThirtySpTwo', pct(h['cells']['Qwen/Qwen3-30B-A3B-Instruct-2507 | sp 2 (temp 1.0)'], 0))
    # ---- feature importance
    nj = J(F('fig10_feature_importance.json')); gs = nj['group_share']; assert nj['n_features'] == {'NIST statistic': 31, 'length': 1, 'past dependence': 19, 'opening bits': 6}
    d('ImpTop', pct(nj['top5'][0][1])); d('ImpShareNist', pct(gs['NIST statistic'], 0)); d('ImpSharePast', pct(gs['past dependence'], 0)); d('ImpShareOpen', pct(gs['opening bits'], 0)); d('ImpShareLen', pct(gs['length'], 0))
    # ---- truncation (closed set, hierarchical)
    tr = J(R(*TRUNC, 'summary.json'))['per_nt']['32']
    d('TruncFullThirtyTwo', pct(tr['hier_full']['acc'])); d('TruncHundThirtyTwo', pct(tr['hier_100']['acc']))
    # ---- closed-set P/R (tempered) and pair accuracies
    d('PRRecallPNineNine', pct(nums['fig13']['by_nt']['32']['recall_at_p99'], 0))
    f15 = nums['fig15']
    d('PairAccMin', pct(f15['min'])); d('PairAccMax', pct(f15['max'])); d('PairAccMean', pct(f15['mean']))
    # ---- absorption (N_t 32): absorption.json of the absorption step
    d('AbsDistOne', pct(J(F('absorption.json'))['nt32']['i_distance1']['observed']))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="write numbers.tex (the \\n... macros of the paper text) from the results and the other steps' outputs")
    ap.add_argument("--data", help="data root (default: $FLIPS_DATA)"); ap.add_argument("--out", help="output root (default: $FLIPS_OUT)")
    a = ap.parse_args(argv)
    if a.data: os.environ["FLIPS_DATA"] = a.data
    if a.out: os.environ["FLIPS_OUT"] = a.out
    out = build()
    path = os.path.join(out_dir(), 'numbers.tex')
    open(path, 'w').write('% generated by paper/stats/make_numbers.py: do not edit\n' + '\n'.join(out) + '\n')
    print(f"[ok] {path} ({len(out)} macros)")


if __name__ == "__main__":
    main()
