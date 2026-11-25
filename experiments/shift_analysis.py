"""Characterise the train->test depot shift per column.
For each test column: KS vs same-named train column, best-matching train column
(KS on raw values), and best log-scale factor if a pure rescale explains it.
Usage: python -I shift_analysis.py ../data
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

D = sys.argv[1] if len(sys.argv) > 1 else "../data"
tr = pd.read_csv(f"{D}/train.csv"); te = pd.read_csv(f"{D}/test.csv")
S = [c for c in tr.columns if c.startswith("s")]
rng = np.random.default_rng(0)
smp = lambda s: s.dropna().sample(min(5000, s.notna().sum()), random_state=0).values

trs = {c: smp(tr[c]) for c in S}
rows = []
for c in S:
    b = smp(te[c])
    if len(b) == 0:
        rows.append(dict(col=c, ks_same=np.nan, best=None, ks_best=np.nan, scale=np.nan,
                         ks_scaled=np.nan, nan_tr=tr[c].isna().mean(), nan_te=1.0)); continue
    ks_same = ks_2samp(trs[c], b).statistic
    ks_all = {o: ks_2samp(trs[o], b).statistic for o in S if len(trs[o])}
    best = min(ks_all, key=ks_all.get)
    # pure rescale: match medians of positive values
    a_pos, b_pos = trs[c][trs[c] > 0], b[b > 0]
    if len(a_pos) > 50 and len(b_pos) > 50:
        k = np.median(b_pos) / np.median(a_pos)
        ks_sc = ks_2samp(trs[c] * k, b).statistic
    else:
        k, ks_sc = np.nan, np.nan
    rows.append(dict(col=c, ks_same=ks_same, best=best, ks_best=ks_all[best], scale=k,
                     ks_scaled=ks_sc, nan_tr=tr[c].isna().mean(), nan_te=te[c].isna().mean()))
r = pd.DataFrame(rows).set_index("col")
r.to_csv("shift_report.csv")
pd.set_option("display.width", 200)
print("columns with KS(same) > 0.1:", (r.ks_same > 0.1).sum(), "of", len(S))
print("\nswapped? (best match is another column and clearly better):")
print(r[(r.best != r.index) & (r.ks_best < r.ks_same - 0.1)].round(3).to_string())
print("\nrescaled? (scaling fixes it):")
print(r[(r.ks_same > 0.1) & (r.ks_scaled < 0.05) & ((r.scale > 1.2) | (r.scale < 0.8))].round(3).to_string())
print("\nunexplained large shifts:")
print(r[(r.ks_same > 0.1) & ~((r.best != r.index) & (r.ks_best < r.ks_same - 0.1))
        & ~(r.ks_scaled < 0.05)].round(3).to_string())
print("\nNaN-rate shifts > 0.05:")
print(r[(r.nan_te - r.nan_tr).abs() > 0.05][["nan_tr", "nan_te"]].round(3).to_string())
