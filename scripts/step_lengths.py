"""Data statistics used in the text: the duration of one evaluation step (200 test events) per dataset, and the arrival
delays of SEC Form 4 filings in the test period. Writes results/step_lengths.json.

  python scripts/step_lengths.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                         # noqa: E402
from lategraph.data import BatchGrid, load          # noqa: E402

out = {}
for d in ("wikipedia", "reddit", "mooc", "lastfm"):
    S = load(d)
    g = BatchGrid(S.t, S.test0, 200)
    out[d] = dict(steps=g.J, minutes_per_step=float((g.T[-1] - g.T[0]) / g.J / 60))
    print(d, out[d])
S = load("sec")
lag = (S.arrival[S.test0:] - S.t[S.test0:]).double() / 86400
out["sec"] = dict(lag_median_days=float(lag.median()), lag_p99_days=float(lag.quantile(0.99)),
                  lag_over4_frac=float((lag > 4).double().mean()), test_events=int(S.num_events - S.test0))
print("sec", out["sec"])
json.dump(out, open(paths.RESULTS / "step_lengths.json", "w"), indent=1)
