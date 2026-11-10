"""Train the TIGER-style restarter for one trained TGN (distillation on the history).

  python scripts/train_restarter.py --data wikipedia --model tgn_wikipedia_s0
Saves $LATEGRAPH_HOME/checkpoints/<model>_restarter.pt.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                                   # noqa: E402
from lategraph.data import load                               # noqa: E402
from lategraph.memory import EventTable                       # noqa: E402
from lategraph.model import load_checkpoint                   # noqa: E402
from lategraph.restarter import train_restarter               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--threads", type=int, default=3)
args = ap.parse_args()
torch.set_num_threads(args.threads)
t0 = time.time()
S = load(args.data)
model = load_checkpoint(paths.CKPT / f"{args.model}.pt")
table = EventTable(S.src, S.dst, S.t, S.msg, torch.full((S.num_events,), -1, dtype=torch.long))
hist = []
R = train_restarter(model, S, table, epochs=args.epochs, log=lambda m: (print(m), hist.append(m)))
torch.save(R.state_dict(), paths.CKPT / f"{args.model}_restarter.pt")
import json, re                                                # noqa: E402
ep = [(float(a), float(b)) for a, b in re.findall(r"MSE ([\d.]+) \(memory variance ([\d.]+)\)", "\n".join(hist))]
(paths.RESULTS / "restarter").mkdir(parents=True, exist_ok=True)
json.dump(dict(model=args.model, epochs=len(ep), mse=ep[-1][0], memory_variance=ep[-1][1], explained_variance=1 - ep[-1][0] / ep[-1][1]),
          open(paths.RESULTS / "restarter" / f"{args.model}.json", "w"), indent=1)
print(f"saved restarter for {args.model} in {time.time() - t0:.0f}s")
