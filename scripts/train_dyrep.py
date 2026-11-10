"""Train a DyRep-style memory model (lategraph/dyrep.py) and save $LATEGRAPH_HOME/checkpoints/dyrep_<data>_s<seed>.pt.

  python scripts/train_dyrep.py --data wikipedia --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                         # noqa: E402
from lategraph.data import load                     # noqa: E402
from lategraph.dyrep import train_dyrep             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--threads", type=int, default=3)
args = ap.parse_args()
paths.ensure_dirs()
tag = f"dyrep_{args.data}_s{args.seed}"
t0 = time.time()
S = load(args.data)
_, info = train_dyrep(S, epochs=args.epochs, seed=args.seed, threads=args.threads, ckpt_path=paths.CKPT / f"{tag}.pt",
                      meta=dict(tag=tag), log=lambda m: print(m, flush=True))
info["wall_sec"] = time.time() - t0
out = paths.RESULTS / "training"; out.mkdir(parents=True, exist_ok=True)
with open(out / f"{tag}.json", "w") as f:
    json.dump(info, f, indent=1)
print(f"{tag}: best val AP {info['best_val_ap']:.4f} at epoch {info['best_epoch']} ({info['wall_sec'] / 60:.1f} min)")
