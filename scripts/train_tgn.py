"""Train one TGN and save its checkpoint (best validation AP).

  python scripts/train_tgn.py --data wikipedia --seed 0                      # standard TGN
  python scripts/train_tgn.py --data wikipedia --seed 0 --emb id             # memory-only TGN
  python scripts/train_tgn.py --data wikipedia --seed 0 --emb nomem          # memory-free control
  python scripts/train_tgn.py --data wikipedia --seed 0 --lam 10 --rho 0.75  # contraction penalty ("repairable")

Checkpoints go to $LATEGRAPH_HOME/checkpoints/<tag>.pt, the training curve to results/training/<tag>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                                 # noqa: E402
from lategraph.data import load                             # noqa: E402
from lategraph.model import TGNConfig                       # noqa: E402
from lategraph.train import fixed_batches, train_tgn        # noqa: E402


def model_tag(data, seed, emb="attn", lam=0.0, rho=0.9, updater="gru", crho=None, c=0.5):
    tag = f"tgn_{data}_s{seed}"
    if emb == "proj" and updater == "rnn":
        return tag + "_jodie"                          # JODIE memory model (RNN updater + time projection)
    if emb != "attn":
        tag += f"_{emb}"
    if updater == "rnn":
        tag += "_rnn"
    elif updater == "crnn":
        tag += f"_crnn{crho:g}" + (f"c{c:g}" if c != 0.5 else "")    # contractive by construction, rho = crho, row bound c
    if lam > 0:
        tag += f"_rho{rho}_lam{lam:g}"
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--emb", default="attn", choices=["attn", "id", "nomem", "proj"])
    ap.add_argument("--updater", default="gru", choices=["gru", "rnn", "crnn"])
    ap.add_argument("--crho", type=float, default=0.9, help="crnn: contraction constant rho = 1 - a_min (1 - c)")
    ap.add_argument("--c", type=float, default=0.5, help="crnn: L1 bound of each row of the memory block of W")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--suffix", default="", help="appended to the checkpoint name (e.g. _ep30 for a longer-trained model)")
    args = ap.parse_args()
    paths.ensure_dirs()
    tag = model_tag(args.data, args.seed, args.emb, args.lam, args.rho, args.updater, args.crho, args.c) + args.suffix
    amin = (1 - args.crho) / (1 - args.c)              # rho = 1 - a_min (1 - c)
    cfg = TGNConfig(emb=args.emb, updater=args.updater, crnn_amin=amin, crnn_c=args.c)
    out_dir = paths.RESULTS / "training"; out_dir.mkdir(parents=True, exist_ok=True)
    S = load(args.data)
    train_b = val_b = None
    if "day_bounds" in S.extra:                      # SEC: one batch per calendar day of occurrence
        train_b = [(lo, hi) for lo, hi in S.extra["day_bounds"] if hi <= S.val0]
        val_b = [(lo, hi) for lo, hi in S.extra["day_bounds"] if lo >= S.val0 and hi <= S.test0]
    log_lines = []

    def log(msg):
        print(msg, flush=True); log_lines.append(msg)

    t0 = time.time()
    _, info = train_tgn(S, cfg, epochs=args.epochs, seed=args.seed, lam=args.lam, rho=args.rho,
                        train_batches=train_b, val_batches=val_b, threads=args.threads,
                        ckpt_path=paths.CKPT / f"{tag}.pt", log=log, meta=dict(tag=tag))
    info["wall_sec"] = time.time() - t0
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(info, f, indent=1)
    log(f"{tag}: best val AP {info['best_val_ap']:.4f} at epoch {info['best_epoch']} ({info['wall_sec']/60:.1f} min)")


if __name__ == "__main__":
    main()
