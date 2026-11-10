"""SEC Form 4 as a two-clock temporal graph (ZipLime "insider-trading" point-in-time data, Apache-2.0).

Each row of the PIT table is one disclosed insider transaction:
  occurrence time  t = event_date (the trade date, day resolution)
  arrival time     a = knowledge_date (EDGAR acceptance, seconds, UTC): when the market could know it
  src = reporting insider (CIK), dst = issuer (CIK); edge features = transaction code, derivative flag, size, role.
Amendments (Form 4/A) are real edits: a row with amendment_status 'amended' supersedes an earlier row
(supersedes_transaction_id); at its arrival the old version must be removed and the new one inserted.

Splits are by occurrence date: train 2023-01-01..2024-12-31, val 2025-01-01..2025-03-31, test 2025-04-01..2025-09-30.
History (train + val) uses the final corrected version of every transaction. The test period keeps every version, with
its own arrival time, so policies can be compared under the real delays and amendments.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import paths
from .data import EventStream

CODES = ["A", "M", "S", "F", "P", "D", "J", "C", "G"]           # everything else -> "other"
ROLES = ["Director", "Other Officer", "CEO", "CFO", "President", "10pct_owner", "COO", "Chair", "Other"]
SPLITS = dict(start="2023-01-01", val="2025-01-01", test="2025-04-01", end="2025-10-01")
COLS = ["operation", "event_date", "knowledge_date", "issuer_cik", "insider_cik", "primary_role", "transaction_code",
        "shares", "price", "transaction_value", "is_10b5_1", "is_derivative", "transaction_id", "logical_transaction_id",
        "revision", "supersedes_transaction_id", "amendment_status", "accession_number"]


def read_pit(root=None) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root or paths.SEC_ROOT)
    files = sorted(glob.glob(str(root / "pit" / "knowledge_year=*" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no PIT parquet files under {root}/pit (see scripts/prepare_data.py)")
    return pa.concat_tables([pq.read_table(f, columns=COLS) for f in files]).to_pandas()


def edge_features(df: pd.DataFrame, stats=None):
    """25-dim edge features; continuous columns are log-scaled and standardised with training statistics."""
    code = df.transaction_code.fillna("").to_numpy()
    f_code = np.stack([code == c for c in CODES] + [~np.isin(code, CODES)], 1).astype(np.float32)
    role = df.primary_role.fillna("Other").to_numpy()
    f_role = np.stack([role == r for r in ROLES], 1).astype(np.float32)
    p10 = df.is_10b5_1.to_numpy()
    f_plan = np.stack([p10 == True, p10 == False], 1).astype(np.float32)          # noqa: E712 (None -> neither)
    cont = np.stack([np.log1p(np.abs(df.shares.fillna(0).to_numpy())),
                     np.log1p(np.abs(df.price.fillna(0).to_numpy())),
                     np.log1p(np.abs(df.transaction_value.fillna(0).to_numpy()))], 1).astype(np.float32)
    if stats is None:
        stats = (cont.mean(0), cont.std(0) + 1e-6)
    cont = (cont - stats[0]) / stats[1]
    deriv = df.is_derivative.fillna(False).to_numpy().astype(np.float32)[:, None]
    return np.concatenate([f_code, deriv, cont, f_role, f_plan], 1), stats


def build_sec_stream(root=None, splits=SPLITS) -> EventStream:
    df = read_pit(root)
    df = df[df.operation == "UPSERT"]                                  # UNRESOLVED amendments are excluded
    df = df[df.amendment_status != "unchanged"]                        # a Form 4/A repeating a row: no new information
    df["occ"] = pd.to_datetime(df.event_date).astype("datetime64[s]").astype(np.int64)
    df["arr"] = df.knowledge_date.dt.tz_convert(None).astype("datetime64[s]").astype(np.int64)
    ts = {k: int(pd.Timestamp(v).value // 10**9) for k, v in splits.items()}
    df = df[(df.occ >= ts["start"]) & (df.occ < ts["end"]) & (df.arr >= df.occ)]
    df = df[df.insider_cik.notna() & df.issuer_cik.notna()].copy()   # a few rows have no reporting-owner CIK
    df["insider_cik"] = df.insider_cik.astype(str); df["issuer_cik"] = df.issuer_cik.astype(str)
    # amendment lineage: which row does each amended row replace?
    df["is_amend"] = df.amendment_status == "amended"
    tid_index = pd.Series(np.arange(len(df)), index=df.transaction_id.to_numpy())
    tid_index = tid_index[~tid_index.index.duplicated()]
    sup = df.supersedes_transaction_id.map(tid_index)                 # row position of the superseded version
    df["sup_pos"] = sup.fillna(-1).astype(np.int64).to_numpy()
    has_old = df.is_amend.to_numpy() & (df.sup_pos.to_numpy() >= 0)
    # history (occurrence before the test period): keep only final versions
    superseded = np.zeros(len(df), bool)
    superseded[df.sup_pos.to_numpy()[has_old]] = True
    hist = df.occ.to_numpy() < ts["test"]
    keep = ~(hist & superseded)
    # amended rows whose predecessor is missing (outside our window) behave like ordinary late events
    df["edit"] = has_old
    df = df[keep].copy()
    # node ids: pure insiders first, then all issuers (contiguous destination range)
    issuers = np.sort(df.issuer_cik.unique())
    insiders = np.sort(np.setdiff1d(df.insider_cik.unique(), issuers))
    nid = {c: i for i, c in enumerate(insiders)}
    nid.update({c: len(insiders) + i for i, c in enumerate(issuers)})
    df["src"] = df.insider_cik.map(nid).astype(np.int64)
    df["dst"] = df.issuer_cik.map(nid).astype(np.int64)
    df["rowid"] = np.arange(len(df))
    # sort by occurrence day, then arrival (a deterministic order inside a day)
    df = df.sort_values(["occ", "arr", "rowid"], kind="stable").reset_index(drop=True)
    old_to_new = np.empty(len(df), np.int64); old_to_new[df.rowid.to_numpy()] = np.arange(len(df))
    t = torch.from_numpy(df.occ.to_numpy())
    val0 = int((df.occ < ts["val"]).sum()); test0 = int((df.occ < ts["test"]).sum())
    train_df = df.iloc[:val0]
    _, stats = edge_features(train_df)
    feat, _ = edge_features(df, stats)
    # amendments in the test period: (new version, the version it replaces), in new event ids
    pos_of_tid = pd.Series(np.arange(len(df)), index=df.transaction_id.to_numpy())
    pos_of_tid = pos_of_tid[~pos_of_tid.index.duplicated()]
    amend_new, amend_old = [], []
    for i in np.flatnonzero(df.edit.to_numpy()):
        j = pos_of_tid.get(df.supersedes_transaction_id.iat[i], None)
        if j is not None and i >= test0 and j >= test0:
            amend_new.append(i); amend_old.append(int(j))
    days = df.occ.to_numpy()
    starts = np.r_[0, np.flatnonzero(days[1:] != days[:-1]) + 1]
    ends = np.r_[starts[1:], len(df)]
    S = EventStream(name="sec", src=torch.from_numpy(df.src.to_numpy()), dst=torch.from_numpy(df.dst.to_numpy()), t=t,
                    msg=torch.from_numpy(feat), num_nodes=len(nid), dst_min=len(insiders), dst_max=len(nid) - 1,
                    val0=val0, test0=test0, arrival=torch.from_numpy(df.arr.to_numpy()))
    S.extra = dict(day_bounds=[(int(a), int(b)) for a, b in zip(starts, ends)],
                   amend_new=torch.tensor(amend_new, dtype=torch.long), amend_old=torch.tensor(amend_old, dtype=torch.long),
                   n_insiders=len(insiders), n_issuers=len(issuers), splits=splits,
                   transaction_code=df.transaction_code.fillna("").to_numpy(),
                   is_derivative=df.is_derivative.fillna(False).to_numpy())
    return S


def load_sec_stream(cache=True) -> EventStream:
    path = paths.SEC_ROOT / "processed_stream.pt"
    if cache and path.exists():
        d = torch.load(path, weights_only=False)
        return EventStream(**d)
    S = build_sec_stream()
    if cache:
        torch.save(dict(name=S.name, src=S.src, dst=S.dst, t=S.t, msg=S.msg, num_nodes=S.num_nodes, dst_min=S.dst_min,
                        dst_max=S.dst_max, val0=S.val0, test0=S.test0, arrival=S.arrival, extra=S.extra), path)
    return S
