"""Download the data: JODIE Wikipedia / Reddit / MOOC / LastFM (through PyG), TGB tgbl-review (event table from the
Hugging Face mirror) and the SEC Form 4 point-in-time files.

  python scripts/prepare_data.py [--only jodie|tgb|sec]
Files go to $LATEGRAPH_HOME/data (default ~/.cache/lategraph/data). The SEC files are pinned to one dataset revision.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph import paths                                    # noqa: E402

SEC_REPO = "ZipLime/insider-trading"
SEC_REVISION = "dd96ce005c3e82438a6017ba157357641b80e159"
SEC_YEARS = [2023, 2024, 2025, 2026]


def jodie():
    from torch_geometric.datasets import JODIEDataset
    for name in ("wikipedia", "mooc", "reddit", "lastfm"):
        d = JODIEDataset(str(paths.JODIE_ROOT), name=name)[0]
        print(f"{name}: {d.num_events} events, {d.num_nodes} nodes")


TGB_MIRROR = "https://huggingface.co/datasets/stanford-star/tgb/resolve/main"


def tgb(name="tgbl-review"):
    """Only the event table (54 MB); TGB's official negative samples (pickles) are not needed for our protocol."""
    out = paths.DATA / "tgb" / name / "db"; out.mkdir(parents=True, exist_ok=True)
    dst = out / "events.parquet"
    if not dst.exists():
        urllib.request.urlretrieve(f"{TGB_MIRROR}/{name}/db/events.parquet", dst)
    from lategraph.data import load
    print(load(name).summary())


def sec():
    for y in SEC_YEARS:
        api = f"https://huggingface.co/api/datasets/{SEC_REPO}/tree/{SEC_REVISION}/data/pit/knowledge_year%3D{y}"
        files = [x["path"] for x in json.load(urllib.request.urlopen(api)) if x["type"] == "file"]
        out = paths.SEC_ROOT / "pit" / f"knowledge_year={y}"; out.mkdir(parents=True, exist_ok=True)
        for p in files:
            dst = out / Path(p).name
            if not dst.exists():
                urllib.request.urlretrieve(f"https://huggingface.co/datasets/{SEC_REPO}/resolve/{SEC_REVISION}/{p}", dst)
        print(f"SEC knowledge_year={y}: {len(files)} files")
    (paths.SEC_ROOT / "REVISION").write_text(SEC_REVISION)
    from lategraph.sec import load_sec_stream
    S = load_sec_stream()
    print(S.summary())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    paths.ensure_dirs()
    if a.only in (None, "jodie"):
        jodie()
    if a.only in (None, "tgb"):
        tgb()
    if a.only in (None, "sec"):
        sec()
