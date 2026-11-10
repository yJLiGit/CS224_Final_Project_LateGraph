"""Where data, checkpoints and results live.

Everything large (raw data, checkpoints) goes under LATEGRAPH_HOME (default ~/.cache/lategraph) so the repository stays
small. Results (small JSON files) are written inside the repository, in results/.
"""
import os
from pathlib import Path

HOME = Path(os.environ.get("LATEGRAPH_HOME", Path.home() / ".cache" / "lategraph"))
DATA = HOME / "data"
JODIE_ROOT = DATA / "jodie"                 # PyG JODIEDataset root: <root>/<name>/raw/<name>.csv
SEC_ROOT = DATA / "sec_form4"               # ZipLime PIT parquet files: <root>/pit/knowledge_year=YYYY/*.parquet
CKPT = HOME / "checkpoints"
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
REPORT = REPO / "report"                    # LaTeX source of the milestone report


def ensure_dirs():
    for p in (DATA, JODIE_ROOT, SEC_ROOT, CKPT, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
