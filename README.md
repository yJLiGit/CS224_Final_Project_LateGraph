# LateGraph — repairing (and really forgetting) temporal-GNN memory when the past changes

CS 224W (Fall 2026) course project — code of the final project (blog post, Colab, paper draft). Yijin Li, Lexi Zhou.

Memory-based temporal GNNs (TGN, JODIE, DyRep) keep one memory vector per node and update it at every event. Real logs
change after the fact: events arrive **late**, filings get **amended**, users ask for **deletions**. One edit to the past
changes every later memory update that read an affected memory (the edit's *dependency cone*). This repository contains

* `RetroactiveTGNMemory` (`lategraph/memory.py`): a subclass of PyG's `TGNMemory` that can insert, delete and amend past
  events and repair its memory **exactly** (recompute only the cone, from a journal of saved states; a **step index**
  visits only the steps the cone touches), **approximately** (ε-repair), or with a proven tolerance (**δ-certified**
  repair, for a contractive memory);
* other memory models: JODIE (RNN memory + time projection), **DyRep** (`lategraph/dyrep.py`: the update reads the
  partner's neighbourhood, with its own journal and exact repair), and `ContractiveRNNCell` (`lategraph/model.py`), a
  memory update that is a contraction for every input;
* `TimeOrderedNeighborStore`: a last-k neighbour store ordered by occurrence time (insertion-order invariant, supports
  deletion), a drop-in for `LastNeighborLoader`;
* experiments: exactness (`run_exactness.py`), late events with 13 policies (`run_stream.py`), the deletion leak with
  three attacks (`run_deletion.py`) and a **likelihood-ratio attack with counterfactual shadow streams**
  (`run_lira.py`; `--know`, `--after` and `--K` weaken the attacker), deletions of **training-period** events against retraining (`run_train_deletion.py`), contraction and
  certificates (`run_contraction.py`), **exact counterfactual explanations** vs attention (`run_influence.py`), DyRep
  (`run_dyrep.py`) and timing benchmarks (`bench_repair.py`, `bench_age.py`);
* data: JODIE Wikipedia / Reddit / MOOC / LastFM, TGB tgbl-review, and the SEC Form 4 two-clock graph (trade date vs.
  EDGAR acceptance time, Form 4/A amendments as real edits).

The milestone version of this code is in `../../milestone/lategraph` (unchanged).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py            # JODIE (via PyG) + SEC Form 4 point-in-time files (pinned revision)
pytest -q tests/                          # kernel = PyG TGNMemory; exact repair = sequential recomputation
```

The Colab notebook `notebooks/LateGraph_final.ipynb` (saved outputs included) runs a 20-minute tour on a CPU: train a
TGN, change the past and repair, the step index, late-event policies, the deletion leak and the likelihood-ratio attack,
the contractive cell and its bound, exact counterfactual explanations, and every result figure from `results/`.

```python
from lategraph.memory import RetroactiveTGNMemory
mem = RetroactiveTGNMemory.from_trained(tgn.memory, events)   # same weights as the trained TGNMemory
mem.warmup(0, stream.test0)                                    # process the history in order
mem.advance(step_events)                                       # one step of the stream (journaled)
mem.edit(insert=late_events, delete=deleted_events)            # change the past
stats = mem.repair("exact")                                    # or "eps" (eps=0.05), "replay", "khop"
z, last_update = mem(n_id)                                     # use it like TGNMemory in eval mode
```

## Reproducing the final results

All large files live in `$LATEGRAPH_HOME` (default `~/.cache/lategraph`; about 4 GB of data and checkpoints, 2.4 GB of it
the Reddit CSV); results are small JSON files in `results/` (one per run). The job lists in `scripts/jobs/` hold every
run behind the paper and the blog post; `jobrunner.py` runs a list N jobs at a time and waits for the checkpoints a job
needs, so lists can be started together:

```bash
python scripts/jobs/jobrunner.py scripts/jobs/final_train_jobs.txt 5 train          # TGN seeds 3-4, LastFM
python scripts/jobs/jobrunner.py scripts/jobs/final_train2_jobs.txt 2 train2        # JODIE, contractive TGN, lambda ablation
python scripts/jobs/jobrunner.py scripts/jobs/final_dyrep_jobs.txt 2 dyrep          # DyRep: training + experiments
python scripts/jobs/jobrunner.py scripts/jobs/final_exp_jobs.txt 6 exp              # exactness, streaming, deletion, LiRA, ...
python scripts/jobs/jobrunner.py scripts/jobs/final_trdel_jobs.txt 2 trdel          # training-period deletions
python scripts/jobs/jobrunner.py scripts/jobs/final_trdel_recent_jobs.txt 1 recent
python scripts/jobs/jobrunner.py scripts/jobs/final_crnn_c_jobs.txt 1 crnn_c         # contractive cell: row-bound ablation
python scripts/jobs/jobrunner.py scripts/jobs/final_review_jobs.txt 1 review        # tgbl-review (then final_review_exp_jobs.txt)
python scripts/jobs/jobrunner.py scripts/jobs/final_review2_jobs.txt 3 review2      # tgbl-review seeds 1-2, learned restart
python scripts/jobs/jobrunner.py scripts/jobs/final_lira_know_jobs.txt 6 know       # weaker attackers (part of the events)
python scripts/jobs/jobrunner.py scripts/jobs/final_lira_after_jobs.txt 5 after     # attacks after the request; fewer pairs (K)
python scripts/jobs/jobrunner.py scripts/jobs/final_ep30_jobs.txt 2 ep30            # 30-epoch models: does the leak change?
python scripts/bench_repair.py --data wikipedia reddit mooc lastfm tgbl-review       # timing, on an idle machine
python scripts/bench_age.py --data wikipedia reddit mooc lastfm                      # repair time against the age of the edit
python scripts/analyze_influence.py                                                  # attention against each channel of influence
python scripts/aggregate.py && python scripts/make_numbers_final.py && python scripts/make_tables_final.py --data \
  && python scripts/make_figures_final.py && python scripts/verify_final.py
```
(`final_rest_jobs.txt` / `final_rest2_jobs.txt` are the same jobs re-prioritised after a restart; `final_review3_jobs.txt`
and `final_review4*_jobs.txt` are the rest of `final_review2_jobs.txt`, run again after a restart.) On one M3 Max laptop
the whole set takes about a day of CPU time. Unit tests: `pytest -q tests/` (18 tests, about 30 s).

## Layout

```
lategraph/   data.py (streams, splits, step grid; JODIE, TGB, SEC)  delays.py  model.py (TGN, JODIE, ContractiveRNNCell)
             train.py  memory.py (RetroactiveTGNMemory)  dyrep.py (DyRep + its retroactive memory)  neighbors.py
             stream.py (policies, evaluation)  restarter.py  sec.py (SEC Form 4)  stats.py  paths.py
scripts/     prepare_data, train_tgn, train_dyrep, train_restarter, run_exactness, run_stream, run_deletion, run_lira,
             run_train_deletion, run_contraction, run_influence, run_dyrep, bench_repair; aggregate, make_numbers_final,
             make_tables_final, make_figures_final, verify_final; jobs/ (job lists and a dependency-aware parallel runner)
tests/       unit tests (synthetic data, no download): kernel = PyG, exact repair = recomputation (float32/float64, ties),
             indexed repair = full walk bit for bit, other memory models, DyRep, the contractive cell and its bounds
notebooks/   LateGraph_final.ipynb (the Colab of the final project, with saved outputs) + make_notebook_final.py;
             the milestone notebook
results/     JSON results of every run, summary.json (results_milestone/: the milestone's runs, not used)
report/      LaTeX source of the milestone report
```
The blog post is in `../blog/`, the paper draft in `../paper/`.

## Data and licences

* JODIE Wikipedia / Reddit / MOOC / LastFM (Kumar et al., KDD 2019), loaded with `torch_geometric.datasets.JODIEDataset`.
* tgbl-review (Huang et al., NeurIPS 2023, TGB) from the Hugging Face mirror `stanford-star/tgb` (`db/events.parquet`).
* SEC Form 4 point-in-time view from `ZipLime/insider-trading` on Hugging Face (Apache-2.0; source: SEC EDGAR), revision
  `dd96ce005c3e82438a6017ba157357641b80e159`, knowledge years 2023-2026.
