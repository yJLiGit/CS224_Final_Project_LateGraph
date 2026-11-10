"""Run the job lines '<script> <args...>' of a file, N at a time, from the repository root.

  python scripts/jobs/jobrunner.py scripts/jobs/main_jobs.txt 6 main     # logs go to $LATEGRAPH_HOME/logs/main/

A job waits until the checkpoints it needs exist: the one named by '--model <tag>' and any listed after '# needs:'
(file names in $LATEGRAPH_HOME/checkpoints), so experiments can be queued behind the training jobs they depend on.
"""
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

lines = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
N = int(sys.argv[2]); prefix = sys.argv[3]
PY = sys.executable
CWD = str(Path(__file__).resolve().parents[2])
HOME = Path(os.environ.get('LATEGRAPH_HOME', os.path.expanduser('~/.cache/lategraph')))
CKPT = HOME / 'checkpoints'
logdir = HOME / 'logs' / prefix; logdir.mkdir(parents=True, exist_ok=True)


def parse(line):
    cmd, _, needs = line.partition('# needs:')
    parts = shlex.split(cmd)
    req = [n for n in needs.split()]
    if '--model' in parts:
        req.append(parts[parts.index('--model') + 1] + '.pt')
    return cmd.strip(), parts, req


def name(i, a):
    return f"{i:03d}_" + Path(a.split()[0]).stem + '_' + '_'.join(x.strip('-').replace('/', '-').replace(':', '') for x in a.split()[1:])[:120]


jobs = [(i,) + parse(l) for i, l in enumerate(lines)]
running = []; t0 = time.time(); todo = list(jobs); fails = 0; waited = set()
while todo or running:
    while len(running) < N:
        ready = next((j for j in todo if all((CKPT / r).exists() for r in j[3])), None)
        if ready is None:
            break
        todo.remove(ready)
        i, a, parts, req = ready
        f = open(logdir / (name(i, a) + '.log'), 'w')
        p = subprocess.Popen([PY, '-W', 'ignore'] + parts, cwd=CWD, stdout=f, stderr=subprocess.STDOUT)
        running.append((p, i, a, f, time.time())); print(f"[{time.time()-t0:6.0f}s] start {i}: {a}", flush=True)
    for r in list(running):
        p, i, a, f, ts = r
        if p.poll() is not None:
            f.close(); running.remove(r); fails += p.returncode != 0
            last = open(logdir / (name(i, a) + '.log')).read().strip().splitlines()[-1:] or ['']
            print(f"[{time.time()-t0:6.0f}s] done {i} ({p.returncode}, {(time.time()-ts)/60:.1f} min): {a} | {last[0][:120]}", flush=True)
    if todo and not running and all(not all((CKPT / r).exists() for r in j[3]) for j in todo):
        missing = sorted({r for j in todo for r in j[3] if not (CKPT / r).exists()})
        key = tuple(missing)
        if key not in waited:
            print(f"[{time.time()-t0:6.0f}s] waiting for {len(missing)} checkpoints, e.g. {missing[:3]}", flush=True); waited.add(key)
    time.sleep(2)
print(f"ALL_DONE ({fails} failed)", flush=True)
