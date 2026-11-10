"""Run a notebook's code cells in one Python namespace and store their real outputs in the .ipynb.

Stores what Jupyter would show: printed text, the value of a trailing expression (text and HTML), and matplotlib
figures at plt.show() or at the end of a cell. Used because no Jupyter kernel can be started on this machine.

  python bake_outputs.py in.ipynb out.ipynb CODE_DIR
"""
import ast
import base64
import contextlib
import io
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import nbformat                      # noqa: E402

src_path, out_path, code_dir = sys.argv[1:4]
os.environ["LATEGRAPH_CODE"] = code_dir
warnings.filterwarnings("ignore")
nb = nbformat.read(src_path, as_version=4)
ns = {"__name__": "__main__"}
state = dict(outputs=[], out=io.StringIO(), err=io.StringIO())


def flush_streams():
    for name in ("out", "err"):
        buf = state[name]
        if buf.getvalue():
            state["outputs"].append(nbformat.v4.new_output("stream", name="stdout" if name == "out" else "stderr",
                                                           text=buf.getvalue()))
            buf.seek(0); buf.truncate(0)


def grab_figures():
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        w, h = fig.get_size_inches() * fig.dpi
        state["outputs"].append(nbformat.v4.new_output(
            "display_data", data={"image/png": base64.b64encode(buf.getvalue()).decode(),
                                  "text/plain": f"<Figure size {w:.0f}x{h:.0f} with {len(fig.axes)} Axes>"}))
        plt.close(fig)


def show(*args, **kwargs):
    flush_streams(); grab_figures()


plt.show = show
t_all, n = time.time(), 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    n += 1; t0 = time.time()
    state["outputs"] = []
    tree = ast.parse(cell.source)
    last = ast.Expression(tree.body.pop().value) if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    with contextlib.redirect_stdout(state["out"]), contextlib.redirect_stderr(state["err"]):
        exec(compile(tree, f"<cell {n}>", "exec"), ns)
        val = eval(compile(last, f"<cell {n}>", "eval"), ns) if last is not None else None
    flush_streams(); grab_figures()
    if val is not None:
        data = {"text/plain": repr(val)}
        html = getattr(val, "_repr_html_", lambda: None)()
        if html:
            data["text/html"] = html
        state["outputs"].append(nbformat.v4.new_output("execute_result", data=data, execution_count=n))
    cell.outputs = state["outputs"]; cell.execution_count = n
    print(f"cell {n}: {len(cell.outputs)} outputs, {time.time() - t0:.0f}s", flush=True)
nbformat.write(nb, out_path)
print(f"all {n} code cells ran in {time.time() - t_all:.0f}s -> {out_path}")
