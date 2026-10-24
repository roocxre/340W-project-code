"""Stage 0: export COG-BCI's notebook.mat (a MATLAB table scipy cannot read) to CSV.

One row per subject, session and n-back level: the session date, the level's position
among the session's 8 tasks, and the experimenter's interruption flag and notes.
Task codes (COG-BCI_info.pdf): 1 PVT, 2 Flanker, 3 Two-Back, 4 One-Back, 5 Zero-Back, 6-8 MATB.
"""
import csv
import sys
import tempfile
import urllib.request
from pathlib import Path

import matio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nbackjoin import sources

URL = f"https://zenodo.org/records/{sources.COG_RECORD}/files/notebook.mat?download=1"
CODE_N = {5: 0, 4: 1, 3: 2}

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "notebook.mat"
    urllib.request.urlretrieve(URL, p)
    nb = matio.load_from_mat(p)["notebook"]

rows = []
top = nb.ravel()[0]
for i in range(29):
    subj = top[f"SBJ_{i + 1}"].ravel()[0]
    for k in range(3):
        s = subj[f"SESS_{k + 1}"].ravel()[0]
        order = [int(v) for v in s["Order"].ravel()]
        notes = " ".join(str(x) for x in s["Notes"].ravel()).strip() if "Notes" in s.dtype.names else ""
        for code, n in CODE_N.items():
            rows.append(dict(subject=f"cog-{i + 1:02d}", session=k + 1, n=n,
                             date=str(s["Date"].ravel()[0])[:10], task_position=order.index(code) + 1,
                             interrupted=bool(s["interrupted"].ravel()[0]), notes=notes))
assert len(rows) == 29 * 3 * 3, len(rows)
out = sources.RESULTS / "cogbci_notebook.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(out, len(rows))
