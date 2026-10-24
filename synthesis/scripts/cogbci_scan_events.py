"""Scan n-back event timing for every COG-BCI v4 subject by reading only the .set files
out of each remote zip with HTTP range requests (the .fdt signal files are skipped)."""
import io
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")
REC = "https://zenodo.org/records/7413650/files/sub-{:02d}.zip?download=1"


class HTTPRange(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def scan(sub):
    f = io.BufferedReader(HTTPRange(REC.format(sub)), buffer_size=1 << 20)
    z = zipfile.ZipFile(f)
    out = []
    with tempfile.TemporaryDirectory() as td:
        for info in z.infolist():
            name = info.filename
            if not name.endswith("BACK.set"):
                continue
            ses = name.split("/")[-3]
            p = Path(td) / f"{ses}_{Path(name).name}"
            p.write_bytes(z.read(info))
            # the .set holds the EEGLAB header and events; the signal lives in the skipped .fdt
            ann = mne.read_annotations(p)
            on, de = np.asarray(ann.onset), np.asarray(ann.description)
            tr = on[[len(d) == 4 and d[2] == "2" for d in de]]
            st = on[[len(d) == 4 and d[2:] == "11" for d in de]]
            out.append(dict(sub=sub, ses=ses, task=Path(name).stem, n_trials=int(len(tr)),
                            soa=round(float(np.median(np.diff(tr))), 3) if len(tr) > 1 else None,
                            n_block_start=int(len(st)), last_trial_s=round(float(tr[-1]), 1) if len(tr) else None,
                            n_boundary=int(sum(d == "boundary" for d in de)),
                            boundary_onsets=[round(float(o), 1) for o, d in zip(on, de) if d == "boundary"],
                            codes=sorted(set(de) - {"boundary"})))
    return out


if __name__ == "__main__":
    subs = [int(s) for s in sys.argv[1:]] or list(range(1, 30))
    for s in subs:
        try:
            for row in scan(s):
                print(json.dumps(row), flush=True)
        except Exception as e:  # keep going; report the failure
            print(json.dumps(dict(sub=s, error=repr(e))), flush=True)
