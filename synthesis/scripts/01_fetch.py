"""Stage 1: fetch the n-back files only, by range reads of the source archives.

shin  78 files (cnt/mrk/mnt_nback.mat, 26 subjects, 2.3 GB) from the 6.1 GB MATLAB archive
      into eeg-analysis/Cleaned Data/, where eegrepro's loader expects them.
cog   522 files (zero/one/twoBACK .set/.fdt, 29 subjects x 3 sessions, ~13 GB) from the
      COG-BCI v4 subject archives into data/cogbci/.
Files already complete are skipped; zipfile checks every member's CRC-32.

usage: 01_fetch.py [shin|cog|all] [--jobs N]
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nbackjoin import loaders, sources
from nbackjoin.remote import extract, open_zip


def fetch_shin():
    z = open_zip(sources.SHIN_URL)
    members = [i for i in z.infolist() if "_nback.mat" in i.filename]
    assert len(members) == 78, len(members)
    for p in extract(z, members, sources.SHIN_DIR):
        print(p.relative_to(sources.SHIN_DIR), flush=True)
    print("shin: 78 files present", flush=True)


def fetch_cog(subject):
    loaders.cog_fetch(subject, sources.COG_DIR)
    print(f"cog sub-{subject:02d}: 18 files present", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("dataset", choices=["shin", "cog", "all"])
ap.add_argument("--jobs", type=int, default=3)
a = ap.parse_args()
if a.dataset in ("shin", "all"):
    fetch_shin()
if a.dataset in ("cog", "all"):
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(fetch_cog, sources.COG_SUBJECTS))
