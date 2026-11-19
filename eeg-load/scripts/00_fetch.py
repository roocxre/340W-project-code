"""Stage 0: put the MAT and STEW datasets in data/, checked against recorded SHA-256 sums.

mat   PhysioNet "EEG During Mental Arithmetic Tasks" (eegmat) 1.0.0, open access. Every
      file in reference/eegmat_SHA256SUMS.txt (PhysioNet's own list, 75 files, 175 MB) is
      downloaded into data/eegmat/ and checked.
stew  IEEE DataPort serves STEW only to logged-in users (a free IEEE account), so it cannot
      be downloaded by a script. Download "STEW Dataset.zip" from
          https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset
      and either pass it with --stew PATH or leave it in data/ or ~/Downloads/. The zip (or an
      already extracted folder) is unpacked into data/STEW Dataset/ and its files are checked
      against reference/STEW_SHA256SUMS.txt (the copy behind the committed results).

usage: 00_fetch.py [mat|stew|all] [--stew PATH] [--check]
    --check   report only; exit 1 if STEW is neither in place nor found (nothing is written)
"""
import argparse
import hashlib
import io
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REF = ROOT / "reference"
MAT_DIR = DATA / "eegmat"
MAT_URL = "https://physionet.org/files/eegmat/1.0.0/"
STEW_DIR = DATA / "STEW Dataset"
STEW_PAGE = "https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset"


def manifest(name):
    out = {}
    for line in (REF / name).read_text().splitlines():
        if line.strip():
            h, f = line.split(None, 1)
            out[f.strip().lstrip("*")] = h
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------- MAT
def _download(url, out, expect):
    if out.exists() and sha256(out) == expect:
        return "present"
    tmp = out.with_name(out.name + ".part")
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if sha256(tmp) == expect:
                tmp.replace(out)
                return "downloaded"
            err = "SHA-256 mismatch"
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 410):
                raise
            err = repr(e)
        except OSError as e:
            err = repr(e)
        dt = min(300, 5 * 2 ** attempt)
        print(f"  retry {out.name} in {dt} s: {err}", flush=True)
        time.sleep(dt)
    raise IOError(f"{out.name}: could not be downloaded intact from {url}")


def fetch_mat():
    files = manifest("eegmat_SHA256SUMS.txt")
    MAT_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        status = list(ex.map(lambda kv: _download(MAT_URL + kv[0], MAT_DIR / kv[0], kv[1]), files.items()))
    shutil.copyfile(REF / "eegmat_SHA256SUMS.txt", MAT_DIR / "SHA256SUMS.txt")
    print(f"mat: {len(files)} files verified in {MAT_DIR.relative_to(ROOT)} "
          f"({status.count('downloaded')} downloaded, {status.count('present')} already present)", flush=True)


# ---------------------------------------------------------------------------- STEW
def _stew_required():
    return {f for f in manifest("STEW_SHA256SUMS.txt") if f.startswith("sub")}


def _complete(folder):
    return folder.is_dir() and all((folder / f).is_file() for f in _stew_required())


def _zip_has_stew(z):
    return any(Path(n).name == "sub01_lo.txt" for n in z.namelist())


def _open_stew_zip(path):
    """The zip itself, or a zip nested inside it, that holds the STEW text files."""
    z = zipfile.ZipFile(path)
    if _zip_has_stew(z):
        return z
    for n in z.namelist():
        if n.lower().endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(z.read(n)))
            if _zip_has_stew(inner):
                return inner
    return None


def find_stew_source(explicit=None):
    """Where the STEW files can be taken from: a complete folder or a zip. None if not found."""
    cands = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    downloads = Path.home() / "Downloads"
    for base in (DATA, downloads):
        if base.is_dir():
            cands += sorted(p for p in base.iterdir() if "stew" in p.name.lower())
    for c in cands:
        if c.is_dir():
            hit = next((p.parent for p in c.rglob("sub01_lo.txt")), None)
            if hit is not None and _complete(hit):
                return hit
        elif c.is_file() and zipfile.is_zipfile(c):
            try:
                if _open_stew_zip(c) is not None:
                    return c
            except (zipfile.BadZipFile, OSError):
                continue
    if explicit:
        raise SystemExit(f"--stew {explicit}: not a STEW folder or zip (no complete set of subXX_lo/hi.txt)")
    return None


def install_stew(explicit=None):
    wanted = manifest("STEW_SHA256SUMS.txt")
    if not _complete(STEW_DIR) or explicit:
        src = find_stew_source(explicit)
        if src is None:
            raise SystemExit(stew_instructions())
        if src.resolve() != STEW_DIR.resolve():
            STEW_DIR.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                for f in wanted:
                    if (src / f).is_file():
                        shutil.copyfile(src / f, STEW_DIR / f)
            else:
                z = _open_stew_zip(src)
                for n in z.namelist():
                    name = Path(n).name
                    if name in wanted and not n.endswith("/"):
                        (STEW_DIR / name).write_bytes(z.read(n))
            print(f"stew: unpacked {src} -> {STEW_DIR.relative_to(ROOT)}", flush=True)
    missing = sorted(f for f in _stew_required() if not (STEW_DIR / f).is_file())
    if missing:
        raise SystemExit(f"stew: {len(missing)} files missing from {STEW_DIR}: {missing[:6]} ...")
    differ = [f for f, h in wanted.items() if (STEW_DIR / f).is_file() and sha256(STEW_DIR / f) != h]
    if differ:
        print(f"WARNING stew: {len(differ)} files differ from the copy behind the committed results "
              f"(e.g. {differ[:3]}); results may differ slightly", flush=True)
    print(f"stew: {len(wanted) - len(differ)} of {len(wanted)} files verified in {STEW_DIR.relative_to(ROOT)}",
          flush=True)


def stew_instructions():
    return (f"\nSTEW is not in place. IEEE DataPort requires a (free) login, so it cannot be fetched "
            f"automatically:\n"
            f"  1. Sign in (or create a free IEEE account) at {STEW_PAGE}\n"
            f"  2. Download 'STEW Dataset.zip' (52 MB)\n"
            f"  3. Leave it in your Downloads folder, or in {DATA}, or pass --stew /path/to/STEW Dataset.zip\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", choices=["mat", "stew", "all"], default="all")
    ap.add_argument("--stew", help="path to 'STEW Dataset.zip' or an extracted STEW folder")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        ok = _complete(STEW_DIR) or find_stew_source(a.stew) is not None
        print("stew: " + ("found" if ok else "NOT FOUND" + stew_instructions()))
        sys.exit(0 if ok else 1)
    if a.dataset in ("mat", "all"):
        fetch_mat()
    if a.dataset in ("stew", "all"):
        install_stew(a.stew)


if __name__ == "__main__":
    main()
