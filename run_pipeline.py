#!/usr/bin/env python3
"""Reproduce every result in synthesis/ and analysis/ from the raw datasets.

    python run_pipeline.py                 # everything, unattended (~1 day); safe to rerun
    python run_pipeline.py --list          # stages, their status and typical run times
    python run_pipeline.py --setup-only    # checks + environment + unit tests, then stop

Needs only Python 3.8+ to start. It installs everything else itself:
  1. preflight: platform, free disk, RAM, and network access to every data source;
  2. environment: .venv/ with Python 3.12 and the exact packages in requirements.txt
     (via uv, which is installed into .pipeline/tools/ if it is not already available);
  3. unit tests of synthesis, analysis and eeg-load's loadrepro (R-LMD, BAO and the
     optimizable classifiers that analysis uses for the Yedukondalu baselines);
  4. every data download (Shin 2018, COG-BCI), then every analysis stage, in order.

eeg-load's own reproduction on MAT and STEW is not part of it: only its code is used.

Each finished stage is recorded in .pipeline/state.json, so after an interruption (crash,
reboot, Ctrl-C) running the same command again resumes at the stage that did not finish.
Stages also cache their own finished pieces (per subject, per model), so little is redone.
Output of every stage goes to the console and to the stage's log file, the same log files
as the committed ones (e.g. analysis/results/log_04_baselines.txt).

Options: --from STAGE (rerun STAGE and everything after it), --only STAGE [...], --redo
(rerun everything), --cpus N (use at most N cores),
--dry-run (print the commands only), --skip-checks (do not stop on platform/disk checks).
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
WORK = REPO / ".pipeline"
STATE = WORK / "state.json"
LOGS = WORK / "logs"
TOOLS = WORK / "tools"
VENV = REPO / ".venv"
REQ = REPO / "requirements.txt"
PY_VERSION = "3.12"
UV_VERSION = "0.11.24"
NEED_DISK_GB = 30          # 13 GB raw EEG + 2 GB windows + caches + environment, with margin
IS_WIN = os.name == "nt"

SOURCES = {   # every URL the pipeline downloads from (HEAD-checked in preflight)
    "Shin 2018 (TU Berlin)": "https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/EEG/EEG_01-26_MATLAB.zip",
    "COG-BCI v4 (Zenodo)": "https://zenodo.org/records/7413650/files/sub-01.zip?download=1",
}

# Packages imported somewhere in the pipeline: checked right after installation.
IMPORTS = ("numpy scipy pandas sklearn matplotlib mne mne_icalabel onnxruntime picard matio numba "
           "joblib optuna pyriemann torch xgboost pytest psutil")


# --------------------------------------------------------------------------- output
class Tee:
    """Mirror everything the pipeline prints into one run log."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.path = path

    def write(self, s):
        try:
            sys.__stdout__.write(s)
        except UnicodeEncodeError:
            sys.__stdout__.write(s.encode("ascii", "replace").decode())
        sys.__stdout__.flush()
        self.f.write(s)
        self.f.flush()


OUT = None


def say(*parts):
    OUT.write(" ".join(str(p) for p in parts) + "\n")


def banner(text):
    say("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def die(msg):
    say(f"\nSTOPPED: {msg}")
    sys.exit(1)


def fmt_dt(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


# --------------------------------------------------------------------------- machine
def ram_gb(available=False):
    """Total physical RAM in GB; with available=True, what a new process can still use right now
    (Windows: free physical memory and free commit, whichever is smaller; Linux: MemAvailable;
    None where it cannot be measured)."""
    try:
        if available and sys.platform.startswith("linux"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
            return None
        if available and not IS_WIN:
            return None
        if IS_WIN:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            if available:
                return min(m.ullAvailPhys, m.ullAvailPageFile) / 2 ** 30
            return m.ullTotalPhys / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:
        return None if available else 8.0


class Resources:
    def __init__(self, cpus_cap=None):
        self.cpus = max(1, min(os.cpu_count() or 1, cpus_cap or 10 ** 6))
        self.ram = ram_gb()

    def jobs(self, gb_per_job):
        """Parallel workers that fit in memory: 2 GB of total RAM left for the system, and no more
        than is available right now, so other programs' memory is respected (see Jobs)."""
        usable = self.ram - 2
        avail = ram_gb(available=True)
        if avail is not None:
            usable = min(usable, avail - 1)
        return max(1, min(self.cpus, int(usable // gb_per_job)))


class Jobs:
    """A --jobs value worked out when its stage starts (memory in use changes during a run)."""

    def __init__(self, f):
        self.f = f

    def __str__(self):
        return str(self.f())


def keep_awake():
    """Stop the machine from sleeping while the pipeline runs (best effort)."""
    try:
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
            return "caffeinate"
        if IS_WIN:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM
            return "SetThreadExecutionState"
        if shutil.which("systemd-inhibit"):
            p = subprocess.Popen(["systemd-inhibit", "--what=sleep:idle", "--why=run_pipeline.py",
                                  "--mode=block", "sleep", "infinity"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            import atexit
            atexit.register(p.terminate)
            return "systemd-inhibit"
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- preflight
NEEDS_NET = ("fetch:",)


def preflight(args, res, todo):
    banner("Preflight checks")
    need_net = any(s.name.startswith(NEEDS_NET) for s in todo) or not venv_python().exists()
    problems, warnings = [], []
    mach = platform.machine().lower()
    say(f"system   {platform.system()} {platform.release()} ({mach}), Python {platform.python_version()} "
        f"at {sys.executable}")
    say(f"machine  {res.cpus} CPU threads, {res.ram:.0f} GB RAM")
    if sys.version_info < (3, 8):
        problems.append("Python 3.8 or newer is needed to start run_pipeline.py")
    if sys.platform == "darwin":
        ver = tuple(int(x) for x in (platform.mac_ver()[0] or "0").split(".")[:2])
        if mach not in ("arm64", "aarch64"):
            problems.append("Intel Macs are not supported: PyTorch and numba publish no wheels for them "
                            "at the pinned versions. Use an Apple Silicon Mac, Linux x86_64 or Windows x86_64.")
        elif ver < (14, 0):
            problems.append(f"macOS {platform.mac_ver()[0]}: macOS 14 (Sonoma) or newer is needed "
                            "(onnxruntime, used by ICLabel, ships wheels for macOS 14+ only).")
    elif sys.platform.startswith("linux") and mach not in ("x86_64", "amd64"):
        warnings.append(f"Linux on {mach}: not tested; mat-io has no prebuilt wheel, so a C compiler "
                        "is needed to install it.")
    elif IS_WIN and mach not in ("amd64", "x86_64"):
        warnings.append(f"Windows on {mach}: not tested; Windows x86_64 is.")

    free = shutil.disk_usage(REPO).free / 2 ** 30
    have = sum(du(REPO / p) for p in ("synthesis/data", "eeg-analysis/Cleaned Data", "synthesis/cache",
                                       "analysis/cache", ".venv")) / 2 ** 30
    need = max(3.0, NEED_DISK_GB - have)
    say(f"disk     {free:.0f} GB free, about {need:.0f} GB still needed")
    if free < need:
        problems.append(f"only {free:.0f} GB free on the disk holding {REPO}; about {need:.0f} GB is needed")
    if res.ram < 8:
        warnings.append(f"{res.ram:.0f} GB RAM: runs with few parallel workers, so it will be slow")

    for name, url in (SOURCES.items() if need_net else ()):
        err, note = None, ""
        for attempt in range(3):
            err = head(url)
            if err == "certificate":
                # This Python's trust store lacks the server's root CA (common on Windows, which adds
                # roots lazily). Reachability is checked without verification here; downloads run in
                # .venv with certifi's CA bundle, and tls_check() verifies that after setup.
                err = head(url, verify=False)
                note = " (certificate verified after setup, with the certifi CA bundle)"
            if err is None:
                break
            time.sleep(10)
        say(f"network  {name}: {'ok' + note if err is None else f'FAILED ({err})'}")
        if err:
            problems.append(f"cannot reach {name} at {url}")

    for w in warnings:
        say(f"WARNING  {w}")
    if problems:
        msg = "\n".join(f"  - {p}" for p in problems)
        if args.skip_checks and not any("reach" in p for p in problems):
            say(f"WARNING  continuing despite (--skip-checks):\n{msg}")
        else:
            die(f"fix the following, then run the same command again:\n{msg}")
    say("preflight ok")


def head(url, verify=True):
    """None if a HEAD request succeeds, "certificate" on a TLS trust failure, else the error."""
    import ssl
    ctx = None if verify else ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60, context=ctx) as r:
            return None if r.status == 200 else f"HTTP {r.status}"
    except Exception as e:
        if verify and "CERTIFICATE_VERIFY_FAILED" in repr(e):
            return "certificate"
        return repr(e)


def ca_bundle():
    """certifi's CA bundle inside .venv (Mozilla's root store), or None before setup."""
    site = VENV / ("Lib/site-packages" if IS_WIN else f"lib/python{PY_VERSION}/site-packages")
    p = site / "certifi" / "cacert.pem"
    return str(p) if p.exists() else None


def tls_check(py):
    """Verified HTTPS to every data source from .venv, as the download stages will do it."""
    code = ("import sys, urllib.request\n"
            "for url in sys.argv[1:]:\n"
            "    try:\n"
            "        urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60).close()\n"
            "        print('ok    ', url)\n"
            "    except Exception as e:\n"
            "        print('FAILED', url, repr(e)); sys.exit(1)\n")
    say("checking verified HTTPS to the data sources from .venv")
    p = run_quiet([py, "-c", code] + list(SOURCES.values()), env=child_env())
    if p.returncode != 0:
        die("a data source failed TLS verification (output above). If this network intercepts HTTPS "
            "(a corporate proxy), set SSL_CERT_FILE to its CA bundle and run again.")


def du(p):
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


# --------------------------------------------------------------------------- environment
def venv_python():
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


class Done:
    def __init__(self, returncode, stdout):
        self.returncode, self.stdout = returncode, stdout


def run_quiet(cmd, **kw):
    """Run a setup command, echoing its output as it arrives."""
    say("  $ " + " ".join(str(c) for c in cmd))
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
    lines = []
    for raw in iter(p.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        lines.append(line)
        say("  | " + line)
    return Done(p.wait(), "\n".join(lines))


def find_uv():
    """uv at the pinned version: .pipeline/tools/ (installed on first use), else uv on PATH."""
    tools_bin = TOOLS / ("Scripts" if IS_WIN else "bin")
    uv = tools_bin / ("uv.exe" if IS_WIN else "uv")
    if uv.exists():
        return uv
    say(f"installing uv {UV_VERSION} into {TOOLS.relative_to(REPO)}/")
    p = run_quiet([sys.executable, "-m", "venv", TOOLS])
    if p.returncode == 0:
        tpy = tools_bin / ("python.exe" if IS_WIN else "python")
        p = run_quiet([tpy, "-m", "pip", "install", "-q", "--disable-pip-version-check", f"uv=={UV_VERSION}"])
        if p.returncode == 0 and uv.exists():
            return uv
    system_uv = shutil.which("uv")
    if system_uv:
        say(f"using uv from PATH: {system_uv}")
        return Path(system_uv)
    die("could not install uv. On Debian/Ubuntu install the venv module first "
        "(sudo apt install python3-venv python3-pip), or install uv from https://docs.astral.sh/uv/ "
        "and run again.")


def env_fingerprint():
    return hashlib.sha256(REQ.read_bytes()).hexdigest()[:16] + f"-py{PY_VERSION}"


def ensure_env():
    banner("Environment (.venv, Python %s, requirements.txt)" % PY_VERSION)
    marker = VENV / ".run_pipeline-env"
    py = venv_python()
    if marker.exists() and marker.read_text().strip() == env_fingerprint() and py.exists():
        say("environment up to date")
        return py
    if VENV.exists():
        ver = subprocess.run([str(py), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True) if py.exists() else None
        if ver is None or ver.returncode != 0 or ver.stdout.strip() != PY_VERSION:
            if (VENV / ".run_pipeline-created").exists():
                say("recreating .venv (wrong or broken Python)")
                shutil.rmtree(VENV)
            else:
                die(f"{VENV} exists but is not a Python {PY_VERSION} environment made by this script. "
                    "Move or delete it and run again.")
    uv = find_uv()
    env = dict(os.environ, UV_PYTHON_DOWNLOADS="automatic")
    if not VENV.exists():
        p = run_quiet([uv, "venv", "--python", PY_VERSION, VENV], env=env)
        if p.returncode != 0:
            die("could not create .venv with uv (output above)")
        (VENV / ".run_pipeline-created").write_text("created by run_pipeline.py\n")
    say("installing requirements (first time: several minutes, ~1-4 GB of downloads)")
    p = run_quiet([uv, "pip", "install", "--python", py, "-r", REQ], env=env)
    if p.returncode != 0:
        die("package installation failed (output above)")
    check = ("import importlib, sys\n"
             f"mods = {IMPORTS.split()!r}\n"
             "bad = []\n"
             "for m in mods:\n"
             "    try: importlib.import_module(m)\n"
             "    except Exception as e: bad.append(f'{m}: {e!r}')\n"
             "import torch\n"
             "dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'\n"
             "print('python', sys.version.split()[0], '| torch', torch.__version__, 'on', dev)\n"
             "print('\\n'.join(bad)); sys.exit(1 if bad else 0)\n")
    p = run_quiet([py, "-c", check], env=child_env())
    if p.returncode != 0:
        die("some packages do not import (listed above)")
    marker.write_text(env_fingerprint() + "\n")
    say("environment ready")
    return py


# PyPI's Windows torch wheels are CPU-only; PyTorch's own index has the CUDA build of the same version.
# CUDA 13.2 matches the build Linux gets from PyPI (and needs an NVIDIA driver for CUDA >= 13.2).
TORCH_CUDA = ("2.15.0+cu132", "https://download.pytorch.org/whl/cu132", (13, 2))


def ensure_cuda_torch(py):
    """On Windows with an NVIDIA GPU, swap in the CUDA build of the pinned torch (EEGNet then trains
    on the GPU). Best effort: any problem leaves the CPU build in place and the run continues."""
    import re
    if not IS_WIN or not shutil.which("nvidia-smi"):
        return
    probe = "import torch; print(torch.__version__, torch.version.cuda or '', torch.cuda.is_available())"
    r = subprocess.run([str(py), "-c", probe], capture_output=True, text=True, env=child_env())
    if r.returncode == 0 and r.stdout.split()[-1] == "True":
        say(f"torch {r.stdout.split()[0]}: CUDA build, GPU available")
        return
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, errors="replace").stdout
    m = re.search(r"CUDA (?:UMD )?Version: (\d+)\.(\d+)", smi)
    need = TORCH_CUDA[2]
    if not m or (int(m.group(1)), int(m.group(2))) < need:
        say(f"WARNING  NVIDIA driver supports CUDA {m.group(1) + '.' + m.group(2) if m else '?'}; "
            f"the GPU build needs {need[0]}.{need[1]}+, so EEGNet trains on the CPU (update the driver to use the GPU)")
        return
    say(f"NVIDIA GPU found: installing torch {TORCH_CUDA[0]} (CUDA build, ~3 GB) so EEGNet trains on it")
    uv = find_uv()
    p = run_quiet([uv, "pip", "install", "--python", py, "--extra-index-url", TORCH_CUDA[1],
                   f"torch=={TORCH_CUDA[0]}"], env=dict(os.environ, UV_HTTP_TIMEOUT="300"))
    r = subprocess.run([str(py), "-c", probe], capture_output=True, text=True, env=child_env())
    if p.returncode == 0 and r.returncode == 0 and r.stdout.split()[-1] == "True":
        say(f"torch {r.stdout.split()[0]}: GPU available")
        return
    say("WARNING  the CUDA build of torch did not work; restoring the CPU build (EEGNet trains on the CPU)")
    run_quiet([uv, "pip", "install", "--python", py, "--reinstall-package", "torch", "-r", REQ])


def child_env(extra=None):
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(k, None)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
               MPLBACKEND="Agg")
    # Trust Mozilla's roots (certifi) in addition to the system store: Windows adds root CAs only on
    # demand, so e.g. TU Berlin's HARICA root can be missing. A user-set SSL_CERT_FILE wins.
    if ca_bundle() and "SSL_CERT_FILE" not in os.environ:
        env["SSL_CERT_FILE"] = ca_bundle()
    env.update(extra or {})
    return env


# --------------------------------------------------------------------------- checks after stages
def count(pattern, n):
    def check():
        found = len(list(REPO.glob(pattern)))
        return None if found == n else f"expected {n} files matching {pattern}, found {found}"
    return check


def table_rows():
    p = REPO / "synthesis/results/nback_features.csv"
    with open(p, "rb") as f:
        rows = sum(1 for _ in f) - 1
    return None if rows == 26137 else f"nback_features.csv has {rows} rows, expected 26137"


def events_ok(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    errs = [r for r in rows if "error" in r]
    if errs or len(rows) != 261:
        return f"{len(rows)} rows (expected 261), {len(errs)} subjects failed: {errs[:2]}"
    return None


# --------------------------------------------------------------------------- stages
class Stage:
    def __init__(self, name, cwd, argv, log, minutes, mode="w", check=None, stdout_to=None,
                 stdout_check=None, aux=False, retries=0, env=None, interpreter=True):
        self.name, self.cwd, self.argv, self.log, self.minutes = name, cwd, argv, log, minutes
        self.mode, self.check, self.stdout_to, self.stdout_check = mode, check, stdout_to, stdout_check
        self.aux, self.retries, self.env, self.interpreter = aux, retries, env or {}, interpreter


def stages(args, res):
    pytest = ["-m", "pytest", "-p", "no:cacheprovider", "tests"]
    S = [
        # unit tests first: an environment problem shows up in minutes, not hours
        Stage("tests:synthesis", "synthesis", pytest + ["-q"], ".pipeline/logs/tests_synthesis.txt", 1),
        Stage("tests:eeg-load", "eeg-load", pytest + ["-q"], ".pipeline/logs/tests_eeg-load.txt", 1),
        Stage("tests:analysis", "analysis", pytest + ["-q"], ".pipeline/logs/tests_analysis.txt", 1),
        # every download before any long computation
        Stage("fetch:cogbci-notebook", "synthesis", ["scripts/00_cogbci_notebook.py"],
              "synthesis/results/log_00_cogbci_notebook.txt", 1, retries=3),
        Stage("fetch:shin", "synthesis", ["scripts/01_fetch.py", "shin"],
              "synthesis/results/log_01_fetch_shin.txt", 30, retries=4,
              check=count("eeg-analysis/Cleaned Data/VP0*-EEG/*_nback.mat", 78)),
        Stage("fetch:cogbci", "synthesis", ["scripts/01_fetch.py", "cog", "--jobs", "6"],
              "synthesis/results/log_01_fetch_cog.txt", 90, retries=4,
              check=lambda: count("synthesis/data/cogbci/sub-*/ses-S*/eeg/*BACK.fdt", 261)()
              or count("synthesis/data/cogbci/sub-*/ses-S*/eeg/*BACK.set", 261)()),
        Stage("fetch:cogbci-events", "synthesis", ["scripts/cogbci_scan_events.py"],
              ".pipeline/logs/cogbci_scan_events.txt", 15, aux=True,
              stdout_to="synthesis/results/cogbci_nback_events.jsonl", stdout_check=events_ok),
        # synthesis: harmonize Shin + COG-BCI and build the joined table
        Stage("synthesis:02_harmonize_shin", "synthesis",
              ["scripts/02_harmonize.py", "shin", "--jobs", Jobs(lambda: res.jobs(1.6))],   # peak 1.2 GB/worker
              "synthesis/results/log_02_harmonize_shin.txt", 15,
              check=count("synthesis/cache/windows/shin-VP0*.npz", 26)),
        Stage("synthesis:02_harmonize_cog", "synthesis",
              ["scripts/02_harmonize.py", "cog", "--jobs", Jobs(lambda: res.jobs(3.2))],    # peak 2.6 GB/worker
              "synthesis/results/log_02_harmonize_cog.txt", 60,
              check=count("synthesis/cache/windows/cog-*.npz", 29)),
        Stage("synthesis:03_table", "synthesis", ["scripts/03_table.py"],
              "synthesis/results/log_03_table.txt", 3, check=table_rows),
        Stage("synthesis:04_viability", "synthesis", ["scripts/04_viability.py"],
              "synthesis/results/log_04_viability.txt", 5),
        Stage("synthesis:05_rlmd_pilot", "synthesis", ["scripts/05_rlmd_pilot.py"],
              "synthesis/results/log_05_rlmd_pilot.txt", 15),
        # analysis: split, features, models, evaluation
        Stage("analysis:01_split", "analysis", ["scripts/01_split.py"],
              "analysis/results/log_01_split.txt", 2),
        Stage("analysis:tests-split", "analysis", pytest + ["-q"],
              ".pipeline/logs/tests_analysis_split.txt", 1),
        Stage("analysis:02_rlmd_features", "analysis",
              ["scripts/02_rlmd_features.py", "--jobs", Jobs(lambda: res.jobs(1))],
              "analysis/results/log_02_rlmd_features.txt", 240,
              check=count("analysis/cache/rlmd/*.npz", 55)),
        Stage("analysis:05_models_part1", "analysis",
              ["scripts/05_models.py", "--models", "lr", "riemann", "svm", "mlp", "--jobs", Jobs(lambda: res.jobs(1.5))],
              "analysis/results/log_05_models_part1.txt", 180),
        Stage("analysis:05_models_part2", "analysis",
              ["scripts/05_models.py", "--models", "rf", "xgb", "--jobs", Jobs(lambda: max(1, min(res.cpus // 2, res.jobs(3))))],
              "analysis/results/log_05_models_part2.txt", 50),
        Stage("analysis:06_eegnet", "analysis", ["scripts/06_eegnet.py"],
              "analysis/results/log_06_eegnet.txt", 90),
        Stage("analysis:03_eda", "analysis", ["scripts/03_eda.py"], "analysis/results/log_03_eda.txt", 1),
    ]
    first = True
    for protocol in ("corrected", "leaky"):
        for task in ("bin", "shin3", "cog3"):
            S.append(Stage(f"analysis:04_baselines_{task}_{protocol}", "analysis",
                           ["scripts/04_baselines.py", "--task", task, "--protocol", protocol,
                            "--jobs", Jobs(lambda: res.jobs(1.5))],
                           "analysis/results/log_04_baselines.txt", 70, mode="w" if first else "a"))
            first = False
    S.append(Stage("analysis:07_evaluate", "analysis", ["scripts/07_evaluate.py"],
                   "analysis/results/log_07_evaluate.txt", 1))
    return S


# --------------------------------------------------------------------------- running
def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)


def run_stage(st, py, dry):
    argv = [str(a) for a in st.argv]                  # resolves Jobs(...) now, at stage start
    cmd = ([str(py)] if st.interpreter else []) + argv
    say(f"$ (cd {st.cwd}) {' '.join(cmd if not st.interpreter else ['python'] + argv)}")
    say(f"  log: {st.log}" + (f"   output: {st.stdout_to}" if st.stdout_to else ""))
    if dry:
        return None
    log = REPO / st.log
    log.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = None
    with open(log, st.mode, encoding="utf-8") as lf:
        if st.stdout_to:
            tmp_out = (REPO / st.stdout_to).with_suffix(".partial")
            out_f = open(tmp_out, "w", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=REPO / st.cwd, env=child_env(st.env), stdout=out_f,
                                 stderr=subprocess.PIPE)
            stream = p.stderr
        else:
            p = subprocess.Popen(cmd, cwd=REPO / st.cwd, env=child_env(st.env), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            stream = p.stdout
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", "replace")
                lf.write(line)
                lf.flush()
                OUT.write("  | " + line if line.endswith("\n") else "  | " + line + "\n")
            rc = p.wait()
        except KeyboardInterrupt:
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
            raise
        finally:
            if st.stdout_to:
                out_f.close()
    if rc != 0:
        return f"exit code {rc}"
    if st.stdout_to:
        problem = st.stdout_check(tmp_out) if st.stdout_check else None
        if problem:
            return f"output not complete ({problem}); {st.stdout_to} left unchanged"
        tmp_out.replace(REPO / st.stdout_to)
    return st.check() if st.check else None


def tail(path, n=25):
    try:
        return "".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(True)[-n:])
    except Exception:
        return ""


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list stages and their status, then exit")
    ap.add_argument("--setup-only", action="store_true", help="preflight, environment and tests only")
    ap.add_argument("--from", dest="start", metavar="STAGE", help="rerun STAGE and every stage after it")
    ap.add_argument("--only", nargs="+", metavar="STAGE", help="run just these stages (name or prefix)")
    ap.add_argument("--redo", action="store_true", help="rerun every stage, even ones recorded as done")
    ap.add_argument("--cpus", type=int, help="use at most this many CPU threads")
    ap.add_argument("--dry-run", action="store_true", help="print what would run")
    ap.add_argument("--skip-checks", action="store_true", help="continue despite platform/disk warnings")
    args = ap.parse_args()

    res = Resources(args.cpus)
    all_stages = stages(args, res)
    state = load_state()
    names = [s.name for s in all_stages]

    def match(key):
        hit = [n for n in names if n == key] or [n for n in names if n.startswith(key)]
        if not hit:
            sys.exit(f"no stage named {key!r}; see --list")
        return hit

    if args.list:
        total = 0
        for s in all_stages:
            done = state.get(s.name, {}).get("done")
            total += s.minutes
            print(f"  {'done ' + done[:16] if done else 'todo':22s} {s.name:42s} ~{fmt_dt(60 * s.minutes):>6s}"
                  f"{'  (optional)' if s.aux else ''}")
        print(f"\n  ~{fmt_dt(60 * total)} in total on an 8-core Apple M1 with 16 GB "
              f"(this machine: {res.cpus} threads, {res.ram:.0f} GB)")
        return

    import signal

    def on_term(signum, frame):        # `kill` / logout: stop the running stage cleanly, like Ctrl-C
        raise KeyboardInterrupt
    for sig in ("SIGTERM", "SIGHUP"):
        # keep a hang-up ignored when started under nohup, so a dropped SSH session doesn't stop it
        if hasattr(signal, sig) and signal.getsignal(getattr(signal, sig)) is not signal.SIG_IGN:
            signal.signal(getattr(signal, sig), on_term)

    t_start = time.time()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    OUT = Tee(LOGS / f"run_{stamp}.log")
    banner(f"run_pipeline.py  {datetime.datetime.now():%Y-%m-%d %H:%M}   (run log: {OUT.path.relative_to(REPO)})")

    if args.start and args.only:
        sys.exit("use either --from or --only, not both")
    if args.redo:
        state = {}
    selected = set(names)
    if args.start:              # STAGE and everything after it; earlier stages are not touched
        later = names[names.index(match(args.start)[0]):]
        selected = set(later)
        for n in later:
            state.pop(n, None)
    if args.only:
        selected = {n for k in args.only for n in match(k)}
        for n in selected:
            state.pop(n, None)
    if args.setup_only:
        selected = {n for n in names if n.startswith("tests:")}
        for n in selected:
            state.pop(n, None)
    todo = [s for s in all_stages if s.name in selected and not state.get(s.name, {}).get("done")]
    skipped = [s.name for s in all_stages if s.name in selected and s not in todo]

    if not args.dry_run:
        preflight(args, res, todo)
        how = keep_awake()
        say(f"sleep prevention: {how or 'not available - keep the machine awake (power settings)'}")
        py = ensure_env()
        ensure_cuda_torch(py)
        if any(s.name.startswith(NEEDS_NET) for s in todo):
            tls_check(py)
    else:
        py = venv_python()
    if skipped:
        say(f"\nalready done (recorded in {STATE.relative_to(REPO)}): {len(skipped)} stages; "
            f"use --from or --redo to rerun")
    est = sum(s.minutes for s in todo)
    say(f"to run: {len(todo)} stages, typically ~{fmt_dt(60 * est)} on an 8-core M1 "
        f"(here: {res.cpus} threads, {res.ram:.0f} GB RAM)")
    soft_fail = []
    for i, st in enumerate(todo, 1):
        banner(f"[{i}/{len(todo)}] {st.name}   (typically ~{fmt_dt(60 * st.minutes)}; started "
               f"{datetime.datetime.now():%H:%M}, {fmt_dt(time.time() - t_start)} into the run)")
        t0 = time.time()
        err = None
        for attempt in range(st.retries + 1):
            err = run_stage(st, py, args.dry_run)
            if err is None or attempt == st.retries:
                break
            wait = 60 * (attempt + 1)
            say(f"\n{st.name} failed ({err}); it resumes where it stopped, retrying in {wait} s "
                f"({attempt + 1}/{st.retries})")
            time.sleep(wait)
        if args.dry_run:
            continue
        if err:
            if st.aux:
                say(f"\nWARNING: optional stage {st.name} failed ({err}); continuing")
                soft_fail.append(st.name)
                continue
            say(f"\nlast lines of {st.log}:\n{tail(REPO / st.log)}")
            die(f"stage {st.name} failed: {err}.\n  Full output: {st.log}\n"
                f"  Run the same command again to resume from this stage.")
        state[st.name] = dict(done=datetime.datetime.now().isoformat(timespec="seconds"),
                              seconds=round(time.time() - t0))
        save_state(state)
        say(f"-> {st.name} done in {fmt_dt(time.time() - t0)}")

    if args.dry_run:
        return
    banner(f"Finished in {fmt_dt(time.time() - t_start)}")
    if args.setup_only:
        say("setup and tests passed. Run `python run_pipeline.py` for the full pipeline.")
        return
    if soft_fail:
        say(f"optional stages that failed (rerun with --only {' '.join(soft_fail)}): {', '.join(soft_fail)}")
    left = [n for n in names if not state.get(n, {}).get("done") and n not in soft_fail]
    if left:
        say(f"{len(left)} stages not yet run: {', '.join(left[:4])}{' ...' if len(left) > 4 else ''}")
        return
    stale = stale_outputs(state)
    if stale:
        say("WARNING: result files older than this pipeline run (not regenerated):")
        for p in stale:
            say(f"  {p}")
    else:
        say("every committed result file was regenerated by this run")
    say("all stages done. Results: synthesis/results/, analysis/results/")


NOT_REGENERATED = {  # committed files the pipeline deliberately does not rewrite (see README)
    "synthesis/results/shin_nback_signal.txt", "analysis/results/log_run_rest.txt",
}


def stale_outputs(state):
    """Result files not rewritten since the first stage of this pipeline ran."""
    times = [datetime.datetime.fromisoformat(v["done"]).timestamp() - v.get("seconds", 0)
             for v in state.values() if "done" in v]
    if not times:
        return []
    t0 = min(times)
    out = []
    for d in ("synthesis/results", "analysis/results"):
        for p in sorted((REPO / d).rglob("*")):
            rel = p.relative_to(REPO).as_posix()
            if p.is_file() and p.name != ".DS_Store" and rel not in NOT_REGENERATED and p.stat().st_mtime < t0:
                out.append(rel)
    return out


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if OUT:
            say("\nInterrupted. Run the same command again to resume from the unfinished stage.")
        sys.exit(130)
