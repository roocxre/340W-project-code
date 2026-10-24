"""Per-dataset readers that return the same structure: a list of runs per session.

A ``Run`` is one continuous recording at its native rate with its native EEG channels,
plus its n-back blocks. Shin has one run per session (the three sessions of
``cnt_nback.mat`` are split at the boundaries eegrepro recovers). COG-BCI has three
runs per session, one per level file.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np

from . import sources
from .remote import fetch_member, open_zip

sys.path.insert(0, str(sources.REPO / "eeg-analysis"))
from eegrepro import data as shin_data  # noqa: E402  (validated loader from the Shin reproduction)

mne.set_log_level("ERROR")


@dataclass
class Block:
    n: int
    onsets: np.ndarray            # trial onsets, samples at the run's native rate
    is_target: np.ndarray         # bool per trial (all False where the dataset does not code it)
    soa: float                    # median trial onset interval, s
    accuracy: float               # fraction of scorable trials answered correctly
    mean_rt: float                # s, correct target responses (COG-BCI) / correct trials (Shin)


@dataclass
class Run:
    dataset: str
    subject: str
    session: int
    raw: mne.io.BaseRaw           # native rate, native EEG channels, volts
    blocks: list[Block] = field(default_factory=list)
    label: str = ""               # file / segment name


# --------------------------------------------------------------------------
# Shin 2018
# --------------------------------------------------------------------------
SHIN_TARGET_CODES = {48, 80}       # 2-back and 3-back targets; 0-back trials all carry 16


def shin_session_runs(subject: str) -> list[list[Run]]:
    rec = shin_data.load_raw(subject, "nback")
    trials = shin_data.trial_table(rec)          # raises if markers and behaviour disagree
    beh = shin_data.load_behavior(subject, "nback")
    eeg = rec.channels(shin_data.EEG_CLAB)
    eog = rec.channels(shin_data.EOG_CLAB)
    code = rec.mrk_code[~np.isin(rec.mrk_code, list(shin_data.NBACK_SERIES))]
    out = []
    for k, (a, b) in enumerate(rec.sessions):
        info = mne.create_info(shin_data.EEG_CLAB + shin_data.EOG_CLAB, rec.fs,
                               ["eeg"] * len(eeg) + ["eog"] * len(eog))
        raw = mne.io.RawArray(rec.x[a:b][:, np.r_[eeg, eog]].T * 1e-6, info, verbose=False)
        run = Run("shin", f"shin-{subject}", k + 1, raw, label=f"{subject}_S{k + 1}")
        in_ses = (trials.sample >= a) & (trials.sample < b)
        for s in np.unique(trials.series[in_ses]):
            m = trials.series == s
            n = int(trials.nback[m][0])
            onsets = trials.sample[m] - a
            res = beh["result"][m]
            scorable = np.arange(m.sum()) >= n      # the first n trials have no correct answer
            rt = beh["reaction_time"][m]
            run.blocks.append(Block(
                n=n, onsets=onsets, is_target=np.isin(code[m], list(SHIN_TARGET_CODES)),
                soa=float(np.median(np.diff(onsets))) / rec.fs,
                accuracy=float(np.mean(res[scorable] == 1)),
                mean_rt=float(np.mean(rt[scorable & (res == 1)])) if np.any(scorable & (res == 1)) else np.nan))
        out.append([run])
    return out


# --------------------------------------------------------------------------
# COG-BCI v4
# --------------------------------------------------------------------------
def cog_fetch(subject: int, dest: Path) -> Path:
    """Fetch one subject's n-back .set/.fdt files (18 files) from the v4 archive into
    dest/sub-XX/ses-SY/eeg/. Archives 3-29 nest an extra sub-XX/ folder; 1-2 do not."""
    url = sources.COG_URL.format(subject)
    z = open_zip(url)
    members = [i for i in z.infolist()
               if Path(i.filename).stem in sources.COG_TASKS and i.filename.endswith((".set", ".fdt"))]
    assert len(members) == 18, (subject, len(members))
    root = dest / f"sub-{subject:02d}"
    for info in members:
        p = Path(info.filename)
        fetch_member(url, z, info, root / p.parts[-3] / "eeg" / p.name)
    return root


def _cog_blocks(raw: mne.io.BaseRaw, n: int) -> list[Block]:
    on = raw.annotations.onset
    de = np.asarray(raw.annotations.description)
    fs = raw.info["sfreq"]
    p = f"6{n}"
    is_trial = np.isin(de, [p + "21", p + "22", p + "23"])
    t_on, t_hit = on[is_trial], de[is_trial] == p + "22"
    # blocks: trial onsets separated by more than 5 s (within-block SOA <= 2.52 s, between
    # blocks 8.8-15 s; the 0-back start marker is duplicated, so block markers are not used)
    cut = np.where(np.diff(t_on) > 5)[0] + 1
    blocks = []
    for idx in np.split(np.arange(len(t_on)), cut):
        t0, t1 = t_on[idx[0]], t_on[idx[-1]]
        soa = float(np.median(np.diff(t_on[idx])))
        end = t1 + soa
        in_blk = (on >= t0) & (on < end)
        correct = (de == p + "32") & in_blk                       # response to a hit
        false_alarm = np.isin(de, [p + "31", p + "33"]) & in_blk  # error / conflict error
        hits = t_on[idx][t_hit[idx]]
        rts = []
        for t in on[correct]:
            prev = hits[hits <= t]
            if len(prev):
                rts.append(t - prev[-1])
        n_tr, n_hit = len(idx), int(t_hit[idx].sum())
        acc = (correct.sum() + (n_tr - n_hit - false_alarm.sum())) / n_tr
        blocks.append(Block(n=n, onsets=np.round(t_on[idx] * fs).astype(int), is_target=t_hit[idx],
                            soa=soa, accuracy=float(acc), mean_rt=float(np.mean(rts)) if rts else np.nan))
    return blocks


def cog_session_runs(subject: int, root: Path):
    """Yield one session (three runs) at a time to bound memory."""
    for k, ses in enumerate(sources.COG_SESSIONS):
        runs = []
        for task, n in sources.COG_TASKS.items():
            raw = mne.io.read_raw_eeglab(root / ses / "eeg" / f"{task}.set", preload=True)
            raw.pick("eeg")                                   # drops ECG1 (the TP9 slot)
            run = Run("cog", f"cog-{subject:02d}", k + 1, raw, label=f"sub-{subject:02d}_{ses}_{task}")
            run.blocks = _cog_blocks(raw, n)
            if [len(b.onsets) for b in run.blocks] != [48, 48, 48]:
                raise ValueError(f"{run.label}: unexpected block structure")
            runs.append(run)
        yield runs


def cog_subject(subject: int):
    """Load and yield sessions from data/cogbci/, fetching the subject first if needed."""
    yield from cog_session_runs(subject, cog_fetch(subject, sources.COG_DIR))
