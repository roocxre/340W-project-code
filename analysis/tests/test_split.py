import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import nbacksplit as ns  # noqa: E402

split_mod = __import__("01_split")


def test_allocate_is_70_20_10_in_whole_units():
    assert list(split_mod.allocate(26)) == [18, 5, 3]
    assert list(split_mod.allocate(20)) == [14, 4, 2]
    assert list(split_mod.allocate(9)) == [6, 2, 1]
    assert all(split_mod.allocate(n).sum() == n for n in range(1, 60))


def test_assign_keeps_units_whole_and_strata_balanced():
    df = pd.DataFrame({"subject": np.repeat([f"s{i}" for i in range(20)], 5),
                       "dataset": np.repeat(["a"] * 10 + ["b"] * 10, 5)})
    s = split_mod.assign(df, ["subject"], ["dataset"], seed=1)
    assert (s.groupby(df.subject).nunique() == 1).all()
    first = df.drop_duplicates("subject")
    per = pd.crosstab(first.dataset, s[first.index]).reindex(columns=list(split_mod.SPLITS))
    assert per.loc["a"].tolist() == per.loc["b"].tolist() == [7, 2, 1]


@pytest.mark.skipif(not (ns.DATA / "train.csv").exists(), reason="run scripts/01_split.py first")
def test_written_split():
    parts = {s: pd.read_csv(ns.DATA / f"{s}.csv", usecols=["row_id", "subject", "dataset", "n"]) for s in ns.SPLITS}
    subjects = [set(p.subject) for p in parts.values()]
    assert not (subjects[0] & subjects[1] or subjects[0] & subjects[2] or subjects[1] & subjects[2])
    rows = pd.concat(parts.values()).row_id
    assert rows.is_unique and len(rows) == 26137 and rows.min() == 0 and rows.max() == 26136
    for p in parts.values():
        assert set(p.n) == {0, 1, 2, 3} and set(p.dataset) == {"shin", "cog"}
    manifest = json.loads((ns.DATA / "split_manifest.json").read_text())
    assert {s: len(v["subjects"]) for s, v in manifest["splits"].items()} == {"train": 38, "test": 11, "val": 6}
