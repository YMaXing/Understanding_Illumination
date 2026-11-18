"""tools/check_pr.py on a throwaway git repository built from the current annotation files."""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M / "tools"))
import check_pr as CP  # noqa: E402
from core import config as K  # noqa: E402

HER, MINE = "adke_neel_026", "taiwei_cui_080"


def git(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True)


def commit(repo, msg):
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)


def write(repo, stem, obj):
    (repo / "metrics" / "annotations" / f"{stem}.json").write_text(json.dumps(obj, indent=1))


def read(repo, stem):
    return json.loads((repo / "metrics" / "annotations" / f"{stem}.json").read_text())


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    d = r / "metrics" / "annotations"
    d.mkdir(parents=True)
    for n in (f"{HER}.json", f"{MINE}.json", "annotators.json"):
        shutil.copy2(M / "annotations" / n, d / n)
    f = read(r, MINE)                                          # give the base a tombstone
    t = f["pairs"].pop(0)
    f["deleted_pairs"] = [dict(id=t["id"], deleted_at="2026-11-18T00:00:00Z", deleted_by="yueren", pair=t)]
    write(r, MINE, f)
    git(r, "init", "-q", "-b", "metrics-v1")
    commit(r, "base")
    git(r, "checkout", "-q", "-b", "her")
    return r


def new_pair(stem, annotator, pid="t=9999999999001"):
    return dict(id=pid, image=stem, type="m", A=dict(x=100, y=100, r=3), B=dict(x=130, y=100, r=3), confidence="sure",
                used_in_loss=False, overlay_seen=False, annotator=annotator, created="2026-11-19T00:00:00.000Z", note="",
                schema_version="1.2")


def test_good_branch(repo):
    f = read(repo, HER)
    f["pairs"].append(new_pair(HER, "annotator_b"))
    f["exhausted"] = {"b": dict(by="annotator_b", at="2026-11-19T00:00:00Z", note="no b edges")}
    write(repo, HER, f)
    a = json.loads((repo / "metrics/annotations/annotators.json").read_text())
    e = next(x for x in a["annotators"] if x["id"] == "annotator_b")
    e["history"].append({"from": e["name"], "to": "Ana L", "at": "2026-11-19T00:00:00Z", "by": "annotator_b"})
    e["name"] = "Ana L"
    (repo / "metrics/annotations/annotators.json").write_text(json.dumps(a, indent=1))
    rel = repo / "metrics/annotations/relabels"
    rel.mkdir()
    pid = read(repo, MINE)["pairs"][0]["id"]
    (rel / f"{MINE}.json").write_text(json.dumps(dict(schema_version="1.2", image=MINE, relabels=[
        dict(pair_id=pid, relabeler="annotator_b", at="2026-11-19T00:00:00Z", type="m", confidence="sure", own_pair=False)])))
    commit(repo, "her batch")
    errors, warns, who, _ = CP.check(CP.Git(repo), "her", "metrics-v1", None, K.load_config())
    assert errors == [] and who == "annotator_b"


def test_bad_branch(repo):
    f = read(repo, HER)
    f["pairs"].append(new_pair(HER, "yueren"))                           # someone else's id on a new pair
    f["split"] = "train" if f["split"] != "train" else "val"             # identity change
    write(repo, HER, f)
    g = read(repo, MINE)                                                 # not her image; resurrects a tombstone
    g["pairs"].append(copy.deepcopy(g["deleted_pairs"][0]["pair"]))
    g["pairs"][1]["annotator"] = "annotator_b"                           # authorship change
    write(repo, MINE, g)
    (repo / "metrics" / "notes.txt").write_text("x")                     # not an annotation file
    a = json.loads((repo / "metrics/annotations/annotators.json").read_text())
    a["annotators"] = [x for x in a["annotators"] if x["id"] != "yueren"]  # id removed
    (repo / "metrics/annotations/annotators.json").write_text(json.dumps(a, indent=1))
    commit(repo, "bad")
    errors, _, _, _ = CP.check(CP.Git(repo), "her", "metrics-v1", "annotator_b", K.load_config())
    txt = "\n".join(errors)
    for needle in ("notes.txt: not an annotation file", f"{MINE}.json: image assigned to 'yueren'", "split changed",
                   "has annotator 'yueren', not 'annotator_b'", "tombstoned pair resurrected", "changed authorship",
                   "id 'yueren' removed"):
        assert needle in txt, needle
    r = subprocess.run([sys.executable, "-B", str(M / "tools" / "check_pr.py"), "her", "--repo", str(repo), "--annotator", "annotator_b", "--base", "metrics-v1"],
                       cwd=M, capture_output=True, text=True)
    assert r.returncode == 1 and "FAILED" in r.stdout
