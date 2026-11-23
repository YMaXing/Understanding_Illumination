"""tools/publish.py on throwaway repositories: a bare "GitHub" remote, the maintainer's repo, an annotator's clone."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M / "tools"))
import publish as PB  # noqa: E402

STEM = "adke_neel_026"


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True,
                          capture_output=True, text=True).stdout.strip()


def commit_all(repo, msg):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)


def tree(repo, ref):
    return set(git(repo, "ls-tree", "-r", "--name-only", ref).splitlines())


def add_pair(path, annotator, pid):
    f = json.loads(path.read_text())
    f["pairs"].append(dict(id=pid, image=f["image"], type="m", A=dict(x=100, y=100, r=3), B=dict(x=130, y=100, r=3),
                           confidence="sure", used_in_loss=False, overlay_seen=False, annotator=annotator,
                           created="2026-11-19T00:00:00.000Z", note="", schema_version="1.2"))
    path.write_text(json.dumps(f, indent=1))


@pytest.fixture()
def repos(tmp_path):
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    work.mkdir()
    git(work, "init", "-q", "-b", "main")
    (work / "README.md").write_text("project\n")
    (work / "illumination_edge").mkdir()
    (work / "illumination_edge" / "other.py").write_text("x = 1\n")
    commit_all(work, "main")
    git(work, "remote", "add", "origin", str(origin))
    git(work, "push", "-q", "origin", "main")
    git(work, "checkout", "-q", "-b", "study1")                       # unpublished local work
    (work / "study.txt").write_text("private study notes\n")
    (work / "illumination_edge" / "common.py").write_text("def load_rgb(p): pass\n")
    commit_all(work, "study")
    git(work, "checkout", "-q", "-b", "metrics-v1")
    d = work / "metrics" / "annotations"
    d.mkdir(parents=True)
    for n in (f"{STEM}.json", "annotators.json"):
        shutil.copy2(M / "annotations" / n, d / n)
    (work / "metrics" / "code.py").write_text("v = 1\n")
    commit_all(work, "metrics")
    git(work, "fetch", "-q", "origin")
    return origin, work, tmp_path


def test_publish_flow(repos):
    origin, work, tmp = repos
    c1 = PB.build(work, trailers=["Co-Authored-By: x"])
    t = tree(work, PB.BRANCH)
    assert "study.txt" not in t and {"README.md", "illumination_edge/other.py", PB.HOOK, "metrics/code.py",
                                     f"metrics/annotations/{STEM}.json"} <= t
    assert git(work, "rev-parse", f"{c1}^") == git(work, "rev-parse", "origin/main")
    assert git(work, "rev-list", "--count", c1) == "2"                 # main + one snapshot, no local history
    assert PB.build(work) == c1                                         # nothing new: no second commit
    git(work, "push", "-q", "origin", PB.BRANCH)

    her = tmp / "her"                                                   # annotator: clone, branch, annotate, "merged PR"
    subprocess.run(["git", "clone", "-q", "--branch", PB.BRANCH, str(origin), str(her)], check=True)
    add_pair(her / "metrics" / "annotations" / f"{STEM}.json", "annotator_b", "t=9999999999001")
    commit_all(her, "annotations")
    git(her, "push", "-q", "origin", PB.BRANCH)

    git(work, "fetch", "-q", "origin")
    (work / "metrics" / "code.py").write_text("v = 2\n")
    commit_all(work, "metrics v2")
    with pytest.raises(SystemExit, match="does not contain yet"):
        PB.build(work)                                                  # would overwrite her pair
    assert PB.pull(work, apply=True, into=work / "metrics" / "annotations") == 0
    commit_all(work, "annotations: merged")
    c3 = PB.build(work)
    f = json.loads(git(work, "show", f"{c3}:metrics/annotations/{STEM}.json"))
    assert "t=9999999999001" in {p["id"] for p in f["pairs"]}
    assert git(work, "show", f"{c3}:metrics/code.py") == "v = 2"
    git(work, "push", "-q", "origin", PB.BRANCH)

    git(her, "pull", "-q", "origin", PB.BRANCH)                         # a code change made on GitHub is refused
    (her / "metrics" / "code.py").write_text("v = 99\n")
    commit_all(her, "edit code on github")
    git(her, "push", "-q", "origin", PB.BRANCH)
    git(work, "fetch", "-q", "origin")
    with pytest.raises(SystemExit, match="changed on GitHub since the last snapshot"):
        PB.build(work)
