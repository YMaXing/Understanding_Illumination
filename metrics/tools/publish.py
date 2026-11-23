#!/usr/bin/env python3
"""Maintainer: publish the pair-annotation tool to GitHub on its own branch, without the rest of the local history.

    metrics/run.sh tools/publish.py pull [--apply]     # 1. bring annotations merged on GitHub into metrics/annotations
    metrics/tools/commit_metrics.sh --annotations "annotations: merged from metric-pair-annotation"   # 2. if 1 changed files
    metrics/run.sh tools/publish.py build              # 3. snapshot metrics-v1 onto the local metric-pair-annotation branch
    git push origin metric-pair-annotation             # 4. by hand, after reading what build printed

Local `metrics-v1` sits on top of unpublished study work, so it is never pushed. The published branch
`metric-pair-annotation` starts from origin/main and gets one snapshot commit per publish: its tree is the parent's
tree with `metrics/` replaced by metrics-v1's `metrics/` and `illumination_edge/common.py` (the loader and noise hooks)
taken from metrics-v1. Nothing else from the local history is included. Each snapshot carries the trailer
`Published-from: metrics-v1 <sha>`.

build refuses (nothing written) when:
  - GitHub's branch has annotation files that are not contained in metrics-v1 (an annotator's merged PR not pulled
    yet): run pull --apply and commit first, so a snapshot never overwrites anyone's pairs;
  - GitHub's branch changed anything under metrics/ other than metrics/annotations/, or illumination_edge/common.py,
    since the last snapshot (it would be overwritten);
  - the local metric-pair-annotation branch and GitHub's have diverged.
Only plumbing on a temporary index: HEAD, the real index and the working tree are untouched. Never pushes.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
sys.path.insert(0, str(M / "annotate"))
import merge as MR  # noqa: E402

BRANCH, SRC, REMOTE = "metric-pair-annotation", "metrics-v1", "origin"
HOOK = "illumination_edge/common.py"
ANN = "metrics/annotations/"
TRAILER = "Published-from: metrics-v1 "


class Git:
    def __init__(self, repo, index=None):
        self.repo, self.env = str(repo), dict(os.environ)
        if index:
            self.env["GIT_INDEX_FILE"] = index

    def run(self, *a, ok=False, inp=None, raw=False):
        r = subprocess.run(["git", "-C", self.repo, *a], capture_output=True, env=self.env,
                           input=inp.encode() if inp is not None else None)
        if r.returncode and not ok:
            raise SystemExit(f"git {' '.join(a)}: {r.stderr.decode().strip()}")
        if r.returncode:
            return None
        return r.stdout if raw else r.stdout.decode().strip()

    def rev(self, ref):
        return self.run("rev-parse", "--verify", "-q", f"{ref}^{{commit}}", ok=True)

    def blobs(self, rev, prefix):
        out = self.run("ls-tree", "-r", rev, "--", prefix, ok=True) or ""
        return {ln.split("\t", 1)[1]: ln.split()[2] for ln in out.splitlines()}

    def json(self, rev, path):
        return json.loads(self.run("show", f"{rev}:{path}"))


def remote_ref(g, remote):
    return g.rev(f"refs/remotes/{remote}/{BRANCH}")


def pick_parent(g, remote, base):
    """Parent of the next snapshot: the local branch if it is at or ahead of GitHub's, else GitHub's, else base."""
    loc, rem = g.rev(f"refs/heads/{BRANCH}"), remote_ref(g, remote)
    if loc and rem and loc != rem:
        if g.run("merge-base", "--is-ancestor", rem, loc, ok=True) is not None:
            return loc, loc
        if g.run("merge-base", "--is-ancestor", loc, rem, ok=True) is not None:
            return rem, loc
        raise SystemExit(f"local {BRANCH} and {remote}/{BRANCH} have diverged; push or reset the local branch first")
    p = loc or rem or g.rev(base)
    if not p:
        raise SystemExit(f"no {BRANCH} and no {base}: fetch first")
    return p, loc


def last_snapshot(g, parent):
    out = g.run("log", "--first-parent", "--format=%H%x00%B%x01", parent)
    for rec in out.split("\x01"):
        if "\x00" in rec:
            h, body = rec.strip().split("\x00", 1)
            if any(ln.startswith(TRAILER) for ln in body.splitlines()):
                return h
    return None


def same_content(a, b):
    """Annotation files equal up to order: pairs, tombstones and seed-log entries are compared by id / seed."""
    key = dict(pairs="id", deleted_pairs="id", s_seed_log="seed")
    norm = lambda f: {k: ({json.dumps(x, sort_keys=True) for x in v} if k in key else v) for k, v in f.items()}
    return norm(a) == norm(b)


def not_contained(g, parent, src, snap=None):
    """Annotation files changed on `parent` since the last snapshot `snap` (all of them if there is none) whose
    content metrics-v1 does not already contain (merging them in would change it). A file GitHub has not touched
    since the snapshot is metrics-v1's own earlier version: later local edits and deletions are not a conflict."""
    theirs, mine = g.blobs(parent, ANN), g.blobs(src, ANN)
    if snap:
        touched = set(g.run("diff", "--name-only", snap, parent, "--", ANN).splitlines())
        theirs = {p: b for p, b in theirs.items() if p in touched}
    bad = []
    for path, blob in sorted(theirs.items()):
        if mine.get(path) == blob:
            continue
        if path not in mine:
            bad.append(f"{path}: only on GitHub")
            continue
        t, m = g.json(parent, path), g.json(src, path)
        name = path[len(ANN):]
        if name == "annotators.json":
            mm = copy.deepcopy(m)
            add, conf = MR.merge_annotators(mm, t)
            if add or conf or mm.get("_changed"):
                bad.append(f"{path}: name tags differ ({len(add)} new, {len(conf)} conflicts)")
        elif name.startswith("relabels/"):
            have = {(e["pair_id"], e["relabeler"]): e for e in m["relabels"]}
            if any(have.get((e["pair_id"], e["relabeler"])) != e for e in t["relabels"]):
                bad.append(f"{path}: relabels not contained")
        elif "/" not in name and name.endswith(".json"):
            merged, rep = MR.merge_file(m, t)
            if rep["conflicts"] or not same_content(merged, m):
                bad.append(f"{path}: +{len(rep['added'])} entries, {len(rep['conflicts'])} conflicts")
        else:
            bad.append(f"{path}: differs")
    return bad


def build(repo, remote=REMOTE, base="origin/main", trailers=()):
    g = Git(repo)
    src = g.rev(SRC)
    if not src:
        raise SystemExit(f"no local {SRC} branch")
    parent, loc = pick_parent(g, remote, base)
    snap = last_snapshot(g, parent)
    if snap is None and g.blobs(parent, "metrics/"):
        raise SystemExit(f"{parent[:10]} already has metrics/ but no snapshot trailer: refusing to overwrite it")
    if snap:
        moved = [p for p in g.run("diff", "--name-only", snap, parent).splitlines()
                 if (p.startswith("metrics/") and not p.startswith(ANN)) or p == HOOK]
        if moved:
            raise SystemExit("changed on GitHub since the last snapshot (would be overwritten):\n  " + "\n  ".join(moved))
    bad = not_contained(g, parent, src, snap)
    if bad:
        raise SystemExit("annotations on GitHub that metrics-v1 does not contain yet:\n  " + "\n  ".join(bad) +
                         "\nRun: metrics/run.sh tools/publish.py pull --apply, then commit with "
                         "metrics/tools/commit_metrics.sh --annotations, then build again.")
    hook = g.run("ls-tree", src, "--", HOOK).split()
    if len(hook) < 3:
        raise SystemExit(f"{SRC} has no {HOOK}")
    with tempfile.TemporaryDirectory() as td:
        gi = Git(repo, index=str(Path(td) / "index"))
        gi.run("read-tree", parent)
        old = gi.run("ls-files", "-z", "--", "metrics", HOOK)
        if old:
            gi.run("update-index", "--force-remove", "-z", "--stdin", inp=old if old.endswith("\0") else old + "\0")
        gi.run("read-tree", "--prefix=metrics/", f"{src}:metrics")
        gi.run("update-index", "--add", "--cacheinfo", f"{hook[0]},{hook[2]},{HOOK}")
        tree = gi.run("write-tree")
    if tree == g.run("rev-parse", f"{parent}^{{tree}}"):
        print(f"nothing to publish: {BRANCH} already equals metrics-v1 {src[:10]}")
        return parent
    msg = (f"Publish the pair-annotation tool from metrics-v1 {src[:10]}\n\n"
           f"metrics/ and {HOOK} only (a snapshot; the local history behind metrics-v1 is not included).\n\n"
           f"{TRAILER}{src}\n" + "".join(f"{t}\n" for t in trailers))
    commit = g.run("commit-tree", tree, "-p", parent, inp=msg)
    g.run("update-ref", f"refs/heads/{BRANCH}", commit, loc or "")
    print(f"{BRANCH} -> {commit[:10]} (parent {parent[:10]}, snapshot of metrics-v1 {src[:10]})")
    print(g.run("diff", "--stat=120", parent, commit).splitlines()[-1])
    outside = [p for p in g.run("diff", "--name-only", parent, commit).splitlines() if not p.startswith("metrics/") and p != HOOK]
    assert not outside, outside
    print(f"files outside metrics/ in the branch that differ from {base}: " +
          (", ".join(p for p in g.run("diff", "--name-only", g.rev(base) or parent, commit).splitlines()
                     if not p.startswith("metrics/")) or "none"))
    print(f"to publish: git push {remote} {BRANCH}")
    return commit


def pull(repo, apply, remote=REMOTE, into=None):
    g = Git(repo)
    rem = remote_ref(g, remote)
    if not rem:
        print(f"{remote}/{BRANCH} does not exist yet: nothing to pull")
        return 0
    with tempfile.TemporaryDirectory() as td:
        arc = Path(td) / "a.tar"
        arc.write_bytes(g.run("archive", "--format=tar", rem, "--", "metrics/annotations", raw=True))
        out = Path(td) / "x"
        with tarfile.open(arc) as t:
            t.extractall(out, filter="data")
        argv = ["merge.py", "--from", str(out / "metrics" / "annotations")] + (["--into", str(into)] if into else []) + \
               (["--apply"] if apply else [])
        old, sys.argv = sys.argv, argv
        try:
            return MR.main()
        finally:
            sys.argv = old


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(M.parent))
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--no-fetch", action="store_true", help="do not run git fetch first")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--apply", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--base", default="origin/main")
    b.add_argument("--trailer", action="append", default=[])
    a = ap.parse_args()
    if not a.no_fetch:
        Git(a.repo).run("fetch", a.remote)
    if a.cmd == "pull":
        return pull(a.repo, a.apply, a.remote)
    build(a.repo, a.remote, a.base, a.trailer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
