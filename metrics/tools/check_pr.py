#!/usr/bin/env python3
"""Pre-merge check of an annotator's branch against the published branch (read-only: only `git show` / `git diff`).

    git fetch origin
    metrics/run.sh tools/check_pr.py origin/annotations/NAME [--base origin/metric-pair-annotation] [--annotator ID] [--repo PATH]

Compares BRANCH with its merge base on --base. Errors (exit code 1):
  - a changed file other than: metrics/annotations/<stem>.json of an image assigned to the branch's annotator,
    metrics/annotations/annotators.json, metrics/annotations/relabels/<stem>.json;
  - a changed annotation file that fails the schema, or whose image identity changed (image, set, split, assigned_to,
    size, image_sha256, image_pixel_sha256 differ from the base);
  - a new pair whose annotator is not the branch annotator; an edited pair whose last_edited_by is not; a pre-existing
    pair whose annotator or created changed; a base pair that vanished without a tombstone;
  - a tombstoned pair resurrected, or a base tombstone removed;
  - s_seed_log not append-only, or a new seed proposed by someone else;
  - relabels: an existing entry changed or removed, or a new entry by someone else;
  - annotators.json: an id removed, an id's created changed, a name changed without a matching history entry, or a
    file that fails its schema.
The branch annotator is --annotator, or else the single assigned_to of the changed annotation files. Warnings (no exit
code): edits or deletions of another annotator's pairs, pending seeds. Then per-image progress against the targets
(exhausted marks count as met; targeted s is maintainer-only and has no target) for every image assigned to the
branch annotator, in annotation order.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
from core import config as K  # noqa: E402
from core import schema as SC  # noqa: E402

ANN_RE = re.compile(r"^metrics/annotations/([A-Za-z0-9_.-]+)\.json$")
REL_RE = re.compile(r"^metrics/annotations/relabels/([A-Za-z0-9_.-]+)\.json$")
IDENTITY = ("image", "set", "split", "assigned_to", "width", "height", "image_sha256", "image_pixel_sha256", "schema_version")


class Git:
    def __init__(self, repo):
        self.repo = str(repo)

    def run(self, *a, ok=False):
        r = subprocess.run(["git", "-C", self.repo, *a], capture_output=True, text=True)
        if r.returncode and not ok:
            raise SystemExit(f"git {' '.join(a)}: {r.stderr.strip()}")
        return r.stdout if not r.returncode else None

    def show(self, rev, path):
        out = self.run("show", f"{rev}:{path}", ok=True)
        return json.loads(out) if out is not None else None


def check(git, branch, base, annotator, cfg):
    errors, warns = [], []
    mb = git.run("merge-base", base, branch).strip()
    changed = [ln.split("\t") for ln in git.run("diff", "--name-status", mb, branch).splitlines() if ln.strip()]
    paths = {c[-1]: c[0] for c in changed}
    ann_changed = {ANN_RE.match(p).group(1): p for p in paths if ANN_RE.match(p) and not p.endswith("/annotators.json")}
    if annotator is None:
        owners = set()
        for stem, p in ann_changed.items():
            f = git.show(mb, p) or git.show(branch, p) or {}
            owners.add(f.get("assigned_to"))
        if len(owners) != 1:
            errors.append(f"cannot tell the branch annotator from the changed files (assigned_to: {sorted(map(str, owners))}); pass --annotator")
            return errors, warns, None, mb
        annotator = owners.pop()
    for p, st in paths.items():
        if p == "metrics/annotations/annotators.json" or REL_RE.match(p):
            continue
        m = ANN_RE.match(p)
        if not m:
            errors.append(f"{p}: not an annotation file (only annotation files, annotators.json and relabels may change)")
            continue
        base_f, new_f = git.show(mb, p), git.show(branch, p)
        if base_f is None:
            errors.append(f"{p}: new annotation file (images are created by the maintainer)")
            continue
        if new_f is None:
            errors.append(f"{p}: annotation file deleted")
            continue
        if base_f.get("assigned_to") != annotator:
            errors.append(f"{p}: image assigned to {base_f.get('assigned_to')!r}, not to {annotator!r}")
        for e in SC.validate_file(new_f):
            errors.append(f"{p}: schema: {e}")
        for k in IDENTITY:
            if base_f.get(k) != new_f.get(k):
                errors.append(f"{p}: {k} changed ({base_f.get(k)!r} -> {new_f.get(k)!r})")
        bp = {q["id"]: q for q in base_f["pairs"]}
        bt = {t["id"]: t for t in base_f.get("deleted_pairs", [])}
        np_ = {q["id"]: q for q in new_f["pairs"]}
        nt = {t["id"]: t for t in new_f.get("deleted_pairs", [])}
        for i in bt:
            if i in np_:
                errors.append(f"{p}: {i}: tombstoned pair resurrected")
            if i not in nt:
                errors.append(f"{p}: {i}: tombstone removed")
        for i, q in np_.items():
            if i in bt:
                continue
            if i not in bp:
                if q["annotator"] != annotator:
                    errors.append(f"{p}: new pair {i} has annotator {q['annotator']!r}, not {annotator!r}")
            elif q != bp[i]:
                b = bp[i]
                if (q["annotator"], q["created"]) != (b["annotator"], b["created"]):
                    errors.append(f"{p}: pair {i} changed authorship ({b['annotator']} -> {q['annotator']})")
                if q.get("last_edited_by") != annotator:
                    errors.append(f"{p}: edited pair {i} has last_edited_by {q.get('last_edited_by')!r}, not {annotator!r}")
                if b["annotator"] != annotator:
                    warns.append(f"{p}: pair {i} by {b['annotator']} was edited")
        for i, b in bp.items():
            if i not in np_:
                if i not in nt:
                    errors.append(f"{p}: pair {i} removed without a tombstone")
                else:
                    if nt[i]["deleted_by"] != annotator:
                        errors.append(f"{p}: pair {i} deleted by {nt[i]['deleted_by']!r}, not {annotator!r}")
                    if b["annotator"] != annotator:
                        warns.append(f"{p}: pair {i} by {b['annotator']} was deleted")
        ol, nl = base_f["s_seed_log"], new_f["s_seed_log"]
        if len(nl) < len(ol) or any({k: a.get(k) for k in ("index", "seed", "x", "y", "proposed_at", "proposed_by")} !=
                                    {k: b.get(k) for k in ("index", "seed", "x", "y", "proposed_at", "proposed_by")} for a, b in zip(ol, nl)):
            errors.append(f"{p}: s_seed_log is not append-only")
        for e in nl[len(ol):]:
            if e["proposed_by"] != annotator:
                errors.append(f"{p}: seed {e['index']} proposed by {e['proposed_by']!r}, not {annotator!r}")
        if any(e["outcome"] == "pending" for e in nl):
            warns.append(f"{p}: a random-s seed is still pending (reopen the image and resolve it)")
    for p in paths:
        if not REL_RE.match(p):
            continue
        b, n = git.show(mb, p), git.show(branch, p)
        if n is None:
            errors.append(f"{p}: relabels file deleted")
            continue
        errors += [f"{p}: schema: {e}" for e in SC.validate(n, SC.load_schema(M / "schema" / "relabels.schema.json"))]
        old = (b or {}).get("relabels", [])
        if n["relabels"][:len(old)] != old:
            errors.append(f"{p}: existing relabels changed or removed")
        for e in n["relabels"][len(old):]:
            if e["relabeler"] != annotator:
                errors.append(f"{p}: relabel of {e['pair_id']} by {e['relabeler']!r}, not {annotator!r}")
    ap = "metrics/annotations/annotators.json"
    if ap in paths:
        b, n = git.show(mb, ap), git.show(branch, ap)
        errors += [f"annotators.json: schema: {e}" for e in SC.validate(n, SC.load_schema(M / "schema" / "annotators.schema.json"))]
        bi, ni = {a["id"]: a for a in b["annotators"]}, {a["id"]: a for a in n["annotators"]}
        for i, a in bi.items():
            if i not in ni:
                errors.append(f"annotators.json: id {i!r} removed")
                continue
            c = ni[i]
            if c["created"] != a["created"]:
                errors.append(f"annotators.json: {i}: created changed")
            if c["history"][:len(a["history"])] != a["history"]:
                errors.append(f"annotators.json: {i}: rename history rewritten")
            if c["name"] != a["name"] and not (c["history"] and c["history"][-1]["to"] == c["name"] and len(c["history"]) > len(a["history"])):
                errors.append(f"annotators.json: {i}: name changed without a rename record")
    return errors, warns, annotator, mb


def progress(git, branch, annotator, cfg):
    T = cfg["annotate"]["targets"]
    order = cfg["annotate"]["image_order"]
    rows = []
    for stem in K.all_stems(cfg):
        f = git.show(branch, f"metrics/annotations/{stem}.json")
        if not f or f.get("assigned_to") != annotator:
            continue
        P, X = f["pairs"], f.get("exhausted") or {}
        n = lambda fn: sum(1 for q in P if fn(q))
        cnt = dict(i=n(lambda q: q["type"] == "i"), m=n(lambda q: q["type"] == "m"), b=n(lambda q: q["type"] == "b"),
                   s_random=n(lambda q: q["type"] == "s" and q.get("pool") == "random"), d=n(lambda q: q["type"] == "d"))
        dsm = n(lambda q: q["type"] == "d" and q.get("material") == "same" and q.get("illumination") == "different")
        lo = dict(i=T["i"], m=T["m"], b=T["b"], s_random=T["s_random"][0], d=T["d"][0])
        met = {k: (k in X) or cnt[k] >= lo[k] for k in cnt}
        share_ok = cnt["d"] == 0 and "d" in X or (cnt["d"] > 0 and dsm / cnt["d"] >= T["d_same_material_diff_illum_min_frac"])
        rows.append((order.index(f["split"]), stem, f["split"], cnt, lo, met, X, dsm, share_ok))
    rows.sort()
    print(f"\nprogress for {annotator} ({len(rows)} images; target met = reached or marked exhausted)")
    print(f"{'image':22s} {'split':5s} " + " ".join(f"{k:>11s}" for k in ("i", "m", "b", "s_random", "d")) + "  d same/diff  done")
    done = 0
    for _, stem, sp, cnt, lo, met, X, dsm, share_ok in rows:
        cell = lambda k: (f"{cnt[k]}/{lo[k]}" + ("x" if k in X else "") + ("*" if met[k] else " "))
        ok = all(met.values()) and share_ok
        done += ok
        print(f"{stem:22s} {sp:5s} " + " ".join(f"{cell(k):>11s}" for k in ("i", "m", "b", "s_random", "d")) +
              f"  {dsm:>4d}/{cnt['d']:<4d}  {'yes' if ok else ''}")
    print(f"* = met, x = marked exhausted. Images complete: {done} of {len(rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("branch")
    ap.add_argument("--base", default="origin/metric-pair-annotation")
    ap.add_argument("--annotator")
    ap.add_argument("--repo", default=str(K.REPO))
    a = ap.parse_args()
    cfg = K.load_config()
    git = Git(a.repo)
    errors, warns, who, mb = check(git, a.branch, a.base, a.annotator, cfg)
    print(f"check_pr: {a.branch} vs {a.base} (merge base {mb[:10]}), annotator {who}")
    for w in warns:
        print(f"  warning: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    if who:
        progress(git, a.branch, who, cfg)
    print(f"\n{'FAILED' if errors else 'OK'}: {len(errors)} error(s), {len(warns)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
