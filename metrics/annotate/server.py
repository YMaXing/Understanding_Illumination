#!/usr/bin/env python3
"""Annotation tool server (Python standard library only). Serves the tool, the prepared image cache and the
per-image annotation files; every write is schema-validated, atomic, and backed up first.

    python3 metrics/annotate/server.py [--port 8765] [--host 127.0.0.1] [--annotations DIR] [--cache DIR]
    (or: metrics/annotate/run.sh, which prepares missing images first and then starts this)

Reads only <cache>/manifest.json (written by prepare.py), the annotation files and annotators.json. It has no route
to predictions or metric runs: the annotation tool never loads model output.

API (JSON):
  GET  /api/manifest                        images, constants
  GET  /api/images                          per image: set, split, pair counts by type and annotator
  GET  /api/img/<stem>/<file>               raw.u16.gz | lI.f32.gz | valid.u8.gz | meta.json | illum.png | material.png | geom.png
  GET  /api/ann/<stem>                      the annotation file; ETag = SHA-256 of its bytes
  PUT  /api/ann/<stem>   (If-Match: etag)   replace the file: 409 if it changed meanwhile, 400 with the errors if invalid
  GET  /api/annotators                      name tags [{id, name, created, history}] (stable ids; pairs store the id)
  POST /api/annotators {"name"}             add a tag (trimmed; an existing name, case-insensitive, returns that tag)
  POST /api/annotators/rename {"id","name","by"}   rename a tag: annotators.json only (no annotation file changes);
                                            refuses empty names and case-insensitive collisions with another tag
  GET  /api/annotators/<id>/usage           how many pairs (and images) carry the tag
  POST /api/seed/<stem>  {"annotator"} (If-Match)   propose a random-s seed: a uniform valid pixel from
                                            random.Random(f"{seed_base}:{stem}:{index}"), appended to s_seed_log
Integrity rules on PUT (besides the schema): image / set / schema_version / size fixed; s_seed_log append-only (its
index, seed, x, y, proposed_* never change); an existing pair id keeps its annotator and created; ids unique across
pairs and deleted_pairs.
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import mimetypes
import os
import random
import re
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import annotators as AN  # noqa: E402  (stdlib only)
from core import schema as SC  # noqa: E402  (stdlib only)
import npyread  # noqa: E402  (stdlib only)

RELABEL_SCHEMA = SC.load_schema(HERE.parent / "schema" / "relabels.schema.json")

STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/lib.js": "lib.js",
          "/style.css": "style.css"}
REVIEW_STATIC = {"/review.html": "review.html", "/review.js": "review.js"}      # only with --review
IMG_FILES = {"raw.u16.gz", "lI.f32.gz", "valid.u8.gz", "meta.json", "illum.png", "material.png", "geom.png"}
SEED_IMMUTABLE = ("index", "seed", "x", "y", "proposed_at", "proposed_by")
LOCK = threading.Lock()


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def etag_of(b):
    return hashlib.sha256(b).hexdigest()


class Store:
    def __init__(self, manifest_path, annotations=None, review=None):
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.cache = Path(manifest_path).parent
        self.ann = Path(annotations or self.manifest["annotations"])
        self.backups = Path(self.manifest["backups"]) if annotations is None else self.ann / ".backups"
        self.relabels = Path(self.manifest["relabels"]) if annotations is None else self.ann / "relabels"
        self.images = {m["stem"]: m for m in self.manifest["images"]}
        self.review = Path(review) if review else None          # runs folder; None = review routes disabled
        self._runs = {}

    # ---------------------------------------------------------------- blind relabels (separate files, append-only)
    def relabel_path(self, stem):
        return self.relabels / f"{stem}.json"

    def read_relabels(self, stem):
        p = self.relabel_path(stem)
        if not p.exists():
            obj = dict(schema_version=self.manifest["schema_version"], image=stem, relabels=[])
            return obj, "none"
        b = p.read_bytes()
        return json.loads(b), etag_of(b)

    def relabel_queue(self, stem, who, own):
        """Geometry only (no type, labels or annotator), in a fixed per-relabeler shuffle; skips done pairs."""
        obj, _ = self.read(stem)
        rl, _ = self.read_relabels(stem)
        done = {e["pair_id"] for e in rl["relabels"] if e["relabeler"] == who}
        cand = [p for p in obj["pairs"] if p["id"] not in done and (own or p["annotator"] != who)]
        random.Random(f"relabel:{stem}:{who}").shuffle(cand)
        total = sum(1 for p in obj["pairs"] if own or p["annotator"] != who)
        return dict(queue=[dict(pair_id=p["id"], A=p["A"], B=p["B"]) for p in cand], total=total, done=total - len(cand))

    def add_relabel(self, stem, entry, if_match):
        with LOCK:
            obj, et = self.read_relabels(stem)
            if if_match != et:
                return None, et, obj
            ann, _ = self.read(stem)
            pairs = {p["id"]: p for p in ann["pairs"]}
            errs = []
            if entry.get("pair_id") not in pairs:
                errs.append(f"no pair {entry.get('pair_id')!r} on {stem}")
            if entry.get("relabeler") not in self.annotators():
                errs.append("relabeler is not a known name tag")
            if any(e["pair_id"] == entry.get("pair_id") and e["relabeler"] == entry.get("relabeler") for e in obj["relabels"]):
                errs.append("this pair is already relabelled by this name tag")
            if not errs:
                entry = dict(entry, own_pair=pairs[entry["pair_id"]]["annotator"] == entry["relabeler"])
                new = dict(obj, relabels=obj["relabels"] + [entry])
                errs = SC.validate(new, RELABEL_SCHEMA)
            if errs:
                raise ValueError(errs)
            self.relabels.mkdir(parents=True, exist_ok=True)
            p = self.relabel_path(stem)
            data = (json.dumps(new, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
            if p.exists():
                bdir = self.backups / "relabels" / stem
                bdir.mkdir(parents=True, exist_ok=True)
                (bdir / f"{stem}.{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json").write_bytes(p.read_bytes())
            tmp = p.with_name(f".{p.name}.tmp{os.getpid()}.{threading.get_ident()}")
            tmp.write_bytes(data)
            os.replace(tmp, p)
            return entry, etag_of(data), new

    # ---------------------------------------------------------------- review (only with --review)
    def runs(self):
        return sorted(d.name for d in self.review.iterdir() if (d / "per_pair.csv").exists() and (d / "summary.json").exists())

    def run(self, name):
        if name not in self.runs():
            raise KeyError(name)
        if name not in self._runs:
            import csv
            d = self.review / name
            with open(d / "per_pair.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            prof = npyread.load(d / "profiles.npz")
            idx = {(prof["methods"][r["method"]], r["image"], r["id"]): (r["start"], r["stop"]) for r in prof["index"]}
            meta = json.loads((d / "summary.json").read_text(encoding="utf-8"))["meta"]
            self._runs[name] = dict(rows=rows, prof=prof, idx=idx, meta=dict(methods=meta["methods"], images=meta["images"],
                                    tau=meta["tau"], groups=meta["groups"], finished=meta.get("finished")))
        return self._runs[name]

    def profile(self, name, method, image, pid):
        R = self.run(name)
        s, e = R["idx"][(method, image, pid)]
        out = {"t": list(R["prof"]["t"][0][s:e])}
        for k in ("I", "R", "L"):
            a = R["prof"][k][0]
            out[k] = [list(a[3 * i:3 * i + 3]) for i in range(s, e)]
        return out

    # ---------------------------------------------------------------- files
    def path(self, stem):
        return self.ann / f"{stem}.json"

    def read(self, stem):
        b = self.path(stem).read_bytes()
        return json.loads(b), etag_of(b)

    def write(self, stem, obj):
        """Backup the current file, then write atomically. Caller holds LOCK."""
        p = self.path(stem)
        data = (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
        if p.exists():
            bdir = self.backups / stem
            bdir.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            (bdir / f"{stem}.{stamp}.json").write_bytes(p.read_bytes())
        tmp = p.with_name(f".{p.name}.tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        return etag_of(data)

    # ---------------------------------------------------------------- checks
    def check(self, stem, new, old):
        errs = SC.validate_file(new)
        if errs:
            return errs
        m = self.images[stem]
        for k, want in (("image", stem), ("set", old["set"]), ("schema_version", old["schema_version"]),
                        ("width", m["W"]), ("height", m["H"])):
            if new.get(k) != want:
                errs.append(f"$.{k}: must stay {want!r}")
        olog, nlog = old.get("s_seed_log", []), new.get("s_seed_log", [])
        if len(nlog) < len(olog):
            errs.append("$.s_seed_log: entries may not be removed (append-only)")
        for a, b in zip(olog, nlog):
            if any(a.get(k) != b.get(k) for k in SEED_IMMUTABLE):
                errs.append(f"$.s_seed_log[{a.get('index')}]: index / seed / x / y / proposed_* are immutable")
        if len(nlog) > len(olog):
            errs.append("$.s_seed_log: new seeds come only from POST /api/seed")
        oldp = {p["id"]: p for p in old["pairs"]} | {t["id"]: t["pair"] for t in old.get("deleted_pairs", [])}
        for p in new["pairs"]:
            o = oldp.get(p["id"])
            if o and (o.get("annotator") != p["annotator"] or o.get("created") != p["created"]):
                errs.append(f"pair {p['id']}: annotator and created never change")
        known = set(self.annotators())
        for p in new["pairs"]:
            if p["id"] not in oldp and p["annotator"] not in known:
                errs.append(f"pair {p['id']}: annotator {p['annotator']!r} is not in annotators.json (choose or add a name tag)")
            if p.get("last_edited_by") and p.get("last_edited_by") != (oldp.get(p["id"]) or {}).get("last_edited_by") \
                    and p["last_edited_by"] not in known:
                errs.append(f"pair {p['id']}: last_edited_by {p['last_edited_by']!r} is not a known name tag")
        for k, mark in (new.get("exhausted") or {}).items():
            if mark != (old.get("exhausted") or {}).get(k) and mark["by"] not in known:
                errs.append(f"exhausted.{k}: {mark['by']!r} is not a known name tag")
        ids = [p["id"] for p in new["pairs"]] + [t["id"] for t in new.get("deleted_pairs", [])]
        if len(ids) != len(set(ids)):
            errs.append("ids must be unique across pairs and deleted_pairs")
        return errs

    # ---------------------------------------------------------------- annotators
    def annotators_path(self):
        return self.ann / "annotators.json"

    def tags(self):
        return AN.load(self.annotators_path())["annotators"]

    def annotators(self):
        """Known annotator ids."""
        return [a["id"] for a in self.tags()]

    def add_annotator(self, name):
        with LOCK:
            obj = AN.load(self.annotators_path())
            e, created = AN.add(obj, name)
            if created:
                AN.save(self.annotators_path(), obj)
            return e, obj["annotators"]

    def rename_annotator(self, id_, name, by):
        with LOCK:
            obj = AN.load(self.annotators_path())
            if by not in {a["id"] for a in obj["annotators"]}:
                raise ValueError("choose your own name tag first")
            e, changed = AN.rename(obj, id_, name, by)
            if changed:
                AN.save(self.annotators_path(), obj)
            return e, obj["annotators"]

    def usage(self, id_):
        n_pairs = n_img = 0
        for p in sorted(self.ann.glob("*.json")):
            if p.name == "annotators.json":
                continue
            obj = json.loads(p.read_text(encoding="utf-8"))
            k = sum(1 for q in obj["pairs"] if q["annotator"] == id_)
            n_pairs += k
            n_img += bool(k) or obj.get("assigned_to") == id_
        return dict(id=id_, pairs=n_pairs, images=n_img)

    # ---------------------------------------------------------------- random s pool
    def propose_seed(self, stem, annotator, if_match):
        C = self.manifest["constants"]
        r = int(C["patch"]["default_r"])
        with LOCK:
            obj, et = self.read(stem)
            if if_match != et:
                return None, et, obj
            log = obj.setdefault("s_seed_log", [])
            idx = len(log)
            seed = f"{self.manifest['seed_base']}:{stem}:{idx}"
            rng = random.Random(seed)
            with gzip.open(self.cache / stem / "valid.u8.gz", "rb") as f:
                valid = f.read()
            W, H = obj["width"], obj["height"]
            for _ in range(200000):
                x, y = rng.randrange(r, W - r), rng.randrange(r, H - r)
                if valid[y * W + x]:
                    break
            else:
                raise RuntimeError("no valid pixel found")
            log.append(dict(index=idx, seed=seed, x=x, y=y, proposed_at=now_iso(), proposed_by=annotator, outcome="pending"))
            errs = SC.validate_file(obj)
            if errs:
                raise ValueError(errs)
            return log[-1], self.write(stem, obj), obj

    def summary(self):
        out = []
        for stem, m in self.images.items():
            d = dict(stem=stem, set=m["set"], split=m.get("split"), assigned_to=m.get("assigned_to"), counts={}, by_annotator={}, exhausted=[])
            if self.path(stem).exists():
                obj = json.loads(self.path(stem).read_text(encoding="utf-8"))
                d["split"], d["exhausted"] = obj.get("split"), sorted(obj.get("exhausted") or {})
                for p in obj["pairs"]:
                    d["counts"][p["type"]] = d["counts"].get(p["type"], 0) + 1
                    a = d["by_annotator"].setdefault(p["annotator"], {})
                    a[p["type"]] = a.get(p["type"], 0) + 1
            out.append(d)
        return out


class Handler(BaseHTTPRequestHandler):
    store = None
    server_version = "metrics-annotate/1.0"

    def log_message(self, fmt, *args):
        if os.environ.get("ANNOTATE_LOG"):
            super().log_message(fmt, *args)

    def send(self, code, body=b"", ctype="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def stem_ok(self, stem):
        if stem not in self.store.images:
            self.send(404, dict(error=f"unknown image {stem!r}"))
            return False
        return True

    def do_GET(self):
        u = urlparse(self.path).path
        S = self.store
        if u in STATIC or (S.review and u in REVIEW_STATIC):
            f = HERE / (STATIC.get(u) or REVIEW_STATIC[u])
            return self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "text/plain")
        q = dict(x.split("=", 1) for x in urlparse(self.path).query.split("&") if "=" in x)
        from urllib.parse import unquote
        q = {k: unquote(v) for k, v in q.items()}
        m = re.fullmatch(r"/api/relabel/queue/([^/]+)", u)
        if m:
            if not self.stem_ok(m.group(1)):
                return
            return self.send(200, S.relabel_queue(m.group(1), q.get("annotator", ""), q.get("own") == "1"))
        m = re.fullmatch(r"/api/relabel/([^/]+)", u)
        if m:
            if not self.stem_ok(m.group(1)):
                return
            obj, et = S.read_relabels(m.group(1))
            return self.send(200, obj, headers={"ETag": et})
        if u.startswith("/api/review/"):
            if not S.review:
                return self.send(404, dict(error="review routes are off (start the server with --review)"))
            try:
                if u == "/api/review/runs":
                    return self.send(200, dict(runs=S.runs()))
                m = re.fullmatch(r"/api/review/([^/]+)/meta", u)
                if m:
                    return self.send(200, S.run(m.group(1))["meta"])
                m = re.fullmatch(r"/api/review/([^/]+)/pairs", u)
                if m:
                    return self.send(200, [r for r in S.run(m.group(1))["rows"] if r["image"] == q.get("image")])
                m = re.fullmatch(r"/api/review/([^/]+)/profile", u)
                if m:
                    return self.send(200, S.profile(m.group(1), q["method"], q["image"], q["id"]))
            except KeyError as e:
                return self.send(404, dict(error=f"not found: {e}"))
        if u == "/api/manifest":
            return self.send(200, S.manifest)
        if u == "/api/images":
            return self.send(200, S.summary())
        if u == "/api/annotators":
            return self.send(200, dict(annotators=S.tags()))
        m = re.fullmatch(r"/api/annotators/([a-z0-9_]+)/usage", u)
        if m:
            return self.send(200, S.usage(m.group(1)))
        m = re.fullmatch(r"/api/img/([^/]+)/([^/]+)", u)
        if m:
            stem, name = m.groups()
            if not self.stem_ok(stem):
                return
            if name not in IMG_FILES or not (S.cache / stem / name).exists():
                return self.send(404, dict(error=f"no {name} for {stem}"))
            b = (S.cache / stem / name).read_bytes()
            if name.endswith(".gz"):
                return self.send(200, b, "application/octet-stream", {"Content-Encoding": "gzip"})
            return self.send(200, b, "image/png" if name.endswith(".png") else "application/json")
        m = re.fullmatch(r"/api/ann/([^/]+)", u)
        if m:
            stem = m.group(1)
            if not self.stem_ok(stem):
                return
            obj, et = S.read(stem)
            return self.send(200, obj, headers={"ETag": et})
        return self.send(404, dict(error="not found"))

    def do_PUT(self):
        u = urlparse(self.path).path
        m = re.fullmatch(r"/api/ann/([^/]+)", u)
        if not m:
            return self.send(404, dict(error="not found"))
        stem = m.group(1)
        if not self.stem_ok(stem):
            return
        try:
            new = self.body()
        except json.JSONDecodeError as e:
            return self.send(400, dict(errors=[f"invalid JSON: {e}"]))
        S = self.store
        with LOCK:
            old, et = S.read(stem)
            if self.headers.get("If-Match") != et:
                return self.send(409, dict(error="the file changed since you loaded it", current=old, etag=et),
                                 headers={"ETag": et})
            errs = S.check(stem, new, old)
            if errs:
                return self.send(400, dict(errors=errs[:50]))
            et2 = S.write(stem, new)
        return self.send(200, dict(ok=True, etag=et2), headers={"ETag": et2})

    def do_POST(self):
        u = urlparse(self.path).path
        S = self.store
        if u == "/api/annotators":
            try:
                e, tags = S.add_annotator(self.body().get("name", ""))
            except ValueError as e:
                return self.send(400, dict(errors=[str(e)]))
            return self.send(200, dict(tag=e, annotators=tags))
        if u == "/api/annotators/rename":
            b = self.body()
            try:
                e, tags = S.rename_annotator(b.get("id", ""), b.get("name", ""), b.get("by", ""))
            except ValueError as e:
                return self.send(400, dict(errors=[str(e)]))
            return self.send(200, dict(tag=e, annotators=tags))
        m = re.fullmatch(r"/api/relabel/([^/]+)", u)
        if m:
            stem = m.group(1)
            if not self.stem_ok(stem):
                return
            try:
                entry, et, obj = S.add_relabel(stem, self.body(), self.headers.get("If-Match"))
            except ValueError as e:
                errs = e.args[0] if isinstance(e.args[0], list) else [str(e)]
                return self.send(400, dict(errors=errs))
            if entry is None:
                return self.send(409, dict(error="the relabel file changed since you loaded it", current=obj, etag=et), headers={"ETag": et})
            return self.send(200, dict(entry=entry, etag=et), headers={"ETag": et})
        m = re.fullmatch(r"/api/seed/([^/]+)", u)
        if m:
            stem = m.group(1)
            if not self.stem_ok(stem):
                return
            who = str(self.body().get("annotator", ""))
            if who not in S.annotators():
                return self.send(400, dict(errors=["choose a name tag first"]))
            entry, et, obj = S.propose_seed(stem, who, self.headers.get("If-Match"))
            if entry is None:
                return self.send(409, dict(error="the file changed since you loaded it", current=obj, etag=et),
                                 headers={"ETag": et})
            return self.send(200, dict(seed=entry, etag=et, file=obj), headers={"ETag": et})
        return self.send(404, dict(error="not found"))


def make_server(manifest_path, host, port, annotations=None, review=None):
    Handler.store = Store(manifest_path, annotations, review)
    return ThreadingHTTPServer((host, port), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(HERE / "cache"), help="folder holding manifest.json (prepare.py)")
    ap.add_argument("--annotations", help="annotation folder (default: the manifest's; tests use a copy)")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--review", nargs="?", const=str(HERE.parent / "runs"), default=None,
                    help="also serve the review page (/review.html) over metric runs in this folder (default metrics/runs); "
                         "off by default, so annotation mode has no route to model output")
    a = ap.parse_args()
    mp = Path(a.cache) / "manifest.json"
    if not mp.exists():
        sys.exit(f"{mp} missing: run metrics/run.sh annotate/prepare.py first")
    man = json.loads(mp.read_text())
    host, port = a.host or man["host"], a.port if a.port is not None else man["port"]
    srv = make_server(mp, host, port, a.annotations, a.review)
    hp = f"http://{host if host != '0.0.0.0' else 'localhost'}:{srv.server_address[1]}"
    print(f"annotation tool: {hp}/   (annotations: {Handler.store.ann}; Ctrl+C stops)", flush=True)
    if a.review:
        print(f"review page:     {hp}/review.html   (runs: {a.review})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
