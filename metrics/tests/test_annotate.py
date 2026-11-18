"""Phase 4: annotation server rules and merge (stdlib parts; the browser side is annotate/selftest.py)."""

import copy
import json
import shutil
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M / "annotate"))
import merge as MR  # noqa: E402
import server as SV  # noqa: E402

STEM = "taiwei_cui_080"
MAN = M / "annotate" / "cache" / "manifest.json"
pytestmark = pytest.mark.skipif(not MAN.exists(), reason="run annotate/prepare.py first")


def copy_ann(dst):
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(M / "annotations" / f"{STEM}.json", dst / f"{STEM}.json")
    shutil.copy2(M / "annotations" / "annotators.json", dst / "annotators.json")
    return dst


@pytest.fixture()
def srv(tmp_path):
    ann = copy_ann(tmp_path / "ann")
    s = SV.make_server(MAN, "127.0.0.1", 0, annotations=str(ann))
    th = threading.Thread(target=s.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{s.server_address[1]}"

    def call(path, method="GET", body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req) as r:
                b = r.read()
                return r.status, r.headers.get("ETag"), (json.loads(b) if "json" in r.headers.get("Content-Type", "") else b)
        except urllib.error.HTTPError as e:
            b = e.read()
            return e.code, e.headers.get("ETag"), (json.loads(b) if b[:1] in b"{[" else b)
    yield call, ann
    s.shutdown()


def put(call, obj, et):
    return call(f"/api/ann/{STEM}", "PUT", obj, {"If-Match": et})


def test_roundtrip_backup_and_conflict(srv):
    call, ann = srv
    st, et, obj = call(f"/api/ann/{STEM}")
    assert st == 200 and len(et) == 64
    obj["pairs"][0]["note"] = "edited by test"
    st, et2, r = put(call, obj, et)
    assert st == 200 and r["etag"] == et2 != et
    assert json.loads((ann / f"{STEM}.json").read_text())["pairs"][0]["note"] == "edited by test"
    assert len(list((ann / ".backups" / STEM).glob("*.json"))) == 1
    st, _, r = put(call, obj, et)                                  # stale etag
    assert st == 409 and r["etag"] == et2


@pytest.mark.parametrize("tamper,needle", [
    (lambda o: o["pairs"][0].pop("annotator"), "annotator"),
    (lambda o: o.update(set="heldout"), "set"),
    (lambda o: o["pairs"][0].update(annotator="someone_else"), "never change"),
    (lambda o: o["pairs"].append(dict(copy.deepcopy(o["pairs"][0]), id="t=1", annotator="nobody_known")), "annotators.json"),
    (lambda o: o["pairs"][0].update(annotator="Yueren"), "annotator"),                       # a name, not an id
    (lambda o: o["s_seed_log"].append(dict(index=0, seed="x", x=1, y=1, proposed_at="2026-11-18T00:00:00Z",
                                           proposed_by="yueren", outcome="pending")), "POST /api/seed"),
    (lambda o: o["pairs"].append(copy.deepcopy(o["pairs"][0])), "duplicate"),
])
def test_rejections(srv, tamper, needle):
    call, ann = srv
    st, et, obj = call(f"/api/ann/{STEM}")
    before = (ann / f"{STEM}.json").read_bytes()
    tamper(obj)
    st, _, r = put(call, obj, et)
    assert st == 400 and any(needle in e for e in r["errors"]), r
    assert (ann / f"{STEM}.json").read_bytes() == before


def test_seed_log_append_only_and_deterministic(srv, tmp_path):
    call, ann = srv
    st, et, obj = call(f"/api/ann/{STEM}")
    assert call(f"/api/seed/{STEM}", "POST", {"annotator": "ghost"}, {"If-Match": et})[0] == 400   # unknown tag
    st, et, r = call(f"/api/seed/{STEM}", "POST", {"annotator": "yueren"}, {"If-Match": et})
    assert st == 200 and r["seed"]["index"] == 0 and r["seed"]["outcome"] == "pending"
    first = r["seed"]
    obj = r["file"]
    obj["s_seed_log"][0]["x"] += 1                                      # immutable
    assert put(call, obj, et)[0] == 400
    obj["s_seed_log"][0]["x"] -= 1
    obj["s_seed_log"][0].update(outcome="rejected")                     # rejected needs a reason (schema)
    assert put(call, obj, et)[0] == 400
    obj["s_seed_log"][0].update(reason="specular", resolved_at="2026-11-18T00:00:00Z", resolved_by="yueren")
    st, et, _ = put(call, obj, et)
    assert st == 200
    obj["s_seed_log"] = []                                              # removal refused
    assert put(call, obj, et)[0] == 400
    store = SV.Store(MAN, str(copy_ann(tmp_path / "other")))            # another machine: same seed for index 0
    e2, _, _ = store.propose_seed(STEM, "yueren", store.read(STEM)[1])
    assert (e2["seed"], e2["x"], e2["y"]) == (first["seed"], first["x"], first["y"])


def test_annotators_and_routes(srv):
    call, ann = srv
    st, _, r = call("/api/annotators", "POST", {"name": "  ana   lopez "})
    assert st == 200 and r["tag"]["name"] == "ana lopez" and r["tag"]["id"] == "ana_lopez" and r["tag"]["history"] == []
    st, _, r = call("/api/annotators", "POST", {"name": "ANA LOPEZ"})
    assert r["tag"]["id"] == "ana_lopez" and [a["name"] for a in r["annotators"]].count("ana lopez") == 1
    assert call("/api/annotators", "POST", {"name": "   "})[0] == 400
    st, _, r = call("/api/annotators", "POST", {"name": "x" * 70})
    assert st == 400 and "too long" in r["errors"][0]
    assert call("/api/annotators", "POST", {"name": "bad\u0007name"})[0] == 400
    for bad in ("/api/ann/nope", f"/api/img/{STEM}/../../config.yaml", f"/api/img/{STEM}/secret.bin", "/runs/", "/api/pred",
                "/config.yaml", "/annotate/server.py"):
        assert call(bad)[0] == 404, bad
    st, _, summ = call("/api/images")
    assert st == 200 and any(x["stem"] == STEM and x["counts"].get("i") for x in summ)


def test_merge(tmp_path):
    a = json.loads((M / "annotations" / f"{STEM}.json").read_text())
    b = copy.deepcopy(a)
    new = dict(copy.deepcopy(a["pairs"][0]), id="t=9999999999999", note="from B")
    b["pairs"].append(new)                                               # added on B
    b["pairs"][1]["note"] = "edited on B"                                # conflict with A's version
    gone = a["pairs"][2]
    a["pairs"] = [p for p in a["pairs"] if p["id"] != gone["id"]]       # deleted on A
    a["deleted_pairs"] = [dict(id=gone["id"], deleted_at="2026-11-18T00:00:00Z", deleted_by="yueren", pair=gone)]
    m, rep = MR.merge_file(a, b)
    ids = [p["id"] for p in m["pairs"]]
    assert new["id"] in ids and gone["id"] not in ids
    assert any(b["pairs"][1]["id"] in c for c in rep["conflicts"])
    assert next(p for p in m["pairs"] if p["id"] == b["pairs"][1]["id"])["note"] == a["pairs"][1]["note"]   # A kept
    from core import schema as SC
    assert SC.validate_file(m) == []


def test_relabel_endpoints_blind_and_append_only(srv):
    call, ann = srv
    st, _, q = call(f"/api/relabel/queue/{STEM}?annotator=yueren&own=0")
    assert st == 200 and q["queue"] == [] and q["total"] == 0              # every pair is Yueren's: none to relabel
    call("/api/annotators", "POST", {"name": "Second Person"})
    st, _, q = call(f"/api/relabel/queue/{STEM}?annotator=second_person&own=0")
    assert q["total"] > 0 and len(q["queue"]) == q["total"]
    assert set(q["queue"][0]) == {"pair_id", "A", "B"}                      # geometry only: no type, labels, annotator
    st2, _, q2 = call(f"/api/relabel/queue/{STEM}?annotator=second_person&own=0")
    assert [x["pair_id"] for x in q2["queue"]] == [x["pair_id"] for x in q["queue"]]   # fixed shuffle
    st, et, rl = call(f"/api/relabel/{STEM}")
    assert st == 200 and et == "none" and rl["relabels"] == []
    pid = q["queue"][0]["pair_id"]
    e = dict(pair_id=pid, relabeler="second_person", at="2026-11-18T00:00:00Z", type="m", confidence="sure")
    st, et2, r = call(f"/api/relabel/{STEM}", "POST", e, {"If-Match": et})
    assert st == 200 and r["entry"]["own_pair"] is False
    assert call(f"/api/relabel/{STEM}", "POST", e, {"If-Match": et})[0] == 409            # stale
    st, _, r = call(f"/api/relabel/{STEM}", "POST", e, {"If-Match": et2})
    assert st == 400 and any("already" in x for x in r["errors"])                          # one per (pair, relabeler)
    bad = dict(e, pair_id=q["queue"][1]["pair_id"], type="i")                              # i needs lit_end
    assert call(f"/api/relabel/{STEM}", "POST", bad, {"If-Match": et2})[0] == 400
    assert call(f"/api/relabel/{STEM}", "POST", dict(e, pair_id="t=1"), {"If-Match": et2})[0] == 400
    orig = json.loads((ann / f"{STEM}.json").read_text())
    assert next(p for p in orig["pairs"] if p["id"] == pid)["type"] != "x"                  # pair file untouched by relabels
    st, _, q3 = call(f"/api/relabel/queue/{STEM}?annotator=second_person&own=0")
    assert pid not in [x["pair_id"] for x in q3["queue"]] and q3["done"] == 1


def test_review_routes_off_without_flag(srv):
    call, ann = srv
    for u in ("/review.html", "/review.js", "/api/review/runs"):
        assert call(u)[0] == 404, u


def test_npyread_matches_numpy(tmp_path):
    import numpy as np
    import npyread as NR
    idx = np.array([(0, "img_a", "t=1", 0, 3), (1, "img_b", "t=22", 3, 5)],
                   dtype=[("method", "i4"), ("image", "U64"), ("id", "U32"), ("start", "i8"), ("stop", "i8")])
    t = np.arange(5, dtype=np.float32)
    I = np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32)
    np.savez_compressed(tmp_path / "p.npz", methods=np.array(["all_R", "model"]), index=idx, t=t, I=I)
    z = NR.load(tmp_path / "p.npz")
    assert z["methods"] == ["all_R", "model"]
    assert z["index"][1] == dict(method=1, image="img_b", id="t=22", start=3, stop=5)
    assert list(z["t"][0]) == t.tolist() and z["I"][1] == (5, 3) and list(z["I"][0]) == I.ravel().tolist()


def test_kappa():
    sys.path.insert(0, str(M / "tools"))
    import relabel_agreement as RA
    assert RA.kappa([]) is None
    assert RA.kappa([("i", "i"), ("m", "m")]) == 1.0
    assert RA.kappa([("i", "i"), ("i", "i")]) is None                      # chance agreement 1: undefined
    k = RA.kappa([("i", "i"), ("i", "m"), ("m", "m"), ("m", "i")])
    assert abs(k - 0.0) < 1e-12


def test_rename_changes_only_annotators_json(srv):
    import hashlib
    call, ann = srv
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ann.glob("*.json") if p.name != "annotators.json"}
    st, _, u = call("/api/annotators/yueren/usage")
    assert st == 200 and u["pairs"] > 0
    st, _, r = call("/api/annotators/rename", "POST", {"id": "yueren", "name": "Yueren Xing", "by": "yueren"})
    assert st == 200 and r["tag"]["name"] == "Yueren Xing" and r["tag"]["id"] == "yueren"
    h = r["tag"]["history"][-1]
    assert (h["from"], h["to"], h["by"]) == ("Yueren", "Yueren Xing", "yueren")
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ann.glob("*.json") if p.name != "annotators.json"}
    assert before == after                                                     # no annotation file rewritten
    call("/api/annotators", "POST", {"name": "Ana Lopez"})
    for bad in ("ANA LOPEZ", "   ", ""):
        st, _, r = call("/api/annotators/rename", "POST", {"id": "yueren", "name": bad, "by": "yueren"})
        assert st == 400, bad
    st, _, r = call("/api/annotators/rename", "POST", {"id": "yueren", "name": "x", "by": "nobody"})
    assert st == 400
    st, _, tags = call("/api/annotators")
    assert next(t for t in tags["annotators"] if t["id"] == "yueren")["name"] == "Yueren Xing"


def test_merge_annotators_by_id():
    mine = dict(schema_version="1.2", annotators=[dict(id="yueren", name="Yueren", created="2026-11-18T00:00:00Z", history=[])])
    theirs = dict(schema_version="1.2", annotators=[
        dict(id="yueren", name="Yueren X", created="2026-11-18T00:00:00Z", history=[dict(**{"from": "Yueren", "to": "Yueren X",
             "at": "2026-11-18T01:00:00Z", "by": "yueren"})]),
        dict(id="annotator_b", name="Ana", created="2026-11-18T00:00:00Z", history=[])])
    add, conf = MR.merge_annotators(mine, theirs)
    assert [e["id"] for e in add] == ["annotator_b"]
    assert len(conf) == 1 and "kept 'Yueren'" in conf[0]
    assert mine["annotators"][0]["name"] == "Yueren"                           # never overwritten


def test_exports_show_renamed_name(cfg, tmp_path):
    """run_metrics per_pair.csv: annotator = the current name, annotator_id = the stable id."""
    import csv as C
    import subprocess
    from conftest import need_dataset
    need_dataset(cfg, [STEM])
    ann = copy_ann(tmp_path / "ann")
    reg = json.loads((ann / "annotators.json").read_text())
    reg["annotators"][0]["name"] = "Renamed Person"
    (ann / "annotators.json").write_text(json.dumps(reg))
    conf = tmp_path / "config.yaml"
    conf.write_text((M / "config.yaml").read_text().replace("annotations: metrics/annotations", f"annotations: {ann}"))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-B", str(M / "run_metrics.py"), "--config", str(conf), "--images", STEM, "--out", str(out), "--test"],
                       cwd=M, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = list(C.DictReader(open(out / "per_pair.csv", newline="")))
    assert rows and {x["annotator"] for x in rows} == {"Renamed Person"} and {x["annotator_id"] for x in rows} == {"yueren"}
