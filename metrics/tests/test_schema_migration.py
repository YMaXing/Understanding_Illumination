"""Schema v1 (§3) and the lossless migration (C5, C6, the 070 ruling)."""

import copy
import json

import pytest

import migrate_v0 as MG
from core import config as K
from core import schema as SC


@pytest.fixture(scope="module")
def files(cfg):
    return MG.read_files(cfg)


def a_pair(**kw):
    p = dict(id="t=1", image="x", type="m", A=dict(x=1, y=2, r=2), B=dict(x=5, y=2, r=2), confidence="sure",
             used_in_loss=False, overlay_seen="unknown", annotator="yueren", created="2026-11-18T00:00:00.000Z", note="",
             schema_version="1.2")
    p.update(kw)
    return p


def a_file(*pairs):
    return dict(schema_version="1.2", image="x", set="dev_v2", split="val", seen_in_training="unknown", assigned_to="yueren",
                image_pixel_sha256="0" * 64, width=10, height=10, pairs=list(pairs), s_seed_log=[], legacy={})


def test_validator_covers_schema():
    assert SC.unsupported_keywords(SC.load_schema()) == []


@pytest.mark.parametrize("pair,ok", [
    (a_pair(), True),
    (a_pair(type="i", lit_end="A"), True),
    (a_pair(type="i"), False),                                            # i needs lit_end
    (a_pair(lit_end="A"), False),                                         # lit_end only on i
    (a_pair(type="s", lighting="lit", pool="random"), True),
    (a_pair(type="s", lighting="lit", pool="random", gap_ref="g1"), False),  # gap_ref only for targeted
    (a_pair(type="s", lighting="shadow", pool="targeted"), False),        # targeted needs gap_ref
    (a_pair(type="s", lighting="shadow", pool="targeted", gap_ref="F10"), True),
    (a_pair(type="d", material="same", illumination="different"), True),
    (a_pair(type="d", material="same"), False),
    (a_pair(material="same", illumination="same"), False),                # d fields only on d
    (a_pair(overlay_seen=True), True),
    (a_pair(overlay_seen=1), False),                                      # True != 1
    (a_pair(confidence="maybe"), False),
    (a_pair(annotator="Yueren"), False),                                  # schema 1.2: pairs store the tag id, not the name
    (a_pair(annotator=" yueren"), False),
    (a_pair(annotator=""), False),
    (a_pair(id="1794"), False),
    (a_pair(A=dict(x=1, y=2, r=0)), False),
    (a_pair(A=dict(x=1, y=2)), False),
    (a_pair(extra=1), False),
    (a_pair(last_edited_by="ana_lopez", edited="2026-11-18T01:02:03Z"), True),
    (a_pair(last_edited_by="Ana Lopez", edited="2026-11-18T01:02:03Z"), False),
    (a_pair(split="eval"), False),                                        # schema 1.1: no pair-level split
    (a_pair(used_in_loss="no"), False),
])
def test_schema_cases(pair, ok):
    assert (SC.validate_file(a_file(pair)) == []) is ok


def test_cross_field_rules():
    assert SC.validate_file(a_file(a_pair(), a_pair()))                     # duplicate id
    assert SC.validate_file(a_file(a_pair(image="y")))                      # image mismatch


def test_file_level_rules():
    assert SC.validate_file(dict(a_file(), split="eval"))                  # split is train / val / test
    f = a_file(); f.pop("split")
    assert SC.validate_file(f)
    ok = dict(a_file(), exhausted={"i": dict(by="yueren", at="2026-11-18T00:00:00Z", note="no more edges")})
    assert SC.validate_file(ok) == []
    assert SC.validate_file(dict(a_file(), exhausted={"i": dict(by="yueren", at="2026-11-18T00:00:00Z", note="")}))
    assert SC.validate_file(dict(a_file(), exhausted={"s_targeted": dict(by="y", at="2026-11-18T00:00:00Z", note="x")}))
    f = a_file(); f.pop("assigned_to")
    assert SC.validate_file(f)
    assert SC.validate_file(dict(a_file(), reflective_advisory="yes"))
    assert SC.validate_file(dict(a_file(), reflective_advisory=True)) == []
    seed = dict(index=0, seed="s:x:0", x=1, y=1, proposed_at="2026-11-18T00:00:00Z", proposed_by="y", outcome="abandoned")
    assert SC.validate_file(dict(a_file(), s_seed_log=[seed]))              # abandoned needs a reason
    assert SC.validate_file(dict(a_file(), s_seed_log=[dict(seed, reason="left the image")])) == []


def test_all_files_valid(files):
    assert len(files) == 100
    for stem, f in files.items():
        assert SC.validate_file(f) == [], stem


def test_migration_lossless(cfg):
    from conftest import need_migration_sources
    need_migration_sources(cfg)
    assert MG.verify(cfg) == []


def test_migration_defaults_and_070(cfg, files):
    n = 0
    ids = set()
    for f in files.values():
        for p in f["pairs"] + [t["pair"] for t in f.get("deleted_pairs", [])]:   # deleted later in the tool: tombstones
            if "v0" not in p.get("legacy", {}):
                continue                                                    # new pairs (tool), not migrated
            n += 1
            assert p["id"] not in ids
            ids.add(p["id"])
            if p.get("last_edited_by"):
                continue                                                    # edited in the tool after migration
            assert (p["annotator"], p["confidence"], p["used_in_loss"], p["overlay_seen"]) == ("yueren", "sure", False, "unknown")
        assert f["seen_in_training"] == "unknown"
    assert n == 871 + 1098 + 1428 + 1614
    p = next(p for p in files["taiwei_cui_070"]["pairs"] if p["id"] == "t=1794165897383")
    assert p["type"] == "b" and p["legacy"]["v0"]["type"] == "m" and p["legacy"]["relabel"]["to"] == "b"


def test_deleted_never_in_pairs(files):
    for f in files.values():
        live = {p["id"] for p in f["pairs"]}
        assert not live & {f"t={d['v0']['t']}" for d in f["legacy"].get("deleted", [])}


def test_annotators_file(cfg):
    a = json.loads((K.repo_path(cfg["paths"]["annotations"]) / "annotators.json").read_text())
    assert SC.validate(a, SC.load_schema(K.METRICS / "schema" / "annotators.schema.json")) == []
    assert [(e["id"], e["name"]) for e in a["annotators"]][0] == ("yueren", "Yueren")
    assert {e["id"] for e in a["annotators"]} >= {"yueren", "annotator_b"}         # annotators may add tags


def test_assignments_and_reflective(cfg, files):
    import json as J
    rec = J.loads(K.repo_path(cfg["paths"]["split_assignment"]).read_text())
    group = {x["stem"]: x["annotator_group"] for x in rec["images"]}
    for s, f in files.items():
        assert f["assigned_to"] == ("yueren" if group[s] == "A" else "annotator_b")
        assert len(f["image_pixel_sha256"]) == 64
    refl = sorted(s for s, f in files.items() if f.get("reflective_advisory"))
    draw = J.loads((K.METRICS / "splits" / "draw54.json").read_text())["images"]
    assert refl == sorted(r["stem"] for r in draw if r["rank"] in (1, 5, 10, 12, 16, 20, 28, 31, 33, 39, 45))


def test_verify_detects_tampering(cfg, files, monkeypatch):
    from conftest import need_migration_sources
    need_migration_sources(cfg)
    f2 = copy.deepcopy(files)
    f2["taiwei_cui_075"]["pairs"][0]["legacy"]["v0"]["a_row"] += 1
    monkeypatch.setattr(MG, "read_files", lambda c: f2)
    assert MG.verify(cfg)
