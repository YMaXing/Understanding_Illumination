"""Checkpoint 5: fresh-clone setup pieces (local config, optional imports, split assignment, schema upgrade,
overlay archive, prepare's hash check)."""

import json
import sys
import tarfile
from pathlib import Path

import pytest

M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M / "tools"))
from core import config as K  # noqa: E402


def test_local_config_merges(tmp_path, monkeypatch):
    lp = tmp_path / "config.local.yaml"
    lp.write_text("dataset:\n  val_set: /somewhere/else\nmaps:\n  illum_edges:\n    dev_v2: {dir: x, key: illum}\n")
    monkeypatch.setattr(K, "LOCAL_CONFIG", lp)
    c = K.load_config()
    assert c["dataset"]["val_set"] == "/somewhere/else" and c["_local"] == str(lp)
    assert c["maps"]["illum_edges"]["dev_v2"]["key"] == "illum" and c["maps"]["geom_v1_1"]["key"] == "w"
    assert c["thresholds"]["tau"] == 0.10                                    # untouched keys survive
    monkeypatch.setattr(K, "LOCAL_CONFIG", tmp_path / "absent.yaml")
    assert K.load_config()["_local"] is None


def test_tracked_config_has_no_machine_paths():
    txt = (M / "config.yaml").read_text()
    assert "/mnt/" not in txt and ":\\" not in txt and "/home/" not in txt
    assert "loss/outputs" not in txt and "illumination_edge/outputs" not in txt


def test_optional_import_placeholder():
    m = K._stub("material_edges_test")
    with pytest.raises(ImportError, match="log_chromaticity/material_edges_test.py"):
        m.detect


def test_split_allocation():
    import assign_splits as AS
    cnt = AS.allocate([21, 25, 54], [60, 20, 20])
    assert cnt == [[13, 4, 4], [15, 5, 5], [32, 11, 11]]
    assert [sum(c[k] for c in cnt) for k in range(3)] == [60, 20, 20]


def test_split_record_matches_files(cfg):
    rec = json.loads(K.repo_path(cfg["paths"]["split_assignment"]).read_text())
    assert rec["totals"] == {"train": 60, "val": 20, "test": 20}
    for x in rec["images"]:
        assert K.split_of(cfg, x["stem"]) == x["split"]
    glossy = {x["stem"] for x in rec["images"] if x["glossy_pool"]}
    assert len(glossy) == 21 and glossy <= K.glossy_pool(cfg)


def test_upgrade_1_1_to_1_2():
    import upgrade_schema_1_2 as U2
    f = dict(schema_version="1.1", image="x", set="draw54", split="val", seen_in_training="unknown", width=5, height=5,
             image_sha256="a" * 64, pairs=[dict(id="t=1", image="x", type="m", A=dict(x=1, y=1, r=1), B=dict(x=3, y=1, r=1),
             confidence="sure", used_in_loss=False, overlay_seen=False, annotator="Yueren", last_edited_by="Yueren",
             history=[dict(op="q", at="2026-11-18T00:00:00Z", by="Yueren", before={}, after={})],
             created="2026-11-18T00:00:00Z", note="", schema_version="1.1")],
             s_seed_log=[dict(index=0, seed="s", x=1, y=1, proposed_at="2026-11-18T00:00:00Z", proposed_by="Yueren", outcome="pending")],
             legacy={}, exhausted={"b": dict(by="Yueren", at="2026-11-18T00:00:00Z", note="n")})
    new = U2.upgrade_file(f, {"yueren": "yueren"}, "b" * 64, "annotator_b", True)
    p = new["pairs"][0]
    assert (p["annotator"], p["last_edited_by"], p["history"][0]["by"], p["schema_version"]) == ("yueren", "yueren", "yueren", "1.2")
    assert new["s_seed_log"][0]["proposed_by"] == "yueren" and new["exhausted"]["b"]["by"] == "yueren"
    assert (new["assigned_to"], new["image_pixel_sha256"], new["reflective_advisory"], new["schema_version"]) == ("annotator_b", "b" * 64, True, "1.2")
    assert U2.upgrade_file(new, {}, "", "", False) is None


def test_upgrade_1_0_to_1_1():
    import upgrade_schema_1_1 as UP
    old = dict(schema_version="1.0", image="x", set="dev_v2", seen_in_training="unknown", width=5, height=5,
               pairs=[dict(id="t=1", image="x", type="m", A=dict(x=1, y=1, r=1), B=dict(x=3, y=1, r=1), confidence="sure",
                           split="eval", overlay_seen="unknown", annotator="Y", created="2026-11-18T00:00:00Z", note="",
                           schema_version="1.0")], s_seed_log=[], legacy={})
    new = UP.upgrade(old, "val")
    assert new["schema_version"] == "1.1" and new["split"] == "val" and list(new)[:4] == ["schema_version", "image", "set", "split"]
    assert "split" not in new["pairs"][0] and new["pairs"][0]["used_in_loss"] is False and new["pairs"][0]["schema_version"] == "1.1"
    assert UP.upgrade(new, "val") is None                                    # idempotent


def test_overlay_archive_roundtrip(cfg, tmp_path, monkeypatch):
    import overlays as OV
    stem = "taiwei_cui_080"
    cache = K.repo_path(cfg["annotate"]["cache"])
    if not (cache / stem / "meta.json").exists():
        pytest.skip("annotation cache not built on this machine (metrics/run.sh annotate/prepare.py)")
    meta = json.loads((cache / stem / "meta.json").read_text())
    if not any("imported" not in o["source"] for o in meta["overlays"]):
        pytest.skip("no overlays built from local detector maps on this machine (export only ships those)")
    arc = tmp_path / "ov.tar.gz"
    OV.export(cfg, arc, stem)
    with tarfile.open(arc) as t:
        names = t.getnames()
    assert "manifest.json" in names and any(n.startswith(stem + "/") for n in names)
    fake = tmp_path / "cache"
    monkeypatch.setitem(cfg["annotate"], "cache", str(fake))
    OV.do_import(cfg, arc)
    imp = json.loads((fake / stem / "imported" / "imported.json").read_text())
    assert set(imp["files"]) <= {"illum", "material", "geom"} and imp["files"]
    bad = tmp_path / "bad.tar.gz"                                             # an archive for a different image file
    with tarfile.open(arc) as t:
        man = json.loads(t.extractfile("manifest.json").read())
        blobs = {n: t.extractfile(n).read() for n in names if n != "manifest.json"}
    man["images"][stem]["image_pixel_sha256"] = "0" * 64
    import io
    with tarfile.open(bad, "w:gz") as t:
        for n, b in [("manifest.json", json.dumps(man).encode())] + list(blobs.items()):
            ti = tarfile.TarInfo(n); ti.size = len(b); t.addfile(ti, io.BytesIO(b))
    fake2 = tmp_path / "cache2"
    monkeypatch.setitem(cfg["annotate"], "cache", str(fake2))
    OV.do_import(cfg, bad)
    assert not (fake2 / stem / "imported").exists()                           # refused


def test_prepare_refuses_hash_mismatch(cfg, tmp_path, monkeypatch):
    sys.path.insert(0, str(M / "annotate"))
    import prepare as PR
    from conftest import need_dataset
    stem = "taiwei_cui_080"
    need_dataset(cfg, [stem])
    ann = tmp_path / "ann"
    ann.mkdir()
    f = json.loads((K.repo_path(cfg["paths"]["annotations"]) / f"{stem}.json").read_text())
    f["image_sha256"] = "f" * 64                                          # file bytes differ, pixels match: accepted
    (ann / f"{stem}.json").write_text(json.dumps(f))
    monkeypatch.setitem(cfg["paths"], "annotations", str(ann))
    PR.check_hash(cfg, stem)
    f["image_pixel_sha256"] = "e" * 64                                    # pixels differ: refused
    (ann / f"{stem}.json").write_text(json.dumps(f))
    with pytest.raises(SystemExit, match="Refusing"):
        PR.check_hash(cfg, stem)


def test_pixel_hash_ignores_metadata(cfg, tmp_path):
    """A re-saved TIFF with Windows-style rating metadata has a different file hash and the same pixel hash."""
    import tifffile
    from core import io as IO
    from conftest import need_dataset
    stem = "taiwei_cui_080"
    need_dataset(cfg, [stem])
    src = K.image_path(cfg, stem)
    a = IO.read_raw(cfg, src)
    dst = tmp_path / "rated.tif"
    tifffile.imwrite(dst, a, compression="lzw", extratags=[(18246, "H", 1, 4, True), (18249, "H", 1, 75, True)])
    assert K.sha256(dst) != K.sha256(src)
    assert IO.pixel_sha256(IO.read_raw(cfg, dst)) == IO.pixel_sha256(a)
    b = a.copy(); b[0, 0, 0] ^= 1
    assert IO.pixel_sha256(b) != IO.pixel_sha256(a)


def test_init_images_writes_schema_1_2(cfg, tmp_path, monkeypatch):
    import shutil
    from conftest import need_dataset
    import init_images
    from core import schema as SC
    stem = "adke_neel_026"
    need_dataset(cfg, [stem])
    shutil.copy2(M / "annotations" / "annotators.json", tmp_path / "annotators.json")
    orig = K.load_config

    def lc(*a, **k):
        c = orig(*a, **k)
        c["paths"]["annotations"] = str(tmp_path)
        return c
    monkeypatch.setattr(K, "load_config", lc)
    monkeypatch.setattr(sys, "argv", ["init_images", "--images", stem, "--assign", "annotator_b"])
    init_images.main()
    f = json.loads((tmp_path / f"{stem}.json").read_text())
    real = json.loads((M / "annotations" / f"{stem}.json").read_text())
    assert SC.validate_file(f) == [] and f["assigned_to"] == "annotator_b"
    assert f["image_pixel_sha256"] == real["image_pixel_sha256"] and f["split"] == real["split"]
    monkeypatch.setattr(sys, "argv", ["init_images", "--images", "x", "--assign", "nobody"])
    with pytest.raises(SystemExit, match="not an id"):
        init_images.main()
