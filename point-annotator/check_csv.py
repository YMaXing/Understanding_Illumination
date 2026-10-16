import csv, glob, os, sys, cv2
IMG_DIR, OUT = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(f"{OUT}/*.csv")):
    rows = list(csv.DictReader(open(f)))
    if not rows:
        print(f, "empty"); continue
    h, w = cv2.imread(os.path.join(IMG_DIR, rows[0]["image"]), cv2.IMREAD_UNCHANGED).shape[:2]
    labels, xy, region_super, super_labels = {}, set(), {}, {}
    for r in rows:
        x, y = int(r["x"]), int(r["y"])
        assert 0 <= x < w and 0 <= y < h, (f, r)
        assert (x, y) not in xy, ("duplicate pixel", f, r)
        xy.add((x, y))
        rid = int(r["region_id"])
        labels.setdefault(rid, set()).add(r["label"])
        sid = int(r.get("super_region_id") or 0)  # older 5-column files: no supers
        region_super.setdefault(rid, set()).add(sid)
        if sid:
            super_labels.setdefault(sid, set()).add(r["label"])
    assert all(len(v) == 1 for v in labels.values()), (f, labels)
    assert sorted(labels) == list(range(1, len(labels) + 1)), (f, sorted(labels))
    assert all(len(v) == 1 for v in region_super.values()), ("region in several supers", f, region_super)
    assert all(len(v) == 1 for v in super_labels.values()), ("super mixes labels", f, super_labels)
    assert sorted(super_labels) == list(range(1, len(super_labels) + 1)), (f, sorted(super_labels))
    print(f, "OK:", len(rows), "points,", len(labels), "regions,", len(super_labels), "super-regions")
