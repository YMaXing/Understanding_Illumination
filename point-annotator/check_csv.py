import csv, glob, os, sys, cv2
IMG_DIR, OUT = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(f"{OUT}/*.csv")):
    rows = list(csv.DictReader(open(f)))
    if not rows:
        print(f, "empty"); continue
    h, w = cv2.imread(os.path.join(IMG_DIR, rows[0]["image"]), cv2.IMREAD_UNCHANGED).shape[:2]
    labels, xy = {}, set()
    for r in rows:
        x, y = int(r["x"]), int(r["y"])
        assert 0 <= x < w and 0 <= y < h, (f, r)
        assert (x, y) not in xy, ("duplicate pixel", f, r)
        xy.add((x, y))
        labels.setdefault(int(r["region_id"]), set()).add(r["label"])
    assert all(len(v) == 1 for v in labels.values()), (f, labels)
    assert sorted(labels) == list(range(1, len(labels) + 1)), (f, sorted(labels))
    print(f, "OK:", len(rows), "points,", len(labels), "regions")
