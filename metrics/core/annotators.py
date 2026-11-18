"""Name tags with stable ids (schema 1.2; stdlib only, shared by the server, merge.py and the reports).

annotators.json = {"schema_version": "1.2", "annotators": [{id, name, created, history}]}
  id       a slug fixed at creation (lower-case letters, digits, "_"); pairs store it in annotator / last_edited_by
  name     the current display name; every view and export resolves id -> name
  history  every rename: {from, to, at, by}
A rename changes only annotators.json; no annotation file is rewritten.
"""

import datetime as dt
import json
import os
import re
from pathlib import Path

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
NAME_MAX = 64


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def clean_name(name):
    if any(ord(c) < 32 or ord(c) == 127 for c in str(name)):
        raise ValueError("a name tag cannot contain control characters")
    name = " ".join(str(name).split())
    if not name:
        raise ValueError("a name tag needs at least one visible character")
    if len(name) > NAME_MAX:
        raise ValueError(f"name tag too long ({len(name)} characters; at most {NAME_MAX})")
    return name


def slug(name, taken=()):
    s = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or "annotator"
    if not s[0].isalnum():
        s = "a" + s
    base, k = s, 2
    while s in taken:
        s, k = f"{base}_{k}", k + 1
    return s


def load(path):
    p = Path(path)
    if not p.exists():
        return dict(schema_version="1.2", annotators=[])
    return json.loads(p.read_text(encoding="utf-8"))


def names(path):
    """id -> current name."""
    return {a["id"]: a["name"] for a in load(path)["annotators"]}


def resolve(path):
    """A function id -> current name (unknown ids are shown as the id itself)."""
    m = names(path)
    return lambda i: m.get(i, i) if i else i


def save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _collides(obj, name, except_id=None):
    return next((a for a in obj["annotators"] if a["id"] != except_id and a["name"].casefold() == name.casefold()), None)


def add(obj, name, id_=None):
    """Add a tag; an existing name (case-insensitive) returns that entry instead. Returns (entry, created)."""
    name = clean_name(name)
    hit = _collides(obj, name)
    if hit:
        return hit, False
    taken = {a["id"] for a in obj["annotators"]}
    i = id_ or slug(name, taken)
    if not ID_RE.match(i) or i in taken:
        raise ValueError(f"bad or duplicate id {i!r}")
    e = dict(id=i, name=name, created=now_iso(), history=[])
    obj["annotators"].append(e)
    return e, True


def rename(obj, id_, new_name, by):
    """Rename tag id_ to new_name (refuses empty names and case-insensitive collisions with another tag)."""
    new_name = clean_name(new_name)
    e = next((a for a in obj["annotators"] if a["id"] == id_), None)
    if e is None:
        raise ValueError(f"no name tag with id {id_!r}")
    hit = _collides(obj, new_name, except_id=id_)
    if hit:
        raise ValueError(f"the name {new_name!r} is already used by another tag ({hit['name']})")
    if new_name == e["name"]:
        return e, False
    e["history"].append({"from": e["name"], "to": new_name, "at": now_iso(), "by": by})
    e["name"] = new_name
    return e, True
