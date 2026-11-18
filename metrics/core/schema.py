"""Stdlib-only validator for metrics/schema/pairs.schema.json (no jsonschema in the repo; the Phase 4 server reuses it).

Supports exactly the JSON Schema keywords listed in SUPPORTED; `unsupported_keywords(schema)` lists any others, and
the tests require it to be empty, so the schema cannot silently use a keyword this validator ignores.

    errors = validate_file(obj)            # [] when valid; each error is "path: message"
"""

import json
import re
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "pairs.schema.json"
SCHEMA_VERSION = "1.2"

SUPPORTED = {"$schema", "$id", "$ref", "$defs", "title", "description", "type", "properties", "required",
             "additionalProperties", "items", "enum", "const", "pattern", "minLength", "minimum",
             "allOf", "not", "if", "then", "else"}
_ANNOTATION_ONLY = {"$schema", "$id", "title", "description"}
_SUBSCHEMA_MAPS = {"properties", "$defs"}
_SUBSCHEMA_ONE = {"items", "not", "if", "then", "else", "additionalProperties"}
_SUBSCHEMA_LIST = {"allOf"}

_schema_cache = None


def load_schema(path=SCHEMA_PATH):
    global _schema_cache
    if path == SCHEMA_PATH and _schema_cache is not None:
        return _schema_cache
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    if path == SCHEMA_PATH:
        _schema_cache = s
    return s


def unsupported_keywords(schema):
    """Keywords used anywhere in schema that this validator does not implement."""
    bad = set()

    def walk(s):
        if not isinstance(s, dict):
            return
        for k, v in s.items():
            if k not in SUPPORTED:
                bad.add(k)
            if k in _SUBSCHEMA_MAPS:
                for sub in v.values():
                    walk(sub)
            elif k in _SUBSCHEMA_ONE:
                walk(v)
            elif k in _SUBSCHEMA_LIST:
                for sub in v:
                    walk(sub)
    walk(schema)
    return sorted(bad)


def _type_ok(x, t):
    if t == "object":
        return isinstance(x, dict)
    if t == "array":
        return isinstance(x, list)
    if t == "string":
        return isinstance(x, str)
    if t == "integer":
        return isinstance(x, int) and not isinstance(x, bool)
    if t == "number":
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if t == "boolean":
        return isinstance(x, bool)
    if t == "null":
        return x is None
    raise ValueError(f"unknown type {t!r}")


def _same(a, b):
    """JSON equality: True != 1, 1 == 1.0."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _resolve(ref, root):
    if not ref.startswith("#/"):
        raise ValueError(f"only local $ref supported: {ref}")
    node = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _validate(x, s, root, path, errs):
    if "$ref" in s:
        _validate(x, _resolve(s["$ref"], root), root, path, errs)
    if "type" in s:
        ts = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_type_ok(x, t) for t in ts):
            errs.append(f"{path}: expected {'/'.join(ts)}, got {type(x).__name__}")
            return
    if "const" in s and not _same(x, s["const"]):
        errs.append(f"{path}: must be {json.dumps(s['const'])}")
    if "enum" in s and not any(_same(x, e) for e in s["enum"]):
        errs.append(f"{path}: {json.dumps(x)} not in {json.dumps(s['enum'])}")
    if isinstance(x, str):
        if "minLength" in s and len(x) < s["minLength"]:
            errs.append(f"{path}: shorter than {s['minLength']}")
        if "pattern" in s and not re.search(s["pattern"], x):
            errs.append(f"{path}: {x!r} does not match {s['pattern']}")
    if isinstance(x, (int, float)) and not isinstance(x, bool) and "minimum" in s and x < s["minimum"]:
        errs.append(f"{path}: {x} < minimum {s['minimum']}")
    if isinstance(x, dict):
        props = s.get("properties", {})
        for k in s.get("required", []):
            if k not in x:
                errs.append(f"{path}: missing required '{k}'")
        for k, v in x.items():
            if k in props:
                _validate(v, props[k], root, f"{path}.{k}", errs)
            elif "additionalProperties" in s:
                ap = s["additionalProperties"]
                if ap is False:
                    errs.append(f"{path}: unexpected field '{k}'")
                elif isinstance(ap, dict):
                    _validate(v, ap, root, f"{path}.{k}", errs)
    if isinstance(x, list) and "items" in s:
        for i, v in enumerate(x):
            _validate(v, s["items"], root, f"{path}[{i}]", errs)
    for sub in s.get("allOf", []):
        _validate(x, sub, root, path, errs)
    if "not" in s:
        e2 = []
        _validate(x, s["not"], root, path, e2)
        if not e2:
            errs.append(f"{path}: must not match {json.dumps(s['not'])}")
    if "if" in s:
        e2 = []
        _validate(x, s["if"], root, path, e2)
        branch = s.get("then") if not e2 else s.get("else")
        if branch is not None:
            _validate(x, branch, root, path, errs)


def validate(obj, schema=None):
    schema = schema if schema is not None else load_schema()
    errs = []
    _validate(obj, schema, schema, "$", errs)
    return errs


def validate_file(obj, schema=None):
    """Schema check plus the cross-field rules a JSON Schema cannot state."""
    errs = validate(obj, schema)
    if errs:
        return errs
    ids = [p["id"] for p in obj["pairs"]]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errs.append(f"$.pairs: duplicate ids {dup}")
    for k, p in enumerate(obj["pairs"]):
        if p["image"] != obj["image"]:
            errs.append(f"$.pairs[{k}].image: {p['image']!r} differs from file image {obj['image']!r}")
        if p["schema_version"] != obj["schema_version"]:
            errs.append(f"$.pairs[{k}].schema_version differs from the file's")
    return errs
