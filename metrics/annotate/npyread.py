"""Read run_metrics.py's profiles.npz with the standard library only (zipfile + the .npy format), for the review page.

Supports what profiles.npz holds: little-endian float32 arrays ('<f4'), unicode arrays ('<U..') and the structured
index (method '<i4', image '<U64', id '<U32', start '<i8', stop '<i8').
"""

import ast
import struct
import zipfile
from array import array


def _header(f):
    if f.read(6) != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = f.read(2)[0]
    n = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
    h = ast.literal_eval(f.read(n).decode("latin1"))
    if h.get("fortran_order"):
        raise ValueError("fortran order not supported")
    return h


def _field_size(code):
    kind, size = code[1], int(code[2:])
    return size * 4 if kind == "U" else size


def _decode(code, b):
    kind = code[1]
    if kind == "U":
        return b.decode("utf-32-le").rstrip("\x00")
    return struct.unpack({"i4": "<i", "i8": "<q", "f4": "<f", "f8": "<d"}[code[1:]], b)[0]


def load(path):
    """dict name -> value: float32 arrays as (array('f'), shape); unicode arrays as lists; the index as list of dicts."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                h = _header(f)
                data = f.read()
            key = name[:-4] if name.endswith(".npy") else name
            d, shape = h["descr"], h["shape"]
            if isinstance(d, list):
                fields = [(n, c, _field_size(c)) for n, c in d]
                size = sum(s for _, _, s in fields)
                rows = []
                for k in range(len(data) // size):
                    rec, off = {}, k * size
                    for n, c, s in fields:
                        rec[n] = _decode(c, data[off:off + s])
                        off += s
                    rows.append(rec)
                out[key] = rows
            elif d == "<f4":
                a = array("f")
                a.frombytes(data)
                out[key] = (a, shape)
            elif d[:2] == "<U":
                s = _field_size(d)
                out[key] = [data[k:k + s].decode("utf-32-le").rstrip("\x00") for k in range(0, len(data), s)]
            else:
                raise ValueError(f"{name}: dtype {d} not supported")
    return out
