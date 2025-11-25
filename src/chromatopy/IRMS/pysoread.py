# isodat_min_reader.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os, re
import math
import struct
from statistics import median
import pandas as pd

# ---------------- Data structures ----------------
@dataclass
class BinaryIsodatFile:
    raw: bytes = b""
    max_pos: int = 0
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    C_blocks: List[Dict[str, Any]] = field(default_factory=list)
    current_nav_block_idx: int = 1
    class_tag: str = "binary_isodat_file"

def template_binary_file_object() -> BinaryIsodatFile:
    return BinaryIsodatFile(class_tag="binary_file")

def template_binary_isodat_file_object() -> BinaryIsodatFile:
    b = template_binary_file_object()
    b.blocks = []
    b.current_nav_block_idx = 1
    b.class_tag = "binary_isodat_file"
    return b

# ---------------- Basic file I/O ----------------
def read_binary_file(filepath: str, bfile: Optional[BinaryIsodatFile] = None) -> BinaryIsodatFile:
    if not os.path.exists(filepath) or os.path.isdir(filepath):
        raise FileNotFoundError(f"file does not exist or is a directory: {filepath}")
    if bfile is None:
        bfile = template_binary_file_object()
    with open(filepath, "rb") as f:
        data = f.read()
    bfile.raw = data
    bfile.max_pos = len(data)
    return bfile

# ---------------- Helpers for parsing ----------------
def _read_int_le_from_two_bytes(b0: int, b1: int) -> int:
    val = (b1 << 8) | b0
    if val >= 0x8000:
        val -= 0x10000
    return val

def get_isodat_control_blocks_config():
    cfg = []

    # C block
    cblock_re = re.compile(
        b"\xff\xff(?:\x00|[\x01-\x0f])\x00.\x00\x43[\x20-\x7e]", re.DOTALL
    )
    def c_start_fn(pos_list): return pos_list
    def c_len_fn(raw: bytes, starts):
        out = []
        for s in starts:
            if s + 5 >= len(raw):
                out.append(0); continue
            n = _read_int_le_from_two_bytes(raw[s+4], raw[s+5])
            out.append(6 + n)
        return out
    def c_data_len_fn(lens): return [max(0, L - 6) for L in lens]
    def c_block_fn(raw: bytes, starts, data_lens):
        out = []
        for s, dl in zip(starts, data_lens):
            beg = s + 6; end = beg + dl
            out.append(raw[beg:end].decode("latin-1", errors="replace") if dl > 0 else "")
        return out
    cfg.append(dict(type="C block", regex=cblock_re, start_fn=c_start_fn,
                    len_fn=c_len_fn, data_len_fn=c_data_len_fn, block_fn=c_block_fn))

    # text block (UTF-16LE)
    text_re = re.compile(b"\xff\xfe\xff", re.DOTALL)
    def t_start_fn(pos_list): return pos_list
    def t_len_fn(raw: bytes, starts):
        out = []
        for s in starts:
            if s + 3 >= len(raw): out.append(0); continue
            n = raw[s+3]
            out.append(4 + n * 2)
        return out
    def t_data_len_fn(lens): return [max(0, (L - 4)//2) for L in lens]
    def t_block_fn(raw: bytes, starts, data_lens):
        out = []
        for s, dl in zip(starts, data_lens):
            y = dl * 2; beg = s + 4; end = beg + y
            out.append(raw[beg:end].decode("utf-16-le", errors="replace") if y > 0 else "")
        return out
    cfg.append(dict(type="text", regex=text_re, start_fn=t_start_fn,
                    len_fn=t_len_fn, data_len_fn=t_data_len_fn, block_fn=t_block_fn))

    # x-000 block
    x000_re = re.compile(b"[\x01-\x1f]\x00{3}", re.DOTALL)
    def x_start_fn(pos_list): return pos_list
    def x_len_fn(raw: bytes, starts): return [4]*len(starts)
    def x_data_len_fn(lens): return [0]*len(lens)
    def x_block_fn(raw: bytes, starts, _): return [f"{raw[s]}-000" for s in starts]
    cfg.append(dict(type="x-000", regex=x000_re, start_fn=x_start_fn,
                    len_fn=x_len_fn, data_len_fn=x_data_len_fn, block_fn=x_block_fn))

    # 0000+ block (runs of 00 ending in non-zero; we subtract that last byte)
    zplus_re = re.compile(b"(?:\x00\x00){2,}[\x01-\xff]", re.DOTALL)
    def z_start_fn(pos_list): return pos_list
    def z_len_fn(_raw: bytes, _starts): return None  # handled below
    def z_data_len_fn(lens): return [0]*len(lens)
    def z_block_fn(_raw: bytes, _starts, lens): return [f"{L}x00" for L in lens]
    cfg.append(dict(type="0000+", regex=zplus_re, start_fn=z_start_fn,
                    len_fn=z_len_fn, data_len_fn=z_data_len_fn, block_fn=z_block_fn))
    return cfg

def find_pattern_blocks(raw: bytes, *, type: str, regex, start_fn, len_fn, data_len_fn, block_fn):
    positions = [m.start() for m in regex.finditer(raw)]
    if not positions:
        return []
    starts = start_fn(positions)

    if type == "0000+":
        lens = []
        for m in regex.finditer(raw):
            L = (m.end() - m.start()) - 1
            lens.append(L if L > 0 else 0)
        lens = lens[:len(starts)]
    else:
        lens = len_fn(raw, starts)

    ends = [s + L - 1 if L > 0 else s - 1 for s, L in zip(starts, lens)]
    data_lens = data_len_fn(lens)
    blocks_text = block_fn(raw, starts, data_lens if type != "0000+" else lens)

    out = []
    for s, e, L, dl, bt in zip(starts, ends, lens, data_lens, blocks_text):
        if L > 0:
            out.append(dict(type=type, start=s, end=e, len=L, data_len=dl, block=bt or ""))
    return out

def find_unknown_blocks(*, raw: bytes, blocks: List[Dict[str, Any]]):
    if not blocks:
        return [dict(type="unknown", start=0, end=len(raw)-1 if len(raw) else -1,
                     len=len(raw), data_len=len(raw), priority=1, block="")]
    bs = sorted(blocks, key=lambda d: d["start"])
    max_pri = max(d.get("priority", 0) for d in bs)
    out = []
    # leading
    if bs[0]["start"] > 0:
        out.append(dict(type="unknown", start=0, end=bs[0]["start"]-1,
                        len=bs[0]["start"], data_len=bs[0]["start"],
                        priority=max_pri+1, block=""))
    # between
    for a, b in zip(bs, bs[1:]):
        s = a["end"] + 1; e = b["start"] - 1
        if e >= s:
            L = e - s + 1
            out.append(dict(type="unknown", start=s, end=e, len=L, data_len=L,
                            priority=max_pri+1, block=""))
    # trailing
    if bs[-1]["end"] < len(raw) - 1:
        s = bs[-1]["end"] + 1; e = len(raw) - 1; L = e - s + 1
        out.append(dict(type="unknown", start=s, end=e, len=L, data_len=L,
                        priority=max_pri+1, block=""))
    return out

def get_unknown_blocks_text(*, blocks: List[Dict[str, Any]], raw: bytes, unknown_block_n_chars: int = 8):
    out = []
    for b in blocks:
        if b.get("type") == "unknown" and (b["start"] + unknown_block_n_chars) < len(raw):
            preview_bytes = raw[b["start"] : b["start"] + unknown_block_n_chars]
            preview = " ".join(str(int(x)) for x in preview_bytes) + " ..."
            max_chars = max(0, b["data_len"] * 3 - 1)
            b["block"] = preview[:max_chars]
        out.append(b)
    return out

def _assert_isodat(bfile: BinaryIsodatFile):
    if bfile.class_tag != "binary_isodat_file":
        raise TypeError("this function is for isodat binary files only")

def find_isodat_structure_blocks(bfile: BinaryIsodatFile, unknown_block_n_chars: int = 8):
    _assert_isodat(bfile)
    cfg = get_isodat_control_blocks_config()

    ctrl_blocks: List[Dict[str, Any]] = []
    for i, row in enumerate(cfg, start=1):
        found = find_pattern_blocks(
            bfile.raw,
            type=row["type"],
            regex=row["regex"],
            start_fn=row["start_fn"],
            len_fn=row["len_fn"],
            data_len_fn=row["data_len_fn"],
            block_fn=row["block_fn"],
        )
        for blk in found:
            blk["priority"] = i
            ctrl_blocks.append(blk)

    unknown_blocks = find_unknown_blocks(raw=bfile.raw, blocks=ctrl_blocks)
    unknown_blocks = get_unknown_blocks_text(blocks=unknown_blocks, raw=bfile.raw, unknown_block_n_chars=unknown_block_n_chars)

    all_blocks = ctrl_blocks + unknown_blocks
    all_blocks.sort(key=lambda d: d.get("start", 0))
    for idx, blk in enumerate(all_blocks, start=1):
        blk["block_idx"] = idx
        blk.setdefault("block", "")
    return all_blocks

# ---------------- Public API ----------------
def read_binary_isodat_file(filepath: str) -> BinaryIsodatFile:
    bfile = read_binary_file(filepath, bfile=template_binary_isodat_file_object())
    bfile.blocks = find_isodat_structure_blocks(bfile)
    bfile.C_blocks = [blk for blk in bfile.blocks if blk.get("type") == "C block"]
    return bfile

# Convenience: JSON-safe dict for inspection/serialization
def to_dict(bfile: BinaryIsodatFile) -> Dict[str, Any]:
    return {
        "size_bytes": bfile.max_pos,
        "num_blocks": len(bfile.blocks),
        "blocks": [
            {
                "block_idx": blk.get("block_idx"),
                "type": blk.get("type"),
                "start": blk.get("start"),
                "end": blk.get("end"),
                "len": blk.get("len"),
                "data_len": blk.get("data_len"),
                "priority": blk.get("priority"),
                "preview": blk.get("block")[:80] if isinstance(blk.get("block"), str) else "",
            }
            for blk in bfile.blocks
        ],
        "num_C_blocks": len(bfile.C_blocks),
    }

# --- Helpers to extract raw bytes from blocks ---

def get_block_bytes(bfile, block_idx: int) -> bytes:
    """Return the exact bytes of the block (header + payload) by 1-based block_idx."""
    blk = next(b for b in bfile.blocks if b["block_idx"] == block_idx)
    return bfile.raw[blk["start"] : blk["end"] + 1]

def iter_blocks_by_type(bfile, type_name: str):
    """Yield (blk, bytes) for all blocks of a given type."""
    for blk in bfile.blocks:
        if blk["type"] == type_name:
            yield blk, bfile.raw[blk["start"] : blk["end"] + 1]

def get_cblock_payload_bytes(bfile, blk) -> bytes:
    """
    For a C block, return just the payload bytes (after the 6-byte header).
    Uses the already-computed 'data_len'.
    """
    assert blk["type"] == "C block"
    start = blk["start"] + 6
    end = start + blk["data_len"]
    return bfile.raw[start:end]

def decode_text_block(bfile, blk) -> str:
    """
    For a 'text' block, return the decoded UTF-16LE string.
    (The regex we used targets the BOM 0xFF 0xFE 0xFF pattern.)
    """
    assert blk["type"] == "text"
    # text payload begins at start+4 and has byte length = (len-4)
    beg = blk["start"] + 4
    end = blk["start"] + blk["len"]
    return bfile.raw[beg:end].decode("utf-16-le", errors="replace")

# %% New functions:
# ---------------- Cursor / state helpers ----------------
def check_bfile(bfile):
    if getattr(bfile, "class_tag", None) != "binary_isodat_file":
        raise TypeError("this function is for isodat binary files only")
    if not hasattr(bfile, "pos"):
        bfile.pos = 0
    if not hasattr(bfile, "max_pos"):
        bfile.max_pos = len(bfile.raw)
    if not hasattr(bfile, "data"):
        bfile.data = {}
    if not hasattr(bfile, "error_prefix"):
        bfile.error_prefix = ""

def iso_source_file_op_error(bfile, msg):
    # R prepends an optional prefix; mirror that behavior.
    prefix = getattr(bfile, "error_prefix", "")
    raise RuntimeError(f"{prefix}{msg}")

def set_binary_file_error_prefix(bfile, prefix=""):
    check_bfile(bfile)
    bfile.error_prefix = (prefix + " - ") if prefix else ""
    return bfile

def move_to_pos(bfile, pos, reset_cap=False):
    check_bfile(bfile)
    if reset_cap:
        bfile.max_pos = len(bfile.raw)
    if pos > bfile.max_pos:
        iso_source_file_op_error(bfile, f"cannot move to position {pos:.0f} as it exceeds position max set at {bfile.max_pos:.0f}")
    bfile.pos = int(pos)
    return bfile

def cap_at_pos(bfile, pos):
    check_bfile(bfile)
    if pos is None:
        raise ValueError("cannot cap at position NULL")
    if pos < bfile.pos:
        iso_source_file_op_error(bfile, f"cannot cap at position {pos:.0f} as it is smaller than current position {bfile.pos:.0f}")
    bfile.max_pos = int(pos)
    return bfile

def skip_pos(bfile, nbyte):
    return move_to_pos(bfile, bfile.pos + int(nbyte))

# # ---------------- C-block navigation (requires name matching) ----------------
# def fetch_C_block(bfile, C_block, min_pos=0, occurence=1, regexp_match=False):
#     """
#     TODO: Port your exact R 'fetch_C_block'. For now:
#     - scan bfile.C_blocks in file order
#     - require blk['start'] >= min_pos
#     - decode payload (latin-1) and match either exact name or regex
#     - return the Nth (occurence) match as a dict like R tibble row
#     """
#     import re
#     check_bfile(bfile)
#     count = 0
#     pat = re.compile(C_block) if regexp_match else None
#     for blk in bfile.C_blocks:
#         if blk["start"] < min_pos:
#             continue
#         name = get_cblock_payload_bytes(bfile, blk).decode("latin-1", errors="replace")
#         ok = (pat.search(name) is not None) if regexp_match else (name == C_block)
#         if ok:
#             count += 1
#             if count == occurence:
#                 # Return fields to mimic R tibble (at least start/end)
#                 return {"start": blk["start"], "end": blk["end"], "len": blk["len"], "data_len": blk["data_len"], "type": blk["type"]}
#     return None

# def cap_at_C_block(bfile, C_block, min_pos=0, occurence=1, regexp_match=False):
#     """
#     TODO: Port exactly. Common behavior: cap at the START of the target C-block.
#     """
#     cblk = fetch_C_block(bfile, C_block, min_pos=min_pos, occurence=occurence, regexp_match=regexp_match)
#     if cblk is None:
#         iso_source_file_op_error(bfile, f"cannot cap at C block '{C_block}'")
#     return cap_at_pos(bfile, cblk["start"])

def move_to_C_block(bfile, C_block, min_pos=0, occurence=1, move_to_end=True, reset_cap=True, regexp_match=False):
    cblock = fetch_C_block(bfile, C_block, min_pos=min_pos, occurence=occurence, regexp_match=regexp_match)
    if cblock is None:
        iso_source_file_op_error(bfile, f"cannot move to C block '{C_block}'")
    if reset_cap:
        bfile.max_pos = len(bfile.raw)
    new_pos = (cblock["end"] + 1) if move_to_end else cblock["start"]
    return move_to_pos(bfile, new_pos)

def move_to_next_C_block(bfile, C_block, reset_cap=False, regexp_match=False):
    return move_to_C_block(bfile, C_block, min_pos=bfile.pos, occurence=1, move_to_end=True, reset_cap=reset_cap, regexp_match=regexp_match)

def move_to_C_block_range(bfile, from_C_block, to_C_block, min_pos=0):
    from_blk = fetch_C_block(bfile, from_C_block, min_pos=min_pos, occurence=1)
    to_blk   = fetch_C_block(bfile, to_C_block,   min_pos=min_pos, occurence=1)
    if (from_blk is not None) and (to_blk is not None) and (to_blk["start"] < from_blk["end"]):
        iso_source_file_op_error(
            bfile,
            f"cannot move to Cblock range, first occurence of block '{from_C_block}' before first block '{to_C_block}'"
        )
    bfile = move_to_C_block(bfile, from_C_block, min_pos=min_pos, reset_cap=True)
    bfile = cap_at_C_block(bfile, to_C_block, min_pos=min_pos)
    return bfile

def move_to_next_C_block_range(bfile, from_C_block, to_C_block):
    return move_to_C_block_range(bfile, from_C_block, to_C_block, min_pos=bfile.pos)

# ---------------- Regex builders ----------------
class BinaryRegexp:
    def __init__(self, label: str, regexp: bytes, size: int):
        self.label = label
        self.regexp = regexp  # bytes-like regex pattern
        self.size = int(size)

# def combine_regexps(regexps, collapse=b""):
#     """
#     TODO: Port your exact combining semantics if different.
#     Concatenate byte-regex patterns in order.
#     """
#     pat = b"".join(r.regexp if isinstance(r.regexp, (bytes, bytearray)) else r.regexp.encode("latin-1") for r in regexps)
#     return pat

def re_combine(*regexps: BinaryRegexp) -> BinaryRegexp:
    # safety
    for r in regexps:
        if not isinstance(r, BinaryRegexp):
            raise TypeError("can only combine binary regexps, generate with re_x() functions")
    label = "".join([r.label for r in regexps])
    size = sum([r.size for r in regexps])
    regexp = combine_regexps(regexps)
    return BinaryRegexp(label=label, regexp=regexp, size=size)

def re_x_000():
    return BinaryRegexp(label="<x-000>", regexp=b"[\x01-\x1f]\x00{3}", size=4)

def re_text_0():
    return BinaryRegexp(label="{text-0}", regexp=b"\xff\xfe\xff\x00", size=4)

def re_text_x():
    return BinaryRegexp(label="{text-x}", regexp=b"\xff\xfe\xff.", size=4)

def re_null(n: int):
    return BinaryRegexp(label=f"<{n}x00>", regexp=fr"\x00{{{n}}}".encode("latin-1"), size=int(n))

def re_not_null(n: int):
    return BinaryRegexp(label=f"<x01-xff{{{n}}}>", regexp=fr"[\x01-\xff]{{{n}}}".encode("latin-1"), size=int(n))

# def get_ctrl_blocks_config():
#     """
#     TODO: Provide your R get_ctrl_blocks_config() mapping.
#     Must return dict-like for names used in re_block(): 'alpha','text','text0','stx', etc.
#     Each entry must have fields: 'regexp' (bytes regex), 'size' (int).
#     """
#     raise NotImplementedError("get_ctrl_blocks_config() not provided yet")

def re_block(block_name: str):
    cfg = get_ctrl_blocks_config()
    block = cfg.get(block_name)
    if block is None:
        raise ValueError(f"encountered unknown block: {block_name}")
    return BinaryRegexp(label=f"<{block_name}>", regexp=block["regexp"], size=int(block["size"]))

def re_direct(regexp: str, label="re-direct", size=None):
    b = regexp if isinstance(regexp, (bytes, bytearray)) else regexp.encode("latin-1")
    if size is None:
        size = len(b)
    return BinaryRegexp(label=f"[{label}]", regexp=b, size=int(size))

def re_unicode(text: str):
    # Build UTF-16LE literal: each ASCII byte followed by \x00, as a regex
    # Example: "ABC" -> b"\x41\x00\x42\x00\x43\x00"
    raw = text.encode("latin-1")
    # represent as direct byte sequence in the regex:
    seq = b"".join(bytes([c, 0x00]) for c in raw)
    return BinaryRegexp(label=f"{{{text}}}", regexp=seq, size=2 * len(text))

# ---------------- Pattern search ----------------
import re as _re

def _search_bytes(pattern_bytes: bytes, haystack: bytes, offset: int = 0, all_matches: bool = False):
    # compile as bytes regex; DOTALL so '.' crosses bytes
    rx = _re.compile(pattern_bytes, flags=_re.DOTALL)
    if all_matches:
        return [m.start() + offset for m in rx.finditer(haystack[offset:])]
    m = rx.search(haystack, pos=offset)
    return (m.start() if m else None)

# def find_next_pattern(bfile, *regexps: BinaryRegexp, max_gap=None, value=False, all=False):
#     check_bfile(bfile)
#     comb = re_combine(*regexps)
#     pos_or_list = _search_bytes(comb.regexp, bfile.raw, offset=bfile.pos, all_matches=all)
#     # normalize to None if nothing
#     if all:
#         positions = [p for p in pos_or_list if p <= bfile.max_pos]
#         return positions if positions else None
#     pos = pos_or_list
#     if pos is None:
#         return None
#     if (max_gap is not None) and (pos > bfile.pos + max_gap):
#         return None
#     if pos > bfile.max_pos:
#         return None
#     if value:
#         # return match length (R's grepRaw(value=TRUE) -> matched bytes)
#         return comb.size
#     return pos
def find_next_pattern(bfile, *regexps: BinaryRegexp, max_gap=None, value=False, all=False):
    check_bfile(bfile)
    comb = re_combine(*regexps)
    rx = _re.compile(comb.regexp, flags=_re.DOTALL)

    if all:
        matches = [m.start() + 0 for m in rx.finditer(bfile.raw, pos=bfile.pos)]
        matches = [p for p in matches if p <= bfile.max_pos]
        return matches if matches else None

    m = rx.search(bfile.raw, pos=bfile.pos)
    if not m:
        return None
    if (max_gap is not None) and (m.start() > bfile.pos + max_gap):
        return None
    if m.start() > bfile.max_pos:
        return None

    if value:
        return m.end() - m.start()   # <-- actual matched byte length

    return m.start()

def find_next_patterns(bfile, *regexps: BinaryRegexp):
    positions = find_next_pattern(bfile, *regexps, value=False, all=True)
    if not positions:
        return None
    positions = [p for p in positions if p <= bfile.max_pos]
    return positions if positions else None

def move_to_next_pattern(bfile, *regexps: BinaryRegexp, max_gap=None, move_to_end=True):
    check_bfile(bfile)
    if (max_gap is not None) and not isinstance(max_gap, (int, float)):
        raise ValueError("max gap must be a number")
    pos = find_next_pattern(bfile, *regexps, max_gap=max_gap)
    if pos is not None:
        n = find_next_pattern(bfile, *regexps, value=True) if move_to_end else 0
        return move_to_pos(bfile, pos + n)
    # error path
    comb = re_combine(*regexps)
    gap_text = f" after maximal gap of {max_gap:.0f} bytes" if max_gap is not None else ""
    preview_len = comb.size + (int(max_gap) if max_gap is not None else 50) + 10
    found_preview = iso_print_source_file_structure(bfile, length=preview_len)
    iso_source_file_op_error(
        bfile,
        f"could not find '{comb.label}'{gap_text} in search interval {bfile.pos:.0f} to {bfile.max_pos:.0f}, found '{found_preview}...'"
    )

# ---------------- Data capture ----------------
# def parse_raw_data(raw_slice: bytes, types, ignore_trailing_zeros=True, exact_length=True, sensible=None, errors=""):
#     """
#     TODO: Port your exact R parser.
#     Must support at least:
#       - 'text' (decode UTF-16LE or ascii per your usage)
#       - 'float' (4-byte LE)
#       - 'double' (8-byte LE)
#       - 'raw' (pass-through)
#     For now, raise to signal we need your R version.
#     """
#     raise NotImplementedError("parse_raw_data() not provided yet")

def iso_print_source_file_structure(bfile, length: int = 80):
    """Tiny helper for error messages (preview bytes from current pos)."""
    end = min(bfile.pos + max(0, int(length)), len(bfile.raw))
    chunk = bfile.raw[bfile.pos:end]
    # show as hex groups
    return " ".join(f"{b:02x}" for b in chunk)

def capture_data_till_pattern(
    bfile,
    id: str,
    type,
    *regexps: BinaryRegexp,
    data_bytes_min: int = 0,
    data_bytes_max: int | None = None,
    move_past_dots: bool = False,
    re_size_exact: bool = True,
    ignore_trailing_zeros: bool = True,
    exact_length: bool = True,
    sensible=None,
):
    # reset
    check_bfile(bfile)
    if not hasattr(bfile, "data"):
        bfile.data = {}
    bfile.data[id] = None

    comb = re_combine(*regexps)

    # move to beginning of target (just before the pattern), capturing data
    start = bfile.pos
    search_start = start + int(data_bytes_min)
    if data_bytes_max is not None:
        # cap search window by max bytes
        cap = min(bfile.max_pos, search_start + int(data_bytes_max))
    else:
        cap = bfile.max_pos

    # manual search window
    rx = _re.compile(comb.regexp, flags=_re.DOTALL)
    m = rx.search(bfile.raw, pos=search_start, endpos=cap)
    if not m:
        # delegate to move_to_next_pattern to raise a consistent error
        move_to_next_pattern(bfile, comb, max_gap=data_bytes_max, move_to_end=False)
        # (won't reach here)
    end = m.start()  # exclusive
    # store data
    if end > start:
        if isinstance(type, str) and type == "raw":
            bfile.data[id] = bfile.raw[start:end]
        else:
            id_text = f"'{id}' capture failed at pos {start}: "
            bfile.data[id] = parse_raw_data(
                bfile.raw[start:end],
                type,
                ignore_trailing_zeros=ignore_trailing_zeros,
                exact_length=exact_length,
                sensible=sensible,
                errors=(bfile.error_prefix + id_text),
            )
    # move past dots?
    if move_past_dots:
        if re_size_exact:
            bfile = move_to_pos(bfile, end + comb.size)
        else:
            bfile = move_to_next_pattern(bfile, comb, max_gap=0)
    else:
        bfile = move_to_pos(bfile, m.start())
    return bfile

# --- PORTS FROM YOUR R HELPERS ---

# fetch_C_block / cap_at_C_block
def fetch_C_block(bfile, C_block: str, min_pos: int = 0, occurence: int | None = None, regexp_match: bool = False):
    check_bfile(bfile)
    if not bfile.C_blocks:
        raise RuntimeError("no C_blocks available")
    if not isinstance(C_block, str) or not C_block:
        raise RuntimeError("C_block name not provided")

    import re
    pat = re.compile(C_block) if regexp_match else None

    # Build list as in R filter(C_blocks, block == <>, start >= min_pos)
    matches = []
    for blk in bfile.C_blocks:
        if blk["start"] < min_pos:
            continue
        payload = get_cblock_payload_bytes(bfile, blk).decode("latin-1", errors="replace")
        ok = (pat.search(payload) is not None) if regexp_match else (payload == C_block)
        if ok:
            # Return a tibble-like row (dict) with the important fields
            row = {
                "start": blk["start"],
                "end": blk["end"],
                "len": blk["len"],
                "data_len": blk["data_len"],
                "type": blk["type"],
                "block": payload,
            }
            matches.append(row)

    if not matches:
        iso_source_file_op_error(bfile, f"block '{C_block}' not found after position {min_pos:.0f}")

    if occurence is not None and occurence > 1 and len(matches) < occurence:
        iso_source_file_op_error(
            bfile,
            f"occurence {occurence:.0f} of block '{C_block}' not found (only {len(matches):.0f} occurences) after position {min_pos:.0f}",
        )

    if occurence is not None and occurence == -1:
        return matches[-1]
    elif occurence is not None:
        return matches[occurence - 1]
    else:
        return matches  # list of rows

def cap_at_C_block(bfile, C_block: str, min_pos: int = 0, occurence: int = 1, regexp_match: bool = False):
    cblk = fetch_C_block(bfile, C_block, min_pos=min_pos, occurence=occurence, regexp_match=regexp_match)
    if cblk is None:
        iso_source_file_op_error(bfile, f"cannot cap at C block '{C_block}'")
    return cap_at_pos(bfile, cblk["start"])

def cap_at_next_C_block(bfile, C_block: str, regexp_match: bool = False):
    return cap_at_C_block(bfile, C_block, min_pos=bfile.pos, occurence=1, regexp_match=regexp_match)

# control-block config for re_block()
def get_ctrl_blocks_config():
    # Each entry needs 'size' and 'regexp' as **bytes regex** patterns.
    def B(s: str) -> bytes:
        # interpret backslash escapes like \x00 literally
        return s.encode("latin-1", "backslashreplace").decode("unicode_escape").encode("latin-1")

    return {
        # specific sequences
        "del-nl":   {"size": 2, "auto": True,  "regexp": B("\x7f\x85")},
        "eop-nl":   {"size": 2, "auto": True,  "regexp": B("\xdc\x85")},
        "0b-80":    {"size": 2, "auto": True,  "regexp": B("\x0b\x80")},
        "e0-81":    {"size": 2, "auto": True,  "regexp": B("\xe0\x81")},
        "ce-80":    {"size": 2, "auto": True,  "regexp": B("\xce\x80")},
        "ce-8a":    {"size": 2, "auto": True,  "regexp": B("\xce\x8a")},
        "ee-85":    {"size": 2, "auto": True,  "regexp": B("\xee\x85")},
        "75-84":    {"size": 2, "auto": True,  "regexp": B("\x75\x84")},
        "ff-80":    {"size": 4, "auto": True,  "regexp": B(r"\x00\xff\x80\x00")},
        "07-80-id": {"size": 6, "auto": True,  "regexp": B("\x05\x80.\xff(\\x00|\x80|\xff){2}")},
        "54-fc":    {"size": 4, "auto": True,  "regexp": B("\x54\xfc\x00\x00")},

        # FFF/EEE combos
        "nl":       {"size": 4, "auto": True,  "regexp": B("\xff\xfe\xff\x0a")},
        "eee-0":    {"size": 4, "auto": True,  "regexp": B("\xef\xef\xef\x00.")},
        "ffff":     {"size": 4, "auto": True,  "regexp": B("\xff{4}")},

        # x000 blocks
        "stx":      {"size": 4, "auto": True,  "regexp": B("\x02\x00\x00\x00")},
        "etx":      {"size": 4, "auto": True,  "regexp": B("\x03\x00\x00\x00")},
        "f-000":    {"size": 4, "auto": True,  "regexp": B("\xff\x00\x00\x00")},

        # C-block (not auto)
        "C-block":  {"size": 20, "auto": False, "regexp": B("\xff\xff(\\x00|[\x01-\x0f])\x00.\x00\x43[\x20-\x7e]+")},

        # text families (UTF-16LE characters)
        "latin":    {"size": 20, "auto": False, "regexp": B("([\x41-\x5a\x61-\x7a]\x00)+")},
        "alpha":    {"size": 20, "auto": False, "regexp": B("([\x41-\x5a\x61-\x7a\x30-\x39]\x00)+")},
        "text":     {"size": 20, "auto": False, "regexp": B("([\x20-\x7e]\x00)+")},
        "text0":    {"size": 20, "auto": False, "regexp": B("([\x20-\x7e]\x00)*")},
        "permil":   {"size": 2,  "auto": False, "regexp": B("\x30\x20")},
    }

# combine_regexps (Python doesn’t have the Windows concat pitfalls—just join bytes)
def combine_regexps(regexps, collapse=b"", var="regexp"):
    parts = []
    for r in regexps:
        val = getattr(r, var, None)
        if val is None:
            raise ValueError("regex object missing 'regexp'")
        parts.append(val)
    return b"".join(parts)

# --- Minimal data parsing machinery to support your use here ---
_DATA_BLOCKS = {
    # size in bytes per element; 'regexp' not strictly needed here
    "float":  {"size": 4},
    "double": {"size": 8},
    "raw":    {"size": 1},
    "text":   {"size": 2},  # UTF-16LE char
}

def get_data_blocks_config():
    return _DATA_BLOCKS

def _remove_trailing_zeros(raw: bytes, unit_size: int) -> bytes:
    i = len(raw)
    while i > 0 and raw[i-1] == 0:
        i -= 1
    # round down to nearest multiple of unit_size
    i -= (i % unit_size)
    return raw[:i]

def parse_raw_data(
    raw: bytes,
    type,  # str OR list[str] (sequence per record)
    n: int | None = None,
    ignore_trailing_zeros: bool = False,
    exact_length: bool = False,
    sensible=None,
    errors: str | None = None,
):
    """
    Minimal but compatible:
      - 'text': decode UTF-16LE (whole slice) and return the string
      - 'raw' : return bytes as-is
      - ['float', 'double', ...]: treat as fixed-size record structure and parse repeated rows
    Returns:
      - for 'text': str
      - for single numeric type: list[float]
      - for mixed numeric types: list[tuple]
    """
    import struct

    # normalize type to list
    types = type if isinstance(type, (list, tuple)) else [type]
    # validate
    for t in types:
        if t not in _DATA_BLOCKS:
            raise RuntimeError(f"encountered invalid data type: {t}")

    # TEXT path (R forbids mixing 'text' with others)
    if "text" in types:
        if len(types) > 1:
            raise RuntimeError("data type 'text' cannot reliably be parsed jointly with other data types")
        # plain decode; if you want to enforce unicode-only, add validation
        return raw.decode("utf-16-le", errors="replace")

    # RAW passthrough
    if types == ["raw"]:
        return raw

    # numeric struct parsing
    sizes = [ _DATA_BLOCKS[t]["size"] for t in types ]
    rec_size = sum(sizes)

    buf = raw
    if ignore_trailing_zeros:
        buf = _remove_trailing_zeros(buf, rec_size)

    if rec_size == 0 or len(buf) < rec_size:
        if errors is not None:
            raise RuntimeError(f"{errors}raw data ({len(buf)} bytes) not long enough for single read of requested data type ({rec_size})")
        return None

    if n is None:
        n = len(buf) // rec_size

    if len(buf) < rec_size * n:
        if errors is not None:
            raise RuntimeError(f"{errors}raw data ({len(buf)} bytes) not long enough for requested byte read ({rec_size*n})")
        return None

    if exact_length and len(buf) != rec_size * n:
        if errors is not None:
            raise RuntimeError(f"{errors}raw data length ({len(buf)} bytes) does not match the requested byte read ({rec_size*n})")
        return None

    # Build struct format for one record (little-endian)
    def fmt_of(t: str) -> str:
        return {"float": "f", "double": "d"}[t]

    fmt = "<" + "".join(fmt_of(t) for t in types)
    step = struct.calcsize(fmt)
    assert step == rec_size

    out = []
    off = 0
    for _ in range(n):
        chunk = buf[off:off+step]
        if len(chunk) < step:
            break
        out.append(struct.unpack(fmt, chunk))
        off += step

    # Flatten if single numeric field
    if len(types) == 1:
        out = [v[0] for v in out]
    return out

# --- Minor stand-ins (safe no-ops) for package globals used in your R ---
def default(name, allow_null=False):
    # No global options; return harmless defaults.
    # Used by: default(debug) -> return False
    if name in ("debug", "quiet"):
        return False
    if not allow_null:
        return False
    return None

def log_message(*args, **kwargs):
    # Quiet by default; you can print if you want to see progress
    pass





# %%
bfile = read_binary_isodat_file('/Users/gerard/Documents/GitHub/chromatoPy/src/chromatopy/IRMS/24414__C18 200ng C24 200ng_20250501-0000.dxf')

# 1) Raw bytes (whole file)
raw_bytes = bfile.raw            # type: bytes
size = bfile.max_pos             # file size

# 2) Block table you can iterate
for blk in bfile.blocks[:5]:
    print(blk["block_idx"], blk["type"], blk["start"], blk["end"], blk["len"])

# 3) Slice any block’s raw payload directly
first = bfile.blocks[0]
payload = raw_bytes[first["start"] : first["end"] + 1]

# 4) Inspect structure as a JSON-safe dict (for logging/debug)
summary = to_dict(bfile)

# 1) Exact bytes of block 1 (C block), including its 6-byte header:
b1_all = get_block_bytes(bfile, 1)

# 2) Just the C-block payload (typically the interesting part):
cblk = next(blk for blk in bfile.blocks if blk["block_idx"] == 1)
b1_payload = get_cblock_payload_bytes(bfile, cblk)

# 3) Unknown block raw bytes (could be padding or something meaningful):
b2 = get_block_bytes(bfile, 2)

# 4) Text blocks decoded:
for blk, _ in iter_blocks_by_type(bfile, "text"):
    print(blk["block_idx"], decode_text_block(bfile, blk))
    
# %%
# Step 2: loop through all C blocks and print their payloads
for blk in bfile.C_blocks:
    payload = get_cblock_payload_bytes(bfile, blk)

    # Option 1: show printable ASCII text (often C blocks contain text)
    try:
        print(f"\nC-block {blk['block_idx']} (len={blk['data_len']} bytes):")
        print(payload.decode('latin-1', errors='replace'))
    except Exception:
        print(f"\nC-block {blk['block_idx']} raw bytes (len={blk['data_len']}):")
        print(payload[:80])  # first 80 bytes only for brevity
        
# %%
for blk in bfile.C_blocks:
    payload = get_cblock_payload_bytes(bfile, blk)
    name = payload.decode('latin-1', errors='replace')
    if "RawData" in name or "Data" in name:
        print(f"\nBlock {blk['block_idx']} ({blk['len']} bytes): {name}")
        
# %%
blk = next(b for b in bfile.C_blocks if b['block_idx'] == 3511)
raw_payload = get_cblock_payload_bytes(bfile, blk)

# print first few bytes in hex
print(raw_payload[:64].hex(' '))

# %%
# Helpers (add once)
def find_cblock_by_name(bfile, name_substr: str):
    """Return (blk, name_string) for the first C-block whose payload contains name_substr."""
    for blk in bfile.C_blocks:
        p = get_cblock_payload_bytes(bfile, blk)
        s = p.decode('latin-1', errors='replace')
        if name_substr in s:
            return blk, s
    return None, None

def slice_bytes(bfile, start: int, nbytes: int) -> bytes:
    end = min(start + nbytes, bfile.max_pos)
    return bfile.raw[start:end]

def print_block_neighborhood(bfile, center_idx: int, k: int = 3):
    """Show k blocks before/after a block index to inspect surroundings."""
    blocks = bfile.blocks
    pos = next(i for i, b in enumerate(blocks) if b["block_idx"] == center_idx)
    lo = max(0, pos - k); hi = min(len(blocks), pos + k + 1)
    for b in blocks[lo:hi]:
        print(f"{b['block_idx']:>6}  {b['type']:<12} start={b['start']:<8} end={b['end']:<8} len={b['len']:<6}")

# Use them
blk, name = find_cblock_by_name(bfile, "CRawData")
print("Found:", blk["block_idx"], name)

# 1) Inspect neighborhood to see what follows CRawData
print_block_neighborhood(bfile, blk["block_idx"], k=5)

# 2) Peek at the next non-C block’s bytes after CRawData (likely where payload starts)
#    (We take the very next block in the global sequence.)
all_blocks = bfile.blocks
pos = next(i for i, b in enumerate(all_blocks) if b["block_idx"] == blk["block_idx"])
if pos + 1 < len(all_blocks):
    next_blk = all_blocks[pos + 1]
    nxt_bytes = slice_bytes(bfile, next_blk["start"], min(256, next_blk["len"]))
    print(f"\nNext block after CRawData: #{next_blk['block_idx']} {next_blk['type']} len={next_blk['len']}")
    print("First bytes (hex):", nxt_bytes.hex(' '))
    
    
# %%
bfile = read_binary_isodat_file('/Users/gerard/Documents/GitHub/chromatoPy/src/chromatopy/IRMS/24414__C18 200ng C24 200ng_20250501-0000.dxf')

# Example: jump to the same high-level region your R uses
bfile = set_binary_file_error_prefix(bfile, "cannot identify measured masses")
bfile = move_to_C_block_range(bfile, "CEvalDataIntTransferPart", "CBinary")

# Find positions of a gas-config pattern the same way
gas_config_name_re = re_combine(re_text_x(), re_block("alpha"), re_text_0(), re_text_x())
config_positions = find_next_patterns(bfile, gas_config_name_re)

# %%
s = 0
for i in range(0,len(bfile.blocks)):
    if bfile.blocks[i]['len']>s:
        s=bfile.blocks[i]['len']
print(s)
# %%
print(bfile.pos)
print(bfile.max_pos)

# %% Pull raw voltages
# --- Minimal end-to-end extractor: ports your R logic without plotting ---

def _parse_numeric_block(raw: bytes, types, endian: str):
    fmt_map = {"float": "f", "double": "d"}
    fmt = endian + "".join(fmt_map[t] for t in types)
    step = struct.calcsize(fmt)
    n = len(raw) // step
    out = []
    off = 0
    for _ in range(n):
        chunk = raw[off:off+step]
        if len(chunk) < step:
            break
        out.append(struct.unpack(fmt, chunk))
        off += step
    return out

def _is_finite(x):
    return (x is not None) and (not isinstance(x, bool)) and math.isfinite(x)

def _plausibility_score(tuples, voltage_idx_start=1):
    if not tuples:
        return -1e9
    times = [r[0] for r in tuples]
    finite_time = [t for t in times if _is_finite(t)]
    sane_time = [t for t in finite_time if (t >= -1e-3 and t <= 1e7)]
    mono = sum(
        1 for i in range(len(times)-1)
        if _is_finite(times[i]) and _is_finite(times[i+1]) and times[i+1] >= times[i]-1e-9
    )
    volt_vals = [abs(v) for r in tuples for v in r[1:] if _is_finite(v)]
    frac_finite_t = len(finite_time) / max(1, len(times))
    frac_sane_t   = len(sane_time) / max(1, len(finite_time))
    frac_mono     = mono / max(1, len(times)-1)
    frac_finite_v = len(volt_vals) / max(1, len(tuples)*(len(tuples[0])-1))
    med_abs_v     = median(volt_vals) if volt_vals else float("inf")
    v_penalty     = -0.5 if not volt_vals else (-1.0 if med_abs_v > 1e9 else 0.0)
    score = 2*frac_finite_t + 2*frac_sane_t + 1.5*frac_mono + 1.0*frac_finite_v + v_penalty
    return score

def decode_numeric_block_with_endian(raw_block: bytes, types):
    tuples_le = _parse_numeric_block(raw_block, types, '<')
    tuples_be = _parse_numeric_block(raw_block, types, '>')
    s_le = _plausibility_score(tuples_le)
    s_be = _plausibility_score(tuples_be)
    if s_le >= s_be:
        return tuples_le, '<'
    else:
        return tuples_be, '>'

# ---------- main extraction function ----------

def extract_dxf_raw_voltage_data_py(bfile, export_path: str | None = None):
    """
    Extracts time–voltage pairs from Thermo Isodat .dxf binary files.
    Tags each record with the 'gas' configuration name,
    detects endian automatically, and returns a pandas DataFrame.

    Parameters
    ----------
    bfile : BinaryIsodatFile
        Parsed file object from read_binary_isodat_file().
    export_path : str, optional
        If provided, DataFrame is saved to CSV or Parquet (depending on extension).

    Returns
    -------
    df : pandas.DataFrame
        Columns: tp, time.s, gas, v<mass>.mV ...
    """
    set_binary_file_error_prefix(bfile, "cannot identify measured masses")
    move_to_C_block_range(bfile, "CEvalDataIntTransferPart", "CBinary")

    gas_config_name_re = re_combine(re_text_x(), re_block("alpha"), re_text_0(), re_text_x())
    config_positions = find_next_patterns(bfile, gas_config_name_re)
    if not config_positions:
        raise RuntimeError("No gas configurations found")

    configs = {}
    caps = config_positions[1:] + [bfile.max_pos]
    for i, pos in enumerate(config_positions):
        move_to_pos(bfile, pos)
        move_to_next_pattern(bfile, re_text_x(), max_gap=0)
        capture_data_till_pattern(bfile, "gas", "text", re_text_0(), re_text_x())
        gas_name = bfile.data.get("gas", "")
        cap_here = caps[i]
        if gas_name in configs:
            configs[gas_name]["cap"] = cap_here
        else:
            configs[gas_name] = {"pos": pos, "cap": cap_here, "masses": []}

    # --- find masses for each gas ---
    for cfg_name, cfg in configs.items():
        move_to_pos(bfile, cfg["pos"])
        cap_at_pos(bfile, cfg["cap"])
        intensity_id = 1
        while True:
            pat = re_unicode(f"rIntensity{intensity_id}")
            pos = find_next_pattern(bfile, pat)
            if pos is None:
                break
            move_to_next_pattern(bfile, pat)
            move_to_next_pattern(bfile, re_text_x(), re_unicode("rIntensity "), max_gap=0)
            capture_data_till_pattern(bfile, "mass", "text", re_text_x(), move_past_dots=True)
            mass = (bfile.data.get("mass") or "").strip()
            if mass:
                cfg["masses"].append(mass)
            intensity_id += 1

    # --- move to data region ---
    set_binary_file_error_prefix(bfile, "cannot recover raw voltages")
    move_to_C_block_range(bfile, "CAllMoleculeWeights", "CMethod")
    move_to_next_C_block(bfile, "CStringArray")
    move_to_next_pattern(bfile, re_unicode("OrigDataBlock"), re_null(4), re_block("stx"))

    data_start_re = re_combine(
        re_text_0(), re_null(4), re_x_000(), re_x_000(),
        re_direct("..", label="..", size=2), re_x_000()
    )
    data_end_re = re_combine(
        re_direct(".{4}", label=".{4}", size=4), re_null(4),
        re_text_0(), re_block("stx")
    )
    gas_config_re = re_combine(re_text_x(), re_block("text"), re_text_0())

    positions = find_next_patterns(bfile, data_start_re)
    if not positions:
        raise RuntimeError("could not find raw voltage data")

    all_rows, endian_flags = [], []
    for pos in positions:
        move_to_pos(bfile, pos + data_start_re.size + 4)
        start_pos = bfile.pos
        move_to_next_pattern(bfile, data_end_re)
        move_to_next_pattern(bfile, gas_config_re, move_to_end=False, max_gap=20)
        skip_pos(bfile, 4)
        capture_data_till_pattern(bfile, "gas", "text", re_text_0(), data_bytes_max=50)
        gas = bfile.data.get("gas", "")
        if gas not in configs:
            available = "', '".join(configs.keys())
            raise RuntimeError(
                f"could not find gas configuration for gas '{gas}', available: '{available}'"
            )
        masses = configs[gas]["masses"]
        if not masses:
            raise RuntimeError(f"could not identify measured ions for gas '{gas}'")

        read_types = ["float"] + ["double"] * len(masses)
        move_to_pos(bfile, start_pos)
        capture_data_till_pattern(bfile, "voltages_raw", "raw", data_end_re)
        raw_block = bfile.data.get("voltages_raw") or b""

        tuples, chosen_endian = decode_numeric_block_with_endian(raw_block, read_types)
        endian_flags.append(chosen_endian)

        mass_cols = [f"v{m}.mV" for m in masses]
        for r in tuples:
            time_s = r[0]
            row = {"time.s": time_s, "gas": gas}
            for name, val in zip(mass_cols, r[1:]):
                row[name] = val
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("no valid voltage tuples found")

    all_rows.sort(key=lambda d: d["time.s"])
    for i, r in enumerate(all_rows, start=1):
        r["tp"] = i

    df = pd.DataFrame(all_rows)
    # Ensure column order
    cols = ["tp", "time.s", "gas"] + [c for c in df.columns if c not in ("tp","time.s","gas")]
    df = df[cols]

    df.attrs["chosen_endian"] = endian_flags[0] if endian_flags else None

    if export_path:
        if export_path.lower().endswith(".csv"):
            df.to_csv(export_path, index=False)
        elif export_path.lower().endswith((".parquet", ".pq")):
            df.to_parquet(export_path, index=False)
        else:
            raise ValueError("Unsupported export format; use .csv or .parquet")

    return df
    
# %%
bfile = read_binary_isodat_file('/Users/gerard/Documents/GitHub/chromatoPy/src/chromatopy/IRMS/Isodat Files/Isodat Files/24345__SZM065 FAME C26-0000.dxf')
df = extract_dxf_raw_voltage_data_py(bfile, export_path="voltages.csv")

print(df.head())
print("Endian detected:", df.attrs["chosen_endian"])
print("Unique gases:", df['gas'].unique())

import matplotlib.pyplot as plt
fig = plt.figure()
# df = df[df["time.s"]>250]
# df = df[df["time.s"]<1500]
plt.plot(df['time.s']/60, df['v2.mV'], c='k')
plt.plot(df['time.s']/60, df['v3.mV'], c='red')
plt.savefig("/Users/gerard/Documents/GitHub/chromatoPy/src/chromatopy/IRMS/24345__SZM065 FAME C26-0000_RAW.png", dpi=100)
plt.show()