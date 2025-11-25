# dxf_reader_isodat_like.py
from __future__ import annotations
from pathlib import Path
import re, struct
import numpy as np
import pandas as pd

UTF16 = "utf-16-le"
DATA_START_RE = re.compile(
    b"\xff\xfe\xff\x00"      # {text-0}
    b"\x00\x00\x00\x00"      # 4x00
    b"[\x01-\x1f]\x00\x00\x00"  # <x-000>
    b"[\x01-\x1f]\x00\x00\x00"  # <x-000>
    b".."
    b"[\x01-\x1f]\x00\x00\x00"  # <x-000>
    , flags=0
)

DATA_END_RE = re.compile(
    b".{4}"                  # any 4
    b"\x00\x00\x00\x00"      # 4x00
    b"\xff\xfe\xff\x00"      # {text-0}
    b"\x02\x00\x00\x00",     # <stx> = 0x02 00 00 00
    flags=0
)


# ----------------- basic helpers -----------------

def read_bytes(path: str|Path) -> bytes:
    return Path(path).read_bytes()

def find_utf16_positions(blob: bytes, text: str) -> list[int]:
    pat = text.encode(UTF16)
    out, i = [], 0
    while True:
        j = blob.find(pat, i)
        if j == -1: break
        out.append(j); i = j + 2
    return out

def scan_utf16_labels(blob: bytes, min_chars: int = 3) -> list[tuple[int,str]]:
    """
    Cheap 'strings' for UTF-16LE labels: looks for printable ASCII with NULs.
    Returns sorted list of (offset, decoded_string).
    """
    pat = re.compile(rb'(?:[\x20-\x7E]\x00){%d,}' % min_chars)
    labels = []
    for m in pat.finditer(blob):
        try:
            s = m.group(0).decode(UTF16).strip("\x00").strip()
        except UnicodeDecodeError:
            continue
        if s:
            labels.append((m.start(), s))
    labels.sort()
    return labels

def next_label_after(labels: list[tuple[int,str]], start: int, default_end: int) -> int:
    for pos, _ in labels:
        if pos > start:
            return pos
    return default_end

# ----------------- “C-block” ranges (approx) -----------------

def find_range_between_labels(labels, begin_key: str, end_key: str, file_len: int) -> tuple[int,int] | None:
    """Return (start,end) by first begin_key then the next end_key label after it."""
    begins = [pos for pos, s in labels if s == begin_key]
    if not begins:
        return None
    start = begins[0]
    # end = first end_key after start, else file end
    for pos, s in labels:
        if pos > start and s == end_key:
            return (start, pos)
    return (start, file_len)

# ----------------- gas discovery (like isoreader) -----------------

def infer_config_masses(blob: bytes, rng: tuple[int,int]) -> dict[str, list[str]]:
    """
    Inside CEvalDataIntTransferPart..CBinary range, find configs and which rIntensityN they have.
    The R code extracts a 'mass text' for each rIntensityN; in many files it’s
    just channels 1..N. We’ll collect N per config and keep simple names first,
    possibly renamed later in the PeakFind range.
    """
    start, end = rng
    win = blob[start:end]

    # A heuristic: configs are introduced by small UTF16 sequences, then contain
    # UTF16 'rIntensity' markers. We’ll split by “alpha” text blocks or just
    # group by nearby 'Smoothing' / 'Peak Center' boundaries later.
    # Simpler: gather segments beginning at each UTF16 block that contains any rIntensity,
    # and give them a provisional config name from the nearest preceding UTF16 short string.
    # (Good enough to group channels; later we may rename from PeakFind block.)
    labels = scan_utf16_labels(win, min_chars=2)
    # offsets are relative to 'start'; adjust to absolute
    labels = [(start+pos, txt) for (pos, txt) in labels]

    rint_positions = []
    for pos, txt in labels:
        if txt.startswith("rIntensity"):
            # rIntensity or rIntensity N
            m = re.match(r"rIntensity ?(\d+)?", txt)
            ch = None
            if m and m.group(1):
                ch = int(m.group(1))
            rint_positions.append((pos, ch))

    if not rint_positions:
        return {}

    # group nearby rIntensity runs into provisional "configs" by big gaps
    rint_positions.sort()
    groups = []
    cur = [rint_positions[0]]
    for a, b in zip(rint_positions, rint_positions[1:]):
        if b[0] - a[0] < 20000:  # heuristic proximity threshold
            cur.append(b)
        else:
            groups.append(cur); cur = [b]
    groups.append(cur)

    configs = {}
    for gi, grp in enumerate(groups, 1):
        # find a label just before the first rIntensity in this group that looks like a short name
        anchor = grp[0][0]
        cand_name = None
        # look back up to ~3 labels
        back = [t for (p,t) in labels if p < anchor][-3:]
        for t in reversed(back):
            t = t.strip().strip(".")
            if t and len(t) <= 40 and not t.startswith("rIntensity"):
                cand_name = t; break
        name = cand_name or f"Config{gi}"
        # channels present:
        chs = [c for _, c in grp if c is not None]
        if not chs:
            # fallback: count distinct 'rIntensity ' occurrences
            chs = list(range(1, 1 + sum(1 for _, t in grp)))
        n = max(chs) if chs else len(grp)
        configs[name] = [str(i) for i in range(1, n+1)]

    return configs

def refine_config_names_from_peakfind(blob: bytes, labels, configs: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Look in CPeakFindParameter..CResultArray for alternative gas names (as in the R code).
    If we find pairs of names around 'Smoothing', rename matching configs.
    """
    file_len = len(blob)
    rng = find_range_between_labels(labels, "CPeakFindParameter", "CResultArray", file_len)
    if not rng:
        return configs

    start, end = rng
    win = blob[start:end]

    # Find 'Smoothing' hits and read the next couple of UTF-16 text runs as possible names
    sm_pos = find_utf16_positions(win, "Smoothing")
    out = dict(configs)
    for sp in sm_pos:
        local = win[sp: sp+1000]  # small window
        local_labels = scan_utf16_labels(local, min_chars=2)
        # get the first two strings after Smoothing
        cand = [s for _, s in local_labels if s and s not in ("Smoothing",)]
        if len(cand) >= 2:
            n1, n2 = cand[0].strip(), cand[1].strip()
            # if n1 matches any existing config key, rename to n2
            for k in list(out.keys()):
                if k == n1 and n2 and n2 != n1:
                    out[n2] = out.pop(k)
                    break
    return out

# ----------------- OrigDataBlock decoding -----------------

import struct
import numpy as np
import pandas as pd

def find_isodat_payload_spans(region: bytes) -> list[tuple[int,int]]:
    """
    Within an OrigDataBlock region (bytes immediately after the UTF-16 'OrigDataBlock' label),
    locate inner [start,end) slices that should contain the numeric array:
      - start at DATA_START_RE end
      - end right before DATA_END_RE start
    Returns list of (a,b) offsets relative to 'region'.
    """
    spans = []
    i = 0
    n = len(region)
    while i < n:
        m_start = DATA_START_RE.search(region, i)
        if not m_start:
            break
        a = m_start.end()
        m_end = DATA_END_RE.search(region, a)
        if not m_end:
            # no terminator; stop scanning this region
            break
        b = m_end.start()
        if b > a:
            spans.append((a, b))
        # continue scanning after this end
        i = m_end.end()
    return spans

def try_decode_records(buf: bytes, n_ch: int, header_skip: int, time_fmt: str, val_fmt: str):
    """
    Decode records like: [time (time_fmt)] + n_ch * [value (val_fmt)]
    Returns DataFrame with columns: time_s, v1..vN or None on failure.
    """

    # quick format sanity
    if not time_fmt or not val_fmt:
        return None
    # enforce little-endian (Isodat is LE)
    def _normalize(fmt: str) -> str:
        # If already explicit little/big endian, keep it.
        if fmt and fmt[0] in "<>":
            return fmt
        # If native/standard or missing, force little-endian.
        return "<" + fmt.lstrip("<>@=!")  # strip any endianness chars before prefixing
    
    time_fmt = _normalize(time_fmt)
    val_fmt  = _normalize(val_fmt)

    # compute sizes safely
    try:
        time_size = struct.calcsize(time_fmt)
        val_size  = struct.calcsize(val_fmt)
    except struct.error:
        return None  # invalid format combo

    if header_skip >= len(buf):
        return None
    data = memoryview(buf)[header_skip:]

    rec_len = time_size + n_ch * val_size
    if rec_len <= 0 or len(data) < rec_len:
        return None
    n_rec = len(data) // rec_len
    if n_rec <= 1:  # too few points to be a trace
        return None

    # fast unpack using numpy frombuffer when possible
    try:
        # reshape as [n_rec, 1 + n_ch] of raw bytes, then split
        # First, view as uint8 and slice; but easier: unpack loop with struct is fine for robustness
        off = 0
        times = np.empty(n_rec, dtype=np.float64)
        vals  = np.empty((n_rec, n_ch), dtype=np.float64)

        unpack_time = struct.Struct(time_fmt).unpack_from
        unpack_val  = struct.Struct(val_fmt).unpack_from

        for i in range(n_rec):
            t = unpack_time(data, off)[0]
            off += time_size
            # read n_ch values
            for j in range(n_ch):
                v = unpack_val(data, off)[0]
                off += val_size
                vals[i, j] = v
            times[i] = t

        # sanity checks
        if not np.isfinite(times).all():
            return None
        # time should be (mostly) increasing
        diffs = np.diff(times)
        if (np.sum(diffs > 0) / max(1, len(diffs))) < 0.7:
            return None

        cols = {"time_s": times}
        for k in range(n_ch):
            cols[f"v{k+1}"] = vals[:, k]
        return pd.DataFrame(cols)
    except Exception:
        return None

def gen_divisible_layouts(region_len: int, time_fmt: str, val_fmt: str,
                          n_candidates=(2,3,4,5,6,7,8), max_hdr=512, min_rows=100):
    """Yield (n_ch, hdr, rows) where (region_len-hdr) % (size(time)+n*size(val))==0."""
    st = FMT_SIZE[time_fmt]; sv = FMT_SIZE[val_fmt]
    out = []
    for n in n_candidates:
        rec = st + n*sv
        for hdr in range(0, min(max_hdr, region_len)):
            rem = region_len - hdr
            if rem > 0 and rem % rec == 0:
                rows = rem // rec
                if rows >= min_rows:
                    out.append((n, hdr, rows, rec))
    # heuristic: prefer (n=3, hdr=0), then more rows
    out.sort(key=lambda t: (t[0] != 3 or t[1] != 0, -t[2]))
    return out

def extract_all_raw_datasets(path: str, verbose: bool = True):
    """
    Locate and decode all OrigDataBlock payloads in a Thermo Isodat .dxf file.

    Returns
    -------
    datasets : dict[str, pandas.DataFrame]
        Mapping from a dataset label (e.g., 'block1_payload1') to a decoded DataFrame
        with columns: ['time_s', 'v1', 'v2', ...].
    all_df : pandas.DataFrame
        Concatenation of all decoded datasets with an extra leading
        'dataset' column indicating the source dataset.
    """
    import re
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # ---------------------------------------------------------------
    # Isodat payload boundary regexes (equivalent to R's re_combine)
    # ---------------------------------------------------------------
    DATA_START_RE = re.compile(
        b"\xff\xfe\xff\x00"      # {text-0}
        b"\x00\x00\x00\x00"      # 4x00
        b"[\x01-\x1f]\x00\x00\x00"
        b"[\x01-\x1f]\x00\x00\x00"
        b".."
        b"[\x01-\x1f]\x00\x00\x00",
        flags=0
    )

    DATA_END_RE = re.compile(
        b".{4}"
        b"\x00\x00\x00\x00"
        b"\xff\xfe\xff\x00"
        b"\x02\x00\x00\x00",
        flags=0
    )

    def find_isodat_payload_spans(region: bytes) -> list[tuple[int, int]]:
        """Locate inner [start,end) slices between data_start_re and data_end_re."""
        spans = []
        i = 0
        n = len(region)
        while i < n:
            m_start = DATA_START_RE.search(region, i)
            if not m_start:
                break
            a = m_start.end()
            m_end = DATA_END_RE.search(region, a)
            if not m_end:
                break
            b = m_end.start()
            if b > a:
                spans.append((a, b))
            i = m_end.end()
        return spans

    # -----------------------------------------------------------------
    # Helpers for brute-force decoding within candidate payloads
    # -----------------------------------------------------------------
    FMT_SIZE = {"<f": 4, ">f": 4, "<d": 8, ">d": 8, "<I": 4, ">I": 4}

    def same_endian(fmt_t: str, fmt_v: str) -> bool:
        return fmt_t[0] == fmt_v[0]

    def gen_divisible_layouts(region_len: int, time_fmt: str, val_fmt: str,
                              n_candidates=(2, 3, 4, 5, 6, 7, 8, 9, 10),
                              max_hdr: int = 4096, min_rows: int = 20):
        """Yield (n_ch, hdr, rows, rec_size) combos that fit evenly."""
        st = FMT_SIZE.get(time_fmt, 0)
        sv = FMT_SIZE.get(val_fmt, 0)
        out = []
        for n in n_candidates:
            rec = st + n * sv
            if rec <= 0:
                continue
            for hdr in range(0, min(max_hdr, region_len)):
                rem = region_len - hdr
                if rem > 0 and rem % rec == 0:
                    rows = rem // rec
                    if rows >= min_rows:
                        out.append((n, hdr, rows, rec))
        out.sort(key=lambda t: (t[0] != 3 or t[1] != 0, -t[2]))
        return out

    def try_decode_records_strict_formats(buf: bytes, n_ch: int, header_skip: int,
                                          time_fmt: str, val_fmt: str):
        """Try decoding; ensure sane time progression and finite voltages."""
        df = try_decode_records(buf, n_ch, header_skip, time_fmt, val_fmt)
        if df is None or df.empty or "time_s" not in df.columns:
            return None
        t = df["time_s"].to_numpy()
        if not np.isfinite(t).all():
            return None
        if len(t) > 1 and (np.diff(t) < 0).mean() > 0.15:
            return None
        V = df.iloc[:, 1:].to_numpy()
        if not np.isfinite(V).all() or np.nanstd(V) < 1e-12:
            return None
        return df

    def read_dxf_bytes(p: str) -> bytes:
        with open(p, "rb") as f:
            return f.read()

    def find_all_offsets(hay: bytes, needle: bytes):
        out = []
        i = 0
        n = len(needle)
        while True:
            j = hay.find(needle, i)
            if j < 0:
                break
            out.append(j)
            i = j + n
        return out

    def u16(s: str) -> bytes:
        return s.encode("utf-16le")

    # -----------------------------------------------------------------
    # Step 1: load and locate "OrigDataBlock"
    # -----------------------------------------------------------------
    blob = read_dxf_bytes(path)
    if verbose:
        print(f"Reading: {Path(path).name}  ({len(blob):,} bytes)")

    marker = u16("OrigDataBlock")
    starts = find_all_offsets(blob, marker)
    if verbose:
        print(f"Found {len(starts)} OrigDataBlock marker(s).")

    if not starts:
        raise RuntimeError("No OrigDataBlock markers found in file.")

    regions = []
    for i, s in enumerate(starts):
        data_start = s + len(marker)
        data_end = starts[i + 1] if i + 1 < len(starts) else len(blob)
        if data_end > data_start:
            regions.append((data_start, data_end))
        else:
            regions.append((data_start, data_start))

    # -----------------------------------------------------------------
    # Step 2: brute-force decode within each region
    # -----------------------------------------------------------------
    datasets = {}
    all_rows = []

    TIME_FMTS = ["<f", "<d", "<I", ">f", ">d", ">I"]
    VAL_FMTS = ["<d", "<f", ">d", ">f"]

    for i, (a, b) in enumerate(regions, start=1):
        region = blob[a:b]
        L = len(region)
        if verbose:
            print(f"[OrigDataBlock {i}] bytes[{a}:{b}] len={L:,}")

        if L <= 0:
            if verbose:
                print("  ✖ empty region; skipping.")
            continue

        spans = find_isodat_payload_spans(region)
        if verbose:
            print(f"  Found {len(spans)} inner payload span(s) by Isodat regex.")

        if not spans:
            spans = [(0, L)]

        region_had_success = False

        for payload_idx, (pa, pb) in enumerate(spans, start=1):
            sub = region[pa:pb]
            if verbose:
                print(f"   • payload {payload_idx}: rel[{pa}:{pb}] len={pb-pa:,}")

            candidates = []
            for tf in TIME_FMTS:
                for vf in VAL_FMTS:
                    if not same_endian(tf, vf):
                        continue
                    layouts = gen_divisible_layouts(len(sub), tf, vf)
                    for (n, hdr, rows, recsz) in layouts:
                        candidates.append((tf, vf, n, hdr, rows))

            seen = set()
            filtered = []
            for tf, vf, n, hdr, rows in candidates:
                key = (tf, vf, n, hdr)
                if key in seen:
                    continue
                seen.add(key)
                filtered.append((tf, vf, n, hdr, rows))

            payload_decoded = False
            for tf, vf, n, hdr, rows in filtered:
                df = try_decode_records_strict_formats(sub, n, hdr, tf, vf)
                if df is not None:
                    payload_decoded = True
                    region_had_success = True
                    if verbose:
                        t = df["time_s"].to_numpy()
                        tmin = float(np.nanmin(t)) if len(t) else np.nan
                        tmax = float(np.nanmax(t)) if len(t) else np.nan
                        print(f"  ✔ decoded: payload={payload_idx}, "
                              f"N={n}, hdr={hdr}, time_fmt={tf}, val_fmt={vf}, "
                              f"rows={len(df)}, time[{tmin:.6g}..{tmax:.6g}]")

                    label = f"block{i}_payload{payload_idx}"
                    datasets[label] = df
                    tmp = df.copy()
                    tmp.insert(0, "dataset", label)
                    all_rows.append(tmp)
                    break

            if not payload_decoded and verbose:
                print(f"  ✖ this payload {payload_idx} could not be decoded.")

        if not region_had_success and verbose:
            print("  ✖ could not decode this OrigDataBlock.")

    if not datasets:
        raise RuntimeError(
            "No OrigDataBlock could be decoded. "
            "Try increasing the header sweep (max_hdr), widening min_rows, "
            "or enabling integer ticks (<I/>I) if not already."
        )

    all_df = pd.concat(all_rows, axis=0, ignore_index=True)
    return datasets, all_df

def find_divisible_layout(region: bytes, n_candidates=(2,3,4,5,6,7,8), max_hdr=256):
    """
    Return a list of (n_ch, header_skip) where (len(region)-header_skip) is divisible
    by (4 + 8*n_ch) and would yield >= ~100 rows.
    """
    L = len(region)
    out = []
    for n_ch in n_candidates:
        rec = 4 + 8*n_ch
        for hdr in range(0, min(max_hdr, L), 1):
            rem = L - hdr
            if rem > 0 and rem % rec == 0:
                rows = rem // rec
                if rows >= 100:  # require a reasonable length
                    out.append((n_ch, hdr, rows))
    # try “n=3, hdr=0” first if present, then sort by rows desc as a good heuristic
    out.sort(key=lambda t: (t[0] != 3 or t[1] != 0, -t[2]))
    return out

def try_decode_records_strict(buf: bytes, n_ch: int, header_skip: int,
                              time_fmt: str = "<f", val_fmt: str = "<d"):
    """Decode and validate with relaxed-but-sane checks."""
    df = try_decode_records(buf, n_ch, header_skip, time_fmt, val_fmt)
    if df is None:
        return None
    t = df["time_s"].values
    if not np.isfinite(t).all():
        return None
    # allow up to 10% non-monotonic steps
    if (np.diff(t) < 0).mean() > 0.10:
        return None
    # plausible time range (0 .. 1e7 s ~ 115 days)
    if t.min() < -1e-3 or t.max() > 1e7:
        return None
    # require finite voltages and non-crazy spread
    V = df.iloc[:, 1:].values
    if not np.isfinite(V).all():
        return None
    # reject if all zeros or variance ~ 0
    if np.nanmax(np.nanstd(V, axis=0)) < 1e-15:
        return None
    return df

# %%

FILE = '/Users/gerard/Documents/GitHub/chromatoPy/src/chromatopy/IRMS/24414__C18 200ng C24 200ng_20250501-0000.dxf'

ds, all_df = extract_all_raw_datasets(
    FILE,
    verbose=True)
print(all_df.head())