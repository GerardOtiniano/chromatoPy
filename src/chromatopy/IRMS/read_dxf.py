from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import pandas as pd


# ----------------------------
# Helper stubs / placeholders
# ----------------------------

def iso_is_continuous_flow(ds: Dict[str, Any]) -> bool:
    """
    R: iso_is_continuous_flow(ds)
    Check that ds is a 'continuous_flow' iso_file.
    """
    return ds.get("type") == "continuous_flow"


def get_ds_file_path(ds: Dict[str, Any]) -> str:
    """
    R: get_ds_file_path(ds)
    Should return path to the raw .dxf/.did file on disk.
    """
    return ds["file_path"]


def read_binary_isodat_file(path: str) -> "BinarySource":
    """
    R: read_binary_isodat_file()
    Should return a BinarySource-like object that allows scanning,
    regex-like searching in the binary, extracting blocks, etc.
    We'll define BinarySource below.
    """
    return BinarySource(path)


def exec_func_with_error_catch(func: Callable, ds: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    R: exec_func_with_error_catch(extract_xxx, ds, ...)
    Calls func(ds, **kwargs) and keeps ds even if func fails.
    """
    try:
        return func(ds, **kwargs)
    except Exception as e:
        # R code is "with_error_catch": it logs but doesn't kill read unless critical.
        # We'll mimic that: store error on ds and continue.
        ds.setdefault("_warnings", []).append(str(e))
        return ds


# ----------------------------
# BinarySource scaffolding
# ----------------------------

class BinarySource:
    """
    A Python stand-in for the R 'binary file' object that the code keeps mutating:
    - source$pos
    - source$max_pos
    - move_to_pos(), cap_at_pos(), etc.
    - capture_data_till_pattern(), find_next_patterns(), etc.

    We can't implement all of this without the full R helpers,
    so we'll just keep the interface and raise NotImplementedError.
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            self.bytes = f.read()

        self.pos = 0
        self.max_pos = len(self.bytes)
        self.data = {}           # place to stash captured info
        self._error_prefix = ""

    # --- mutators that mirror the R pipe chain ---

    def set_binary_file_error_prefix(self, prefix: str) -> "BinarySource":
        self._error_prefix = prefix
        return self

    def move_to_C_block_range(self, start_block: str, end_block: str) -> "BinarySource":
        # R: move_to_C_block_range("CEvalDataIntTransferPart","CBinary")
        raise NotImplementedError

    def move_to_pos(self, pos: int, reset_cap: bool = False) -> "BinarySource":
        self.pos = pos
        if reset_cap:
            self.max_pos = len(self.bytes)
        return self

    def cap_at_pos(self, cap: int) -> "BinarySource":
        self.max_pos = min(cap, len(self.bytes))
        return self

    def move_to_next_pattern(self, *patterns, move_to_end: bool = True, max_gap: Optional[int] = None) -> "BinarySource":
        # Should search forward from self.pos to find next occurrence of any combined regex/pattern sequence
        raise NotImplementedError

    def move_to_next_C_block(self, block_name: str) -> "BinarySource":
        raise NotImplementedError

    def capture_data_till_pattern(self, key: str, dtype: Any, *stop_patterns,
                                  move_past_dots: bool = False,
                                  data_bytes_max: Optional[int] = None) -> "BinarySource":
        """
        In R, this stores captured result under source$data[[key]].
        Here we'll set self.data[key] = <parsed_value>
        """
        raise NotImplementedError

    # --- finders ---

    def find_next_patterns(self, pattern) -> List[int]:
        raise NotImplementedError

    def find_next_pattern(self, pattern, *extra_patterns) -> Optional[int]:
        raise NotImplementedError

    # --- capping helpers used via lambdas in R ---

    def cap_at_next_C_block(self, block_name: str) -> "BinarySource":
        raise NotImplementedError

    def cap_at_pos_of_pattern(self, pattern) -> "BinarySource":
        raise NotImplementedError


# ----------------------------
# Regex builder stubs
# ----------------------------

def re_text_x():
    raise NotImplementedError

def re_block(name: str):
    raise NotImplementedError

def re_text_0():
    raise NotImplementedError

def re_unicode(txt: str):
    raise NotImplementedError

def re_combine(*parts):
    """
    In R: re_combine(...) builds a composite regex object with $size etc.
    We'll just return a dict so code that expects .get('size') won't crash.
    """
    return {"parts": parts, "size": None}

def re_null(n: int):
    raise NotImplementedError

def re_direct(pattern: str, size: Optional[int] = None, label: Optional[str] = None):
    raise NotImplementedError

def re_x_000():
    raise NotImplementedError


# ----------------------------
# Other extract_... helpers
# These are referenced in iso_read_dxf() but not defined here.
# We'll stub them so the translation is complete.
# ----------------------------

def extract_isodat_sequence_line_info(ds: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError

def extract_isodat_measurement_info(ds: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError

def extract_isodat_datetime(ds: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError

def extract_H3_factor_info(ds: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError

def extract_MS_integration_time_info(ds: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError

def extract_isodat_reference_values(ds: Dict[str, Any], cap_at_fun: Callable) -> Dict[str, Any]:
    raise NotImplementedError

def extract_isodat_resistors(ds: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError

def extract_isodat_continuous_flow_vendor_data_table(ds: Dict[str, Any], cap_at_fun: Callable) -> Dict[str, Any]:
    raise NotImplementedError

def cap_at_next_C_block(bin_src: BinarySource, block_name: str) -> BinarySource:
    return bin_src.cap_at_next_C_block(block_name)

def cap_at_pos(bin_src: BinarySource, pos: int) -> BinarySource:
    return bin_src.cap_at_pos(pos)

def find_next_pattern(bin_src: BinarySource, *patterns):
    return bin_src.find_next_pattern(*patterns)

def default(key: str):
    """
    R code uses default(debug) for a global option.
    We'll just always return False for 'debug' so we don't spam logs.
    """
    return False

def log_message(*args):
    print("".join(str(a) for a in args))


# ----------------------------
# Main translation of iso_read_dxf
# ----------------------------

def iso_read_dxf(ds: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Python translation of iso_read_dxf()
    """
    if options is None:
        options = {}

    # safety check
    if not iso_is_continuous_flow(ds):
        raise ValueError("data structure must be a 'continuous_flow' iso_file")

    # read binary file
    ds["source"] = read_binary_isodat_file(get_ds_file_path(ds))

    # process file info
    if ds.get("read_options", {}).get("file_info", False):
        ds = exec_func_with_error_catch(extract_isodat_sequence_line_info, ds)
        ds = exec_func_with_error_catch(extract_isodat_measurement_info, ds)
        ds = exec_func_with_error_catch(extract_isodat_datetime, ds)
        ds = exec_func_with_error_catch(extract_H3_factor_info, ds)
        ds = exec_func_with_error_catch(extract_MS_integration_time_info, ds)

    # process raw data
    # NOTE: R has ds$read_option$raw_data (singular option), which might be a typo.
    # I'll support both spellings: 'read_option' and 'read_options'.
    read_raw_flag = (
        ds.get("read_option", {}).get("raw_data", False)
        or ds.get("read_options", {}).get("raw_data", False)
    )
    if read_raw_flag:
        ds = exec_func_with_error_catch(extract_dxf_raw_voltage_data, ds)

    # process method info
    if ds.get("read_options", {}).get("method_info", False):
        ds = exec_func_with_error_catch(
            extract_isodat_reference_values,
            ds,
            cap_at_fun=lambda bin_src: cap_at_next_C_block(bin_src, "CResultArray"),
        )
        ds = exec_func_with_error_catch(extract_isodat_resistors, ds)

    # process pre-evaluated data table
    if ds.get("read_options", {}).get("vendor_data_table", False):
        ds = exec_func_with_error_catch(
            extract_isodat_continuous_flow_vendor_data_table,
            ds,
            cap_at_fun=lambda bin_src: cap_at_pos(
                bin_src,
                find_next_pattern(bin_src, re_unicode("DetectorDataBlock")),
            ),
        )

    return ds


# ----------------------------
# Translation of extract_dxf_raw_voltage_data
# ----------------------------

def extract_dxf_raw_voltage_data(ds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Python translation of extract_dxf_raw_voltage_data()
    This function mutates ds['source'] (BinarySource)
    and populates ds['raw_data'] as a pandas DataFrame.
    """

    src: BinarySource = ds["source"]

    # 1. Identify gas configurations
    # R:
    # ds$source <- ds$source |>
    #   set_binary_file_error_prefix("cannot identify measured masses") |>
    #   move_to_C_block_range("CEvalDataIntTransferPart", "CBinary")
    src = (
        src
        .set_binary_file_error_prefix("cannot identify measured masses")
        .move_to_C_block_range("CEvalDataIntTransferPart", "CBinary")
    )
    ds["source"] = src

    # regex for gas config name
    gas_config_name_re = re_combine(
        re_text_x(), re_block("alpha"), re_text_0(), re_text_x()
    )

    config_positions = src.find_next_patterns(gas_config_name_re)
    config_caps = config_positions[1:] + [src.max_pos] if config_positions else []

    configs: Dict[str, Dict[str, Any]] = {}

    if len(config_positions) == 0:
        # no configs → just return ds unchanged
        return ds

    # Parse each gas configuration block: find config name & track [pos, cap]
    for i, start_pos in enumerate(config_positions):
        src = (
            src
            .move_to_pos(start_pos)
            .move_to_next_pattern(re_text_x(), max_gap=0)
            .capture_data_till_pattern("gas", "text", re_text_0(), re_text_x())
        )
        gas_name = src.data.get("gas", "")

        if gas_name in configs:
            # already saw this config, so extend cap
            configs[gas_name]["cap"] = config_caps[i]
        else:
            # new config entry
            configs[gas_name] = {
                "pos": start_pos,
                "cap": config_caps[i],
                "masses": [],
            }

    if len(configs) == 0:
        raise ValueError("could not find gas configurations")

    # 2. For each config, find all measured masses (rIntensity1, rIntensity2, ...)
    for gas_conf_name, info in configs.items():
        if default("debug"):
            log_message(
                f"processing config '{gas_conf_name}' ({info['pos']}-{info['cap']})"
            )

        src = (
            src
            .move_to_pos(info["pos"])
            .cap_at_pos(info["cap"])
        )

        intensity_id = 1
        while True:
            # look for "rIntensity{intensity_id}"
            pat = re_unicode(f"rIntensity{intensity_id}")
            if src.find_next_pattern(pat) is None:
                break

            src = (
                src
                .move_to_next_pattern(pat)
                .move_to_next_pattern(re_text_x(), re_unicode("rIntensity "), max_gap=0)
                .capture_data_till_pattern("mass", "text", re_text_x(), move_past_dots=True)
            )
            mass_val = src.data.get("mass")
            configs[gas_conf_name]["masses"].append(mass_val)

            intensity_id += 1

    # 3. Find alternative gas config names (sometimes they rename configs)
    #    This is used to rename keys in `configs`
    src = (
        src
        .move_to_C_block_range("CPeakFindParameter", "CResultArray")
    )
    ds["source"] = src

    smoothing_positions = src.find_next_patterns(re_unicode("Smoothing"))

    gas_name_end_re = re_combine(re_null(4), re_direct("[\x01-\xff]", label="x01-xff"))
    gas_name_re = re_combine(re_text_x(), re_block("text0"), gas_name_end_re)

    for pos in smoothing_positions:
        # gas_name1
        src = src.move_to_pos(pos, reset_cap=True)
        src = (
            src
            .cap_at_pos(src.find_next_pattern(re_unicode("Peak Center")))
            .move_to_next_pattern(gas_name_re, move_to_end=False)
            .move_to_pos(src.pos + 4)  # skip 4 bytes analogous to skip_pos(4)
            .capture_data_till_pattern(
                "gas_name1", "text", gas_name_end_re, data_bytes_max=50
            )
        )
        gas_name1 = src.data.get("gas_name1", "")

        # gas_name2
        next_gas_name_pos = src.find_next_pattern(gas_name_re)
        if next_gas_name_pos is not None and next_gas_name_pos < src.max_pos:
            src = (
                src
                .move_to_next_pattern(gas_name_re, move_to_end=False)
                .move_to_pos(src.pos + 4)  # skip 4 bytes
                .capture_data_till_pattern(
                    "gas_name2", "text", gas_name_end_re, data_bytes_max=50
                )
            )
            gas_name2 = src.data.get("gas_name2", "")

            if (
                gas_name2 != ""
                and gas_name1 != gas_name2
                and gas_name1 in configs
            ):
                if default("debug"):
                    log_message(
                        f"renaming config '{gas_name1}' to '{gas_name2}' (non-standard config name)"
                    )
                # rename dictionary key
                configs[gas_name2] = configs.pop(gas_name1)

    # 4. Move to block that actually stores raw voltages
    src = (
        src
        .set_binary_file_error_prefix("cannot recover raw voltages")
        .move_to_C_block_range("CAllMoleculeWeights", "CMethod")
        .move_to_next_C_block("CStringArray")
        .move_to_next_pattern(
            re_unicode("OrigDataBlock"), re_null(4), re_block("stx")
        )
    )
    ds["source"] = src

    # regex / markers defining each dataset
    data_start_re = re_combine(
        re_text_0(), re_null(4), re_x_000(), re_x_000(),
        re_direct("..", size=2, label=".."), re_x_000()
    )
    data_end_re = re_combine(
        re_direct(".{4}", label=".{4}"), re_null(4),
        re_text_0(), re_block("stx")
    )
    gas_config_re = re_combine(re_text_x(), re_block("text"), re_text_0())

    voltages_frames: List[pd.DataFrame] = []
    positions = src.find_next_patterns(data_start_re)

    for pos in positions:
        # move to beginning of numeric data
        # R: ds$source |> move_to_pos(pos + data_start_re$size + 4L)
        offset = (data_start_re.get("size") or 0) + 4
        src = src.move_to_pos(pos + offset)
        start_pos = src.pos

        # identify gas config for this block
        capture = (
            src
            .move_to_next_pattern(data_end_re)
            .move_to_next_pattern(gas_config_re, move_to_end=False, max_gap=20)
            .move_to_pos(src.pos + 4)  # skip 4 bytes like skip_pos(4)
            .capture_data_till_pattern("gas", "text", re_text_0(), data_bytes_max=50)
        )
        gas_config = capture.data.get("gas", "")
        gas_data_block_end = {"gas": gas_config, "pos": capture.pos}

        if default("debug"):
            log_message(
                f"processing data for '{gas_config}' "
                f"({start_pos}-{gas_data_block_end['pos']})"
            )

        # find config definition
        if gas_config not in configs:
            available = "', '".join(configs.keys())
            raise ValueError(
                f"could not find gas configuration for gas '{gas_config}', "
                f"available: '{available}'"
            )

        masses = configs[gas_config].get("masses")
        if masses is None:
            raise ValueError(
                f"could not identify measured ions for gas '{gas_config}'"
            )

        mass_cols = [f"v{m}.mV" for m in masses]

        # capture voltage block
        src = src.capture_data_till_pattern(
            "voltages",
            ["float"] + ["double"] * len(masses),
            data_end_re,
        )

        # turn into dataframe
        block_df = pd.DataFrame(src.data.get("voltages", []))
        # rename columns -> time.s, v{mass}.mV ...
        block_df.columns = ["time.s"] + mass_cols

        voltages_frames.append(block_df)

    # combine all blocks
    if len(voltages_frames) == 0:
        raise ValueError("could not find raw voltage data")

    voltages_all = pd.concat(voltages_frames, ignore_index=True)

    # sort by time.s, add 'tp' (time point index starting at 1)
    voltages_all = voltages_all.sort_values("time.s").reset_index(drop=True)
    voltages_all["tp"] = range(1, len(voltages_all) + 1)

    # final column order: tp, time.s, everything else
    cols = ["tp", "time.s"] + [c for c in voltages_all.columns if c not in ("tp", "time.s")]
    ds["raw_data"] = voltages_all.loc[:, cols]

    # keep updated source
    ds["source"] = src
    return ds