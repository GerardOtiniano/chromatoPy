import json
import os
import platform
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import sys

# ──────────────────────────────────────────────────────────────
#  Default configuration & helpers
# ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "Standard": {
        "checked": True,
        "traces": {"744": "Internal Standard"},
        "window": [10, 30],
    },
    "brGDGTs": {
        "checked": True,
        "traces": {
            "1050": "IIIa, IIIa'', IIIa'",
            "1048": "IIIb, IIIb'",
            "1046": "IIIc, IIIc'",
            "1036": "IIa, IIa'",
            "1034": "IIb, IIb'",
            "1032": "IIc, IIc'",
            "1022": "Ia",
            "1020": "Ib",
            "1018": "Ic",
        },
        "window": [20, 40],
    },
    "isoGDGTs": {
        "checked": True,
        "traces": {
            "1302": "GDGT-0",
            "1300": "GDGT-1",
            "1298": "GDGT-2",
            "1296": "GDGT-3",
            "1292": "GDGT-4, GDGT-4'",
        },
        "window": [5, 25],
    },
    "OH-GDGTs": {
        "checked": True,
        "traces": {
            "1300": "OH-GDGT-0",
            "1298": "OH-GDGT-1, 2OH-GDGT-0",
            "1296": "OH-GDGT-2",
        },
        "window": [35, 50],
    },
}

# GDGT_CONFIG_PATH = os.path.join(os.getcwd(), "src/chromatopy/config/chromatopy_gdgt_config.json")
APP_NAME = "chromatopy"

def get_resource_path(relative_path):
    """Resolve path inside bundled PyInstaller app or fallback to source."""
    try:
        base_path = sys._MEIPASS  # PyInstaller runtime dir
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def get_user_config_path():
    """Return writable path for user config based on OS."""
    if platform.system() == "Darwin":
        base = os.path.expanduser(f"~/Library/Application Support/{APP_NAME}")
    elif platform.system() == "Windows":
        base = os.path.join(os.getenv("APPDATA"), APP_NAME)
    else:  # Linux or other
        base = os.path.expanduser(f"~/.config/{APP_NAME}")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "chromatopy_gdgt_config.json")


DEFAULT_CONFIG_PATH = get_resource_path("src/chromatopy/config/chromatopy_gdgt_config.json")
USER_CONFIG_PATH = get_user_config_path()

# def load_gdgt_config():
#     if os.path.exists(GDGT_CONFIG_PATH):
#         try:
#             with open(GDGT_CONFIG_PATH) as f:
#                 return json.load(f)
#         except json.JSONDecodeError:
#             # corrupted → fall back to defaults
#             pass
#     save_gdgt_config(DEFAULT_CONFIG)
#     return json.loads(json.dumps(DEFAULT_CONFIG))  # deep-copy


# def save_gdgt_config(cfg):
#     with open(GDGT_CONFIG_PATH, "w") as f:
#         json.dump(cfg, f, indent=4)

def load_gdgt_config():
    # Use user config if available
    if os.path.exists(USER_CONFIG_PATH):
        try:
            with open(USER_CONFIG_PATH) as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass  # corrupted → fall back to default

    # Load default config from bundled asset
    try:
        with open(DEFAULT_CONFIG_PATH) as f:
            default = json.load(f)
    except Exception:
        default = DEFAULT_CONFIG  # fallback to hardcoded default

    save_gdgt_config(default)  # copy to user space
    return json.loads(json.dumps(default))  # return deep copy


def save_gdgt_config(cfg):
    with open(USER_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

def open_gdgt_selector(app: toga.App):
    """
    Launch a Toga window that lets users edit the GDGT groups/traces.
    Pass `self` from your main toga.App.
    """
    cfg = load_gdgt_config()

    win = toga.Window(title="Target GDGTs", size=(960, 900), resizable=True)

    # Root scroll box
    root = toga.Box(style=Pack(direction=COLUMN, padding=16))
    win.content = toga.ScrollContainer(horizontal=False, content=root)

    # bookkeeping
    name_vars, chk_vars = {}, {}
    trace_inputs, label_inputs = {}, {}

    # ── helpers ──────────────────────────────────────────────
    def remove_trace(g_key, t_key):
        cfg[g_key]["traces"].pop(t_key, None)
        redraw()

    def spacer(height=8):
        """Vertical blank space."""
        return toga.Box(style=Pack(height=height))

    # ── UI builder ───────────────────────────────────────────
    def redraw():
        root.children.clear()
        name_vars.clear(); chk_vars.clear()
        trace_inputs.clear(); label_inputs.clear()

        for g_key, g_data in cfg.items():
            # card wrapper
            card = toga.Box(style=Pack(direction=COLUMN, padding=12, padding_bottom=18))
            root.add(card)

            # header row (switch + name)
            hdr = toga.Box(style=Pack(direction=ROW, alignment="center", padding_bottom=6))
            card.add(hdr)

            chk = toga.Switch("Enable", style=Pack(width=90))
            chk.value = g_data.get("checked", True)
            chk_vars[g_key] = chk
            hdr.add(chk)

            name_in = toga.TextInput(value=g_key, style=Pack(flex=2, padding_right=6))
            name_vars[g_key] = name_in
            hdr.add(name_in)

            # RT window row
            win_row = toga.Box(style=Pack(direction=ROW, alignment="center", padding_bottom=6))
            card.add(win_row)

            win_row.add(toga.Label("RT min:", style=Pack(padding_left=4, padding_right=2)))
            min_in = toga.TextInput(value=str(g_data.get("window", [0, 0])[0]),
                                    style=Pack(width=80, padding_right=8))
            win_row.add(min_in)

            win_row.add(toga.Label("RT max:", style=Pack(padding_right=2)))
            max_in = toga.TextInput(value=str(g_data.get("window", [0, 0])[1]),
                                    style=Pack(width=80))
            win_row.add(max_in)

            g_data["__min_rt_in__"] = min_in
            g_data["__max_rt_in__"] = max_in

            # trace list
            trace_inputs[g_key], label_inputs[g_key] = {}, {}
            for t_key, lab in g_data["traces"].items():
                row = toga.Box(style=Pack(direction=ROW, alignment="center", padding_bottom=4))
                card.add(row)

                t_in = toga.TextInput(value=t_key, style=Pack(width=120, padding_right=6))
                l_in = toga.TextInput(value=lab, style=Pack(flex=1, padding_right=6))
                del_btn = toga.Button(
                    "✕",
                    on_press=lambda w, g=g_key, t=t_key: remove_trace(g, t),
                    style=Pack(width=32),
                )
                row.add(t_in); row.add(l_in); row.add(del_btn)

                trace_inputs[g_key][t_key] = t_in
                label_inputs[g_key][t_key] = l_in

            # add-trace button
            card.add(
                toga.Button(
                    "+  Add trace",
                    on_press=lambda w, g=g_key: (
                        cfg[g]["traces"].update(
                            {f"new_trace_{len(cfg[g]['traces'])+1}": "New Label"}
                        ),
                        redraw(),
                    ),
                    style=Pack(width=120, padding_top=4),
                )
            )

            # subtle divider
            div = toga.Box(style=Pack(height=1, background_color="silver"))
            root.add(div)

        # footer (once)
        footer = toga.Box(style=Pack(direction=ROW, alignment="center", padding_top=12))
        root.add(footer)

        footer.add(
            toga.Button(
                "+  Add GDGT type",
                on_press=lambda w: (
                    cfg.update({
                        f"NewGDGT_{len(cfg)+1}": {
                            "checked": True,
                            "traces": {"new_trace": "New Label"},
                            "window": [0, 0],
                        }
                    }),
                    redraw(),
                ),
                style=Pack(padding_right=6),
            )
        )

        footer.add(
            toga.Button(
                "Restore defaults",
                on_press=lambda w: (
                    save_gdgt_config(DEFAULT_CONFIG),
                    cfg.clear(),
                    cfg.update(json.loads(json.dumps(DEFAULT_CONFIG))),
                    redraw(),
                ),
                style=Pack(padding_right=6),
            )
        )

        # save button
        def on_save(widget):
            new_cfg = {}
            for old_key, name_in in name_vars.items():
                gname = name_in.value.strip() or old_key
                new_cfg[gname] = {
                    "checked": chk_vars[old_key].value,
                    "window": [
                        float(cfg[old_key]["__min_rt_in__"].value),
                        float(cfg[old_key]["__max_rt_in__"].value),
                    ],
                    "traces": {},
                }
                for orig_t, t_in in trace_inputs[old_key].items():
                    key = t_in.value.strip()
                    val = label_inputs[old_key][orig_t].value.strip()
                    if key and val:
                        new_cfg[gname]["traces"][key] = val

            save_gdgt_config(new_cfg)
            win.close()

        footer.add(toga.Button("Save", on_press=on_save))

    redraw()
    win.show()


# ──────────────────────────────────────────────────────────────
#  Data-loading helper (unchanged logic)
# ──────────────────────────────────────────────────────────────
def load_gdgt_window_data():
    """
    Return GDGT metadata dict:
      {"names": [...], "GDGT_dict": [...], "Trace":[...], "window":[...]}
    """
    config = load_gdgt_config()
    names, gdgt_dicts, traces, windows = [], [], [], []

    for gtype, group in config.items():
        if not group.get("checked", True):
            continue
        win = group.get("window")
        if not (isinstance(win, list) and len(win) == 2 and all(isinstance(v, (int, float)) for v in win)):
            raise ValueError(f"Invalid or missing window for GDGT group: {gtype}")

        group_dict, group_trace_ids = {}, []
        for trace_id, label_str in group.get("traces", {}).items():
            labels = [lbl.strip() for lbl in label_str.split(",")]
            group_dict[trace_id] = labels if len(labels) > 1 else labels[0]
            group_trace_ids.append(trace_id)

        names.append(gtype)
        gdgt_dicts.append(group_dict)
        traces.append(group_trace_ids)
        windows.append(win)

    return {"names": names, "GDGT_dict": gdgt_dicts, "Trace": traces, "window": windows}