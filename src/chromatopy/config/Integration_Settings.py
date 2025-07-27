"""
Integration-settings editor rewritten for **Toga**.
Everything that was pure-Tk has been replaced with Toga widgets;
logic, data schema, and JSON I/O are unchanged.
"""

import json
import os
import platform
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import sys

APP_NAME = "chromatopy"

def get_user_integration_path():
    """Return writable path for integration settings config."""
    if platform.system() == "Darwin":
        base = os.path.expanduser(f"~/Library/Application Support/{APP_NAME}")
    elif platform.system() == "Windows":
        base = os.path.join(os.getenv("APPDATA"), APP_NAME)
    else:
        base = os.path.expanduser(f"~/.config/{APP_NAME}")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "integration_settings.json")

def get_default_integration_path():
    try:
        base = sys._MEIPASS
    except AttributeError:
        base = os.path.abspath(".")
    return os.path.join(base, "src/chromatopy/config/integration_settings.json")

INTEGRATION_CONFIG_PATH = get_user_integration_path()

INTEGRATION_SCHEMA = [
    {
        "name": "peak_neighborhood_n",
        "default": 5,
        "type": "int",
        "description": "Max number of peaks to consider in the peak neighborhood (default 5).",
    },
    {
        "name": "smoothing_window",
        "default": 13,
        "type": "int",
        "description": "Window size for the Savitzky-Golay smoothing algorithm (default 13).",
    },
    {
        "name": "smoothing_factor",
        "default": 3,
        "type": "int",
        "description": "Order of polynomial for Savitzky-Golay smoothing algorithm (default 3).",
    },
    {
        "name": "gaus_iterations",
        "default": 1000,
        "type": "int",
        "description": "Number of Gaussian-fitting iterations (default 1000).",
    },
    {
        "name": "maximum_peak_amplitude",
        "default": "",
        "type": "float_or_none",
        "description": "Maximum peak amplitude for identifying peaks (leave blank for None).",
    },
    {
        "name": "peak_boundary_derivative_sensitivity",
        "default": 0.01,
        "type": "float",
        "description": "Sensitivity for boundary-derivative threshold (default 0.01).",
    },
    {
        # (kept exactly as in the source – note the missing "name" key is
        # an existing quirk in the original schema)
        "default": 0.001,
        "type": "float",
        "description": "Minimum prominence for a detected peak (default 0.001).",
    },
]


# ──────────────────────────────────────────────────────────────
#  JSON helpers
# ──────────────────────────────────────────────────────────────
def load_integration_settings():
    if os.path.exists(INTEGRATION_CONFIG_PATH):
        try:
            with open(INTEGRATION_CONFIG_PATH) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        # Optional: load bundled default if exists
        try:
            with open(get_default_integration_path()) as f:
                data = json.load(f)
        except Exception:
            data = {}

    # Fill in missing defaults
    for p in INTEGRATION_SCHEMA:
        if "name" in p and p["name"] not in data:
            data[p["name"]] = p["default"]
    return data


def save_integration_settings(settings_dict):
    with open(INTEGRATION_CONFIG_PATH, "w") as f:
        json.dump(settings_dict, f, indent=4)


# ──────────────────────────────────────────────────────────────
#  Toga window
# ──────────────────────────────────────────────────────────────
def open_settings(app):
    """
    Opens the Integration Settings editor.
    Pass in your running `toga.App` instance, e.g. `open_settings(self)` from your main app.
    """
    settings = load_integration_settings()

    # Scrollable root container
    scroll = toga.ScrollContainer(horizontal=False)
    root_box = toga.Box(style=Pack(direction=COLUMN, margin=10, background_color = "#F7ECE1"))
    scroll.content = root_box
    prev_window = app.main_window.content
    prev_title = app.main_window.title
    app.main_window.content = scroll
    app.main_window.title = "Integration Settings"

    entry_vars = {}  # param-name → (TextInput, type)

    # Build parameter rows
    for p in INTEGRATION_SCHEMA:
        # parameters missing a "name" key are skipped (matches original behaviour)
        if "name" not in p:
            continue

        param_box = toga.Box(style=Pack(direction=COLUMN, margin_top = 12, margin_left = 10))
        root_box.add(param_box)

        # Name (bold)
        param_box.add(
            toga.Label(
                p["name"],
                style=Pack(font_weight="bold", font_size=12, margin_bottom=2, margin_left = 10, color = "#0D1B1E")
            )
        )
        # Description (grey)
        param_box.add(
            toga.Label(
                p["description"],
                style=Pack(color = "#0D1B1E", font_size=12, margin_bottom=4, margin_left = 10),
            )
        )
        # Entry field
        txt = toga.TextInput(value=str(settings[p["name"]]), style=Pack(width=300, margin_left = 10, background_color="#3B4954", color = "#F7ECE1"))
        param_box.add(txt)
        entry_vars[p["name"]] = (txt, p["type"])

    # Save button
    def on_save(widget):
        new_settings = {}
        for name, (txt, typ) in entry_vars.items():
            value = txt.value.strip()
            # type-coerce with same rules as original Tk code
            if typ == "bool":
                new_settings[name] = value.lower() in ("1", "true", "yes", "y")
            elif typ == "int":
                try:
                    new_settings[name] = int(value)
                except ValueError:
                    new_settings[name] = settings[name]
            elif typ == "float":
                try:
                    new_settings[name] = float(value)
                except ValueError:
                    new_settings[name] = settings[name]
            elif typ == "float_or_none":
                if value.lower() in ("none", ""):
                    new_settings[name] = None
                else:
                    try:
                        new_settings[name] = float(value)
                    except ValueError:
                        new_settings[name] = settings[name]
            else:  # treat as plain string
                new_settings[name] = value

        save_integration_settings(new_settings)

    def go_back(widget):
        app.main_window.content = prev_window
        app.main_window.title = prev_title

    button_row = toga.Box(style=Pack(direction=ROW))
    back_path = "Icons/back.png"
    back_icon = toga.Icon(back_path)
    button_row.add(toga.Button(icon=back_icon, on_press=go_back,
                               style=Pack(margin_left=60, margin_right=170, height=40, width=60, margin_top=25)))

    button_row.add(toga.Button("Save", on_press=on_save,
                               style=Pack(margin_left=170, margin_right=60, height=40, width=60, margin_top=25,
                                          background_color="#3B4954",
                                          color="#F7ECE1",
                                          font_weight="bold", font_size=12)))
    root_box.add(button_row)