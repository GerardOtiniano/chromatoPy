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

def get_user_plot_path():
    """Return writable path for integration settings config."""
    if platform.system() == "Darwin":
        base = os.path.expanduser(f"~/Library/Application Support/{APP_NAME}")
    elif platform.system() == "Windows":
        base = os.path.join(os.getenv("APPDATA"), APP_NAME)
    else:
        base = os.path.expanduser(f"~/.config/{APP_NAME}")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "plot_settings.json")

def get_default_plot_path():
    try:
        base = sys._MEIPASS
    except AttributeError:
        base = os.path.abspath(".")
    return os.path.join(base, "src/chromatopy/config/plot_settings.json")

PLOT_CONFIG_PATH = get_user_plot_path()

PLOT_SCHEMA = [
    {
        "name": "compounds",
        "default": "",
        "type": "str_or_none",
        "description": "List of compounds separated by commas.",
    },
    {
        "name": "time_header",
        "default": "RT (min)",
        "type": "str",
        "description": "Header for the data column storing retention-time values.",
    },
    {
        "name": "signal_header",
        "default": "Signal",
        "type": "str",
        "description": "Header for the data column storing signal values.",
    },
    {
        "name": "min_window",
        "default": 10.5,
        "type": "float",
        "description": "Upper window bound for x-axis.",
    },
    {
        "name": "max_window",
        "default": 20,
        "type": "float",
        "description": "Lower window bound for x-axis.",
    }
]


# ──────────────────────────────────────────────────────────────
#  JSON helpers
# ──────────────────────────────────────────────────────────────
def load_plot_settings():
    if os.path.exists(PLOT_CONFIG_PATH):
        try:
            with open(PLOT_CONFIG_PATH) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        # Optional: load bundled default if exists
        try:
            with open(get_default_plot_path()) as f:
                data = json.load(f)
        except Exception:
            data = {}

    # Fill in missing defaults
    for p in PLOT_SCHEMA:
        if "name" in p and p["name"] not in data:
            data[p["name"]] = p["default"]

    return data


def save_plot_settings(settings_dict):
    with open(PLOT_CONFIG_PATH, "w") as f:
        json.dump(settings_dict, f, indent=4)


# ──────────────────────────────────────────────────────────────
#  Toga window
# ──────────────────────────────────────────────────────────────
def open_plot_settings(app):
    """
    Opens the Integration Settings editor.
    Pass in your running `toga.App` instance, e.g. `open_settings(self)` from your main app.
    """
    settings = load_plot_settings()

    # Scrollable root container
    scroll = toga.ScrollContainer(horizontal=False)
    root_box = toga.Box(style=Pack(direction=COLUMN, padding=10,background_color = "#F7ECE1"))
    scroll.content = root_box
    prev_window = app.main_window.content
    prev_title = app.main_window.title
    app.main_window.content = scroll
    app.main_window.title = "Plot Settings"

    entry_vars = {}  # param-name → (TextInput, type)

    # Build parameter rows
    for p in PLOT_SCHEMA:
        param_box = toga.Box(style=Pack(direction=COLUMN, padding_top = 12, padding_left = 10))
        root_box.add(param_box)

        # Name (bold)
        param_box.add(
            toga.Label(
                p["name"],
                style=Pack(font_weight="bold", font_size=12, padding = 2,padding_left = 10, color="#0D1B1E"),
            )
        )
        # Description (grey)
        param_box.add(
            toga.Label(
                p["description"],
                style=Pack(color = "#0D1B1E", font_size=12, padding_bottom=4, padding_left = 10)
            )
        )
        # Entry field
        if p["name"] == "compounds":
            txt = toga.TextInput(placeholder= "Enter list of compounds.",
                                 style=Pack(width=300, padding_left=10, background_color="#3B4954", color="#F7ECE1"))
        else:
            txt = toga.TextInput(value=str(settings[p["name"]]),
                                 style=Pack(width=300, padding_left=10, background_color="#3B4954", color="#F7ECE1"))

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
            elif typ == "str_or_none":
                if value.lower() in ("none", ""):
                    new_settings[name] = None
                else:
                    new_settings[name] = value
            else:  # treat as plain string
                new_settings[name] = value

        save_plot_settings(new_settings)

    def go_back(widget):
        app.main_window.content = prev_window
        app.main_window.title = prev_title

    button_row = toga.Box(style=Pack(direction=ROW))
    back_path = "Icons/back.png"
    back_icon = toga.Icon(back_path)
    button_row.add(toga.Button(icon=back_icon, on_press=go_back,
                               style=Pack(margin_left = 60, margin_right = 170, height=40, width=60, padding_top=25)))

    button_row.add(toga.Button("Save", on_press=on_save,
                               style=Pack( margin_left = 170, margin_right = 60, height=40, width=60, padding_top=25, background_color="#3B4954",
                                          color="#F7ECE1",
                                          font_weight="bold", font_size=12)))
    root_box.add(button_row)
