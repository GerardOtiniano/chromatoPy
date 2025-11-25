# src/chromatopy/utils/folder_handling.py

def get_gdgt(gdgt_oi):
    """
    Retrieves metadata for GDGT (Glycerol Dialkyl Glycerol Tetraethers) types based on user selection.
    
    This function constructs and returns a structure containing GDGT group metadata, including compound names, trace IDs, and retention time windows. The user selects which GDGT groups they want to include, and the corresponding data structures are aggregated.
    
    Parameters
    ----------
    gdgt_oi : str
        A string indicating the user's selection of GDGT groups. 
        - "1" for isoGDGTs.
        - "2" for brGDGTs.
        - "3" for OH-GDGTs.
        - "4" or "1,2,3" for all GDGT types (isoGDGTs, brGDGTs, OH-GDGTs).
        Multiple selections can be provided as a comma-separated string (e.g., "1,2").
    
    Returns
    -------
    dict
        A dictionary containing metadata for the selected GDGT groups, with the following keys:
        - "names" (list): A list of names for each GDGT group.
        - "GDGT_dict" (list): A list of dictionaries, where each dictionary maps trace IDs to compound names.
        - "Trace" (list): A list of trace IDs for each GDGT group.
        - "window" (list): A list of retention time windows (in minutes) for each GDGT group.
    
    Notes
    -----
    - The default reference structure is always included, containing the reference trace (744).
    - When all GDGT types are selected (gdgt_oi == "4"), isoGDGTs, brGDGTs, and OH-GDGTs are included.
    - The function builds the final structure by appending metadata for the selected GDGT groups from predefined data structures.
    
    Example
    -------
    If the user selects "1,2", the function will return a combined structure containing metadata for isoGDGTs and brGDGTs, along with the reference trace.
    
    """
    # Define the data structures
    ref_struct = {"name": ["Reference"], "GDGT_dict": {"744": "Standard"}, "Trace": ["744"], "window": [10, 30]}
    iso_struct = {"name": ["isoGDGTs"], "GDGT_dict": {"1302": "GDGT-0", "1300": "GDGT-1", "1298": "GDGT-2", "1296": "GDGT-3", "1292": ["GDGT-4", "GDGT-4'"]}, "Trace": ["1302", "1300", "1298", "1296", "1292"], "window": [0, 35]}
    br_struct = {
        "name": ["brGDGTs"],
        "GDGT_dict": {"1050": ["IIIa","IIIa''", "IIIa'"], "1048": ["IIIb", "IIIb'"], "1046": ["IIIc", "IIIc'"], "1036": ["IIa", "IIa'"], "1034": ["IIb", "IIb'"], "1032": ["IIc", "IIc'"], "1022": "Ia", "1020": "Ib", "1018": "Ic"},
        "Trace": ["1050", "1048", "1046", "1036", "1034", "1032", "1022", "1020", "1018"],
        "window": [20, 55],
    }
    oh_struct = {"name": ["OH-GDGTs"], "GDGT_dict": {"1300": "OH-GDGT-0", "1298": ["OH-GDGT-1", "2OH-GDGT-0"], "1296": "OH-GDGT-2"}, "Trace": ["1300", "1298", "1296"], "window": [35, 50]}

    # Map the user's input to the corresponding data structures
    gdgt_map = {"1": iso_struct, "2": br_struct, "3": oh_struct}

    # All GDGT types selected
    if gdgt_oi == "4":
        gdgt_oi = "1,2,3"

    # Initialize the final structure with ref_struct values
    combined_struct = {"name": ref_struct["name"].copy(), "GDGT_dict": ref_struct["GDGT_dict"].copy(), "Trace": ref_struct["Trace"].copy(), "window": ref_struct["window"].copy()}

    # Split the user input and iterate through each selection
    name_list = [ref_struct["name"]]
    gdgt_dict_list = [ref_struct["GDGT_dict"]]
    trace_list = [ref_struct["Trace"]]
    window_list = [ref_struct["window"]]

    if len(gdgt_oi) > 1:
        selected_types = gdgt_oi.split(",")
    else:
        selected_types = gdgt_oi
    for gdgt_type in selected_types:
        gdgt_type = gdgt_type.strip()
        if gdgt_type in gdgt_map:
            struct = gdgt_map[gdgt_type]
            name_list.append(struct["name"])
            gdgt_dict_list.append(struct["GDGT_dict"])
            trace_list.append(struct["Trace"])
            window_list.append(struct["window"])

    return {
        "names": name_list,
        "GDGT_dict": gdgt_dict_list,
        "Trace": trace_list,
        "window": window_list,}
import json
from pathlib import Path
import sys
import copy

GDGT_META_PATH = Path(__file__).with_name("gdgt_meta.json")
GDGT_META_DEFAULT_PATH = Path(__file__).with_name("gdgt_meta_default.json")

try:
    from PyQt5 import QtWidgets, QtCore
except ImportError:
    QtWidgets = None
    QtCore = None

def load_gdgt_meta_json(path: Path = GDGT_META_PATH) -> dict:
    """Load the JSON config (Standard, brGDGTs, etc.)."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_gdgt_default_json(path: Path = GDGT_META_DEFAULT_PATH) -> dict:
    """Load the factory-default GDGT JSON config."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_gdgt_meta_json(meta_json: dict, path: Path = GDGT_META_PATH) -> None:
    """Save the JSON config back to disk."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)
        
class _GroupWidget(QtWidgets.QGroupBox):
    """
    A "card" for a single GDGT group: enable checkbox, name, RT window, trace rows.
    Uses data in the form:
        {
            "name": "brGDGTs",
            "enabled": True,
            "window": [20.0, 40.0],
            "traces": { "1050": "IIIa, IIIa'', IIIa'", ... }
        }
    """
    remove_me = QtCore.pyqtSignal(object)                             
    def __init__(self, group_data: dict, parent=None, hide_enable=False):
        super().__init__(parent)
        self._data = group_data
        self.trace_rows = []

        self.setTitle("")
        main_layout = QtWidgets.QVBoxLayout(self)

        # --- HEADER ROW (Enable + Name + X DELETE BUTTON) ---
        header = QtWidgets.QHBoxLayout()

        # Enable checkbox
        if not hide_enable:
            self.enable_cb = QtWidgets.QCheckBox("Enable")
            self.enable_cb.setChecked(group_data.get("enabled", True))
            header.addWidget(self.enable_cb)
        else:
            self.enable_cb = None

        # Name
        self.name_edit = QtWidgets.QLineEdit(group_data.get("name", "GDGT"))
        header.addWidget(self.name_edit, 1)

        # X delete button (top-right)
        delete_btn = QtWidgets.QPushButton("✕")
        delete_btn.setMaximumWidth(32)
        delete_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #3B4954; color: #F7ECE1; }"
        )
        delete_btn.clicked.connect(lambda: self.remove_me.emit(self))
        header.addWidget(delete_btn)

        main_layout.addLayout(header)

        # --- RT window row ---
        win_layout = QtWidgets.QHBoxLayout()
        win_layout.addWidget(QtWidgets.QLabel("RT min:"))

        win = group_data.get("window", [0.0, 0.0])
        min_val = win[0] if len(win) >= 1 else 0.0
        max_val = win[1] if len(win) >= 2 else 0.0

        self.min_edit = QtWidgets.QLineEdit(str(min_val))
        self.min_edit.setMaximumWidth(80)
        win_layout.addWidget(self.min_edit)

        win_layout.addSpacing(12)
        win_layout.addWidget(QtWidgets.QLabel("RT max:"))
        self.max_edit = QtWidgets.QLineEdit(str(max_val))
        self.max_edit.setMaximumWidth(80)
        win_layout.addWidget(self.max_edit)

        win_layout.addStretch(1)
        main_layout.addLayout(win_layout)

        # --- trace list area ---
        self.traces_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(self.traces_layout)

        for trace_id, label in group_data.get("traces", {}).items():
            self._add_trace_row(trace_id, label)

        # --- add-trace button ---
        add_row = QtWidgets.QHBoxLayout()
        add_row.addStretch(1)
        add_btn = QtWidgets.QPushButton("+ Add trace")
        add_btn.clicked.connect(self.add_new_trace)
        add_row.addWidget(add_btn)
        main_layout.addLayout(add_row)

        # styling (roughly matches your Toga colours)
        self.setStyleSheet(
            """
            QGroupBox {
                border: 1px solid #3B4954;
                border-radius: 6px;
                margin-top: 8px;
                background-color: #c2c5aa;
            }
            """
        )

    def _add_trace_row(self, trace_id: str, label: str):
        row_widget = QtWidgets.QWidget(self)
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        trace_edit = QtWidgets.QLineEdit(trace_id)
        trace_edit.setPlaceholderText("Trace ID")
        trace_edit.setMaximumWidth(120)

        label_edit = QtWidgets.QLineEdit(label)
        label_edit.setPlaceholderText("Label (e.g., IIIa, IIIa'', IIIa')")

        del_btn = QtWidgets.QPushButton("✕")
        del_btn.setMaximumWidth(32)

        def on_delete():
            row_widget.setParent(None)
            self.trace_rows = [r for r in self.trace_rows if r[0] is not row_widget]

        del_btn.clicked.connect(on_delete)

        row_layout.addWidget(trace_edit)
        row_layout.addWidget(label_edit, 1)
        row_layout.addWidget(del_btn)

        self.traces_layout.addWidget(row_widget)
        self.trace_rows.append((row_widget, trace_edit, label_edit))

    def add_new_trace(self):
        idx = len(self.trace_rows) + 1
        self._add_trace_row(f"new_trace_{idx}", "New Label")

    def to_group_data(self) -> dict:
        """
        Convert the widget state back to a plain dict:
        {
            "name": str,
            "enabled": bool,  # True if no checkbox (e.g., reference), else checkbox state
            "window": [min_rt, max_rt],
            "traces": {trace_id: label_str, ...}
        }
        """
        name = self.name_edit.text().strip() or self._data.get("name", "GDGT")
    
        # RT window
        try:
            min_rt = float(self.min_edit.text())
            max_rt = float(self.max_edit.text())
        except ValueError:
            raise ValueError(f"Invalid RT window for group '{name}'")
    
        traces = {}
        for _, trace_edit, label_edit in self.trace_rows:
            tid = trace_edit.text().strip()
            lbl = label_edit.text().strip()
            if tid and lbl:
                traces[tid] = lbl
        enabled = True if self.enable_cb is None else self.enable_cb.isChecked()
    
        return {
            "name": name,
            "enabled": enabled,
            "window": [min_rt, max_rt],
            "traces": traces,
        }

class _GdgtMetaDialog(QtWidgets.QDialog):
    """
    Dialog that holds multiple _GroupWidget "cards" and returns updated group data.
    """

    def __init__(self, groups_data: list[dict], default_groups_data: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("chromatoPy – GDGT Settings")
        self.resize(900, 600)

        # live state (current config being shown/edited)
        self._groups_data = copy.deepcopy(groups_data)
        # factory defaults (from gdgt_meta_default.json)
        self._default_groups_data = copy.deepcopy(default_groups_data)

        self.group_widgets: list[_GroupWidget] = []

        main_layout = QtWidgets.QVBoxLayout(self)

        # Scroll area for the cards
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        self.groups_layout = QtWidgets.QVBoxLayout(scroll_widget)
        self.groups_layout.setContentsMargins(8, 8, 8, 8)
        self.groups_layout.setSpacing(12)
        scroll.setWidget(scroll_widget)

        main_layout.addWidget(scroll, 1)

        # build initial group cards from current data
        self._build_group_widgets(self._groups_data)

        # footer buttons
        footer = QtWidgets.QHBoxLayout()
        main_layout.addLayout(footer)

        add_group_btn = QtWidgets.QPushButton("+ Add GDGT type")
        add_group_btn.clicked.connect(self.add_group)
        footer.addWidget(add_group_btn)

        restore_btn = QtWidgets.QPushButton("Restore Defaults")
        restore_btn.clicked.connect(self.restore_defaults)
        footer.addWidget(restore_btn)

        footer.addStretch(1)

        ok_btn = QtWidgets.QPushButton("Save && Use")
        ok_btn.clicked.connect(self.on_accept)
        footer.addWidget(ok_btn)

    def _build_group_widgets(self, groups_data: list[dict]):
        """Clear and rebuild the card widgets from groups_data."""
        # clear layout
        while self.groups_layout.count():
            item = self.groups_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        self.group_widgets.clear()

        for i, gdata in enumerate(groups_data):
            is_reference = (i == 0)
            gw = _GroupWidget(gdata, parent=self, hide_enable=is_reference)
            gw.remove_me.connect(self._on_remove_group)
            self.groups_layout.addWidget(gw)
            self.group_widgets.append(gw)

        self.groups_layout.addStretch(1)

    def add_group(self):
        # default new group
        gdata = {
            "name": f"NewGDGT_{len(self.group_widgets) + 1}",
            "enabled": True,
            "window": [0.0, 0.0],
            "traces": {"new_trace": "New Label"},
        }
        gw = _GroupWidget(gdata, parent=self)
        gw.remove_me.connect(self._on_remove_group)
        self.groups_layout.insertWidget(self.groups_layout.count() - 1, gw)
        self.group_widgets.append(gw)

    def _on_remove_group(self, gw: _GroupWidget):
        """Remove a group card from the dialog."""
        if gw in self.group_widgets:
            self.group_widgets.remove(gw)
        gw.setParent(None)
        gw.deleteLater()

    def restore_defaults(self):
        """Reset all GDGT groups to the factory defaults from gdgt_meta_default.json."""
        self._groups_data = copy.deepcopy(self._default_groups_data)
        self._build_group_widgets(self._groups_data)

    def on_accept(self):
        try:
            updated = []
            for gw in self.group_widgets:
                updated.append(gw.to_group_data())
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid configuration", str(exc))
            return
        self._groups_data = updated
        self.accept()

    def get_groups_data(self) -> list[dict]:
        return self._groups_data

def json_to_groups(meta_json: dict) -> list[dict]:
    """
    Convert gdgt_meta.json structure into the list-of-dicts
    used by _GdgtMetaDialog / _GroupWidget.
    """
    groups = []

    # ensure Standard is first if present
    keys = list(meta_json.keys())
    if "Standard" in keys:
        keys.remove("Standard")
        keys.insert(0, "Standard")

    for name in keys:
        cfg = meta_json[name]
        groups.append(
            {
                "name": name,
                "enabled": cfg.get("checked", True),
                "window": cfg.get("window", [0.0, 0.0]),
                "traces": cfg.get("traces", {}),
            }
        )
    return groups

def groups_to_json(groups: list[dict]) -> dict:
    """
    Convert the dialog groups back into the gdgt_meta.json structure.
    """
    meta_json = {}

    for g in groups:
        name = g["name"].strip() or "GDGT"
        meta_json[name] = {
            "checked": g.get("enabled", True),
            "traces": g.get("traces", {}),
            "window": g.get("window", [0.0, 0.0]),
        }

    return meta_json

def edit_gdgt_meta_qt(initial_meta: dict) -> dict:
    """
    Open the GDGT settings dialog.

    - Loads current settings from gdgt_meta.json (if present).
    - Uses gdgt_meta_default.json as factory defaults.
    - Lets the user edit them.
    - "Restore Defaults" repopulates from gdgt_meta_default.json.
    - Saves changes back to gdgt_meta.json.
    - Returns a gdgt_meta-style dict:
        {"names": [...], "GDGT_dict": [...], "Trace": [...], "window": [...]}
      containing only the *enabled* groups.
    """
    if QtWidgets is None:
        print("PyQt5 not available; using default GDGT metadata without GUI.")
        return initial_meta

    # ---- load default JSON (must exist) ----
    default_json = load_gdgt_default_json()
    default_groups = json_to_groups(default_json)

    # ---- load current JSON (or fall back to defaults) ----
    try:
        meta_json = load_gdgt_meta_json()
    except FileNotFoundError:
        # if no current config, start from defaults and write gdgt_meta.json once
        meta_json = default_json
        save_gdgt_meta_json(meta_json)

    groups_data = json_to_groups(meta_json)

    # ---- show dialog ----
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv or ["chromatoPy"])
        owns_app = True

    dlg = _GdgtMetaDialog(groups_data, default_groups)
    dlg.setWindowModality(QtCore.Qt.ApplicationModal)
    dlg.raise_()
    dlg.activateWindow()

    result = dlg.exec_()

    if owns_app:
        app.quit()

    if result != QtWidgets.QDialog.Accepted:
        # user cancelled → keep old JSON and convert it to gdgt_meta-style
        return json_to_gdgt_meta(meta_json)

    updated_groups = dlg.get_groups_data()

    # ---- update JSON on disk ----
    new_json = groups_to_json(updated_groups)
    save_gdgt_meta_json(new_json)

    # ---- convert back to gdgt_meta-style dict (for old pipeline) ----
    return json_to_gdgt_meta(new_json)

def json_to_gdgt_meta(meta_json: dict) -> dict:
    """
    Convert gdgt_meta.json structure to chromatoPy's traditional metadata:
      {"names": [...], "GDGT_dict": [...], "Trace": [...], "window": [...]}

    Only includes groups with checked == True.
    """
    new_meta = {"names": [], "GDGT_dict": [], "Trace": [], "window": []}

    # ensure Standard appears first if present
    keys = list(meta_json.keys())
    if "Standard" in keys:
        keys.remove("Standard")
        keys.insert(0, "Standard")

    for name in keys:
        cfg = meta_json[name]

        # skip disabled groups
        if not cfg.get("checked", True):
            continue

        traces_dict = {}
        trace_ids = []

        for tid, label_str in cfg.get("traces", {}).items():
            tid = tid.strip()
            label_str = label_str.strip()

            if not tid or not label_str:
                continue

            # split label string if needed
            parts = [p.strip() for p in label_str.split(",") if p.strip()]
            if len(parts) == 1:
                traces_dict[tid] = parts[0]
            else:
                traces_dict[tid] = parts

            trace_ids.append(tid)

        # Append to old-style structure
        new_meta["names"].append([name])
        new_meta["GDGT_dict"].append(traces_dict)
        new_meta["Trace"].append(trace_ids)
        new_meta["window"].append(cfg.get("window", [0.0, 0.0]))

    return new_meta