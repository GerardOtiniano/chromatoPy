# This file defines: peak_label_editor()
import sys
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QHBoxLayout, QMessageBox
)

DEFAULT_LABELS = {
    "Peak Labels": ["C16", "C17", "C18", "C19", "C20Z2",
                    "C20", "C21", "C22", "C23", "C24",
                    "C25", "C26", "C27", "C28", "C29",
                    "C30", "C31", "C32"],
    "x limits": [3, 17]
}

MODULE_DIR = os.path.dirname(__file__)
DEFAULT_JSON = os.path.join(MODULE_DIR, "peak_labels.json")

def load_peak_labels(json_path: str = DEFAULT_JSON):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return DEFAULT_LABELS

def save_peak_labels(data, json_path: str = DEFAULT_JSON):
    # debug print so you can see where it goes:
    # print(f"⤷ writing peak_labels.json to: {json_path!r}")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

class PeakLabelEditor(QDialog):
    def __init__(self, json_path: str = DEFAULT_JSON):
        super().__init__()
        self.json_path = json_path
        self.label_dict = load_peak_labels(json_path)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Peak Labels (comma-separated):"))
        self.labels_input = QLineEdit(", ".join(self.label_dict.get("Peak Labels", [])))
        layout.addWidget(self.labels_input)

        layout.addWidget(QLabel("X limits (e.g., 5, 17):"))
        self.xlims_input = QLineEdit(", ".join(str(x) for x in self.label_dict.get("x limits", [5, 17])))
        layout.addWidget(self.xlims_input)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_and_exit)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def save_and_exit(self):
        try:
            labels = [l.strip() for l in self.labels_input.text().split(",") if l.strip()]
            xlims = [float(x.strip()) for x in self.xlims_input.text().split(",")]
            if len(xlims) != 2:
                raise ValueError("x limits must have exactly two numbers.")
            new_data = {"Peak Labels": labels, "x limits": xlims}
            save_peak_labels(new_data, self.json_path)
            QMessageBox.information(self, "Saved", f"Saved to {self.json_path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

def peak_label_editor(json_path: str = DEFAULT_JSON):
    app = QApplication.instance() or QApplication(sys.argv)
    editor = PeakLabelEditor(json_path)
    editor.exec_()
