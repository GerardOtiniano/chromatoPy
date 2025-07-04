import os
import toga
from toga.style import Pack
from .hplc_integration import hplc_integration
from .config.GDGT_configuration import open_gdgt_selector
from .config.Integration_Settings import open_settings, load_integration_settings


class ChromatoPyApp(toga.App):
    def __init__(self, formal_name, app_id):
        super().__init__(formal_name=formal_name, app_id=app_id)

    def startup(self):
        # Main window
        self.main_window = toga.MainWindow(title="ChromatoPy", size=(400, 420), resizable=False)

        # ─── Layout container ───
        main_box = toga.Box(style=Pack(direction="column", padding=20))

        # Path input
        self.path_input = toga.TextInput(
            placeholder="Enter/Path/To/Raw/Data",
            style=Pack(width=360, padding=(0, 0, 10, 0)),
        )
        main_box.add(self.path_input)

        # GDGT selector & settings buttons
        gdgt_btn = toga.Button("Target GDGTs", on_press=self.on_gdgt_selector, style=Pack(padding=5))
        settings_btn = toga.Button("Integration Settings", on_press=self.on_settings, style=Pack(padding=5))
        main_box.add(gdgt_btn)
        main_box.add(settings_btn)

        # Start-processing button
        start_btn = toga.Button("Start Processing", on_press=self.validate_and_start, style=Pack(padding=(15, 0)))
        main_box.add(start_btn)

        # Error / status label
        self.error_label = toga.Label("", style=Pack(color="red", padding=(5, 0)))
        main_box.add(self.error_label)

        # Set content & show
        self.main_window.content = main_box
        self.main_window.show()

    def on_gdgt_selector(self, widget):
        open_gdgt_selector(self)

    def on_settings(self, widget):
        open_settings(self)

#     def validate_and_start(self, widget):
#         self.error_label.text = ""  # clear prior message
#         folder = self.path_input.value.strip().strip("'\"")
#
#         if not os.path.isdir(folder):
#             self.error_label.text = "Folder does not exist"
#             return
#
#         csvs = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
#         if not csvs:
#             self.error_label.text = "No .csv files found in that folder"
#             return
#
#         settings = load_integration_settings()
#         settings["folder_path"] = folder
#         try:
#             hplc_integration(**settings)
#             self.main_window.info_dialog("Done", "HPLC integration completed successfully.")
#         except Exception as e:
#             self.error_label.text = f"Error: {e}"
    def validate_and_start(self, widget):
        self.error_label.text = ""  # clear prior message
        folder = self.path_input.value.strip().strip("'\"")

        if not os.path.isdir(folder):
            self.error_label.text = "Folder does not exist"
            return

        csvs = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
        if not csvs:
            self.error_label.text = "No .csv files found in that folder"
            return

        settings = load_integration_settings()
        settings["folder_path"] = folder
        try:
            result = hplc_integration(**settings)

            if result == "aborted":
                self.error_label.text = "Integration aborted by user."
                return

            self.main_window.info_dialog("Done", "HPLC integration completed successfully.")
        except Exception as e:
            self.error_label.text = f"Error: {e}"


def main():
    return ChromatoPyApp("ChromatoPy", "com.GerardOtiniano.chromatopy")