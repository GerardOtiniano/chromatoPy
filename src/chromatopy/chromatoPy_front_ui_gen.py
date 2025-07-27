import os
import asyncio
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from toga.dialogs import InfoDialog
import logging
from datetime import datetime

from .hplc_integration_gen import hplc_integration_gen
from .config.Integration_Settings import load_integration_settings, open_integration_settings
from .config.Plot_Settings import load_plot_settings, open_plot_settings


class ChromatoPyApp(toga.App):
    def __init__(self, formal_name, app_id):
        super().__init__(formal_name=formal_name, app_id=app_id)

    def logging_setup(self, folder_path):
        logs_dir = os.path.join(folder_path, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        log_filename = os.path.join(logs_dir, f"chromatopy_{timestamp}.log")

        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        logging.basicConfig(
            filename=log_filename,
            filemode='w',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

        logging.info("ChromatoPy initialized.")

    def startup(self):
        # Main window
        self.main_window = toga.MainWindow(title="ChromatoPy", size=(600, 600), resizable=True)

        # ─── Layout container ───
        main_box = toga.Box(style=Pack(direction="column", margin=10, background_color="#F7ECE1"))

        # ─── Image ───
        image_path = "Icons/chromatoPy2.png"
        image = toga.Image(image_path)
        image_view = toga.ImageView(image, style=Pack(width=250, height=250, margin=(20, 175, 0, 175)))
        main_box.add(image_view)

        # Path Input
        self.path_input = toga.TextInput(placeholder="Enter/Path/To/Raw/Data",
                                         style=Pack(margin_left=90, height=25, width=330, font_size=12,
                                                    background_color="#3B4954", color="#F7ECE1"))

        browse_button = toga.Button("Browse", on_press=self.select_folder,
                                    style=Pack(margin_right=90, height=25, width=90,
                                               background_color="#3B4954", color="#F7ECE1", font_weight="bold",
                                               font_size=12))

        folder_row = toga.Box(style=Pack(direction=ROW, margin=(20, 0, 20, 0)))
        folder_row.add(self.path_input)
        folder_row.add(browse_button)
        main_box.add(folder_row)

        settings_btn = toga.Button("Integration Settings", on_press=self.on_integration_settings,
                                   style=Pack(height=25, width=360, margin=(0, 120, 0, 120),
                                              background_color="#3B4954", color="#F7ECE1", font_weight="bold",
                                              font_size=12))
        main_box.add(settings_btn)
        plot_settings_btn = toga.Button("Plot Settings", on_press=self.on_plot_settings,
                                        style=Pack(height=25, width=360, margin=(15, 120, 0, 120),
                                                   background_color="#3B4954", color="#F7ECE1", font_weight="bold",
                                                   font_size=12))
        main_box.add(plot_settings_btn)

        # Start-processing button
        start_btn = toga.Button("Start Processing", on_press=self.validate_and_start,
                                style=Pack(height=30, width=400, margin=(20, 100, 0, 100),
                                           background_color="#3B4954", color="#F7ECE1", font_weight="bold",
                                           font_size=14))
        main_box.add(start_btn)

        # Error / status label
        self.error_label = toga.Label("", style=Pack(color="#EA0F0B", margin=(20, 300, 20, 20), font_weight="bold",
                                                     font_size=12))

        main_box.add(self.error_label)

        # Set content & show
        self.main_window.content = main_box
        self.main_window.show()

    async def select_folder(self, widget):
        home_dir = os.path.expanduser("~")
        dialog = toga.SelectFolderDialog(title="Select Raw Data Folder", initial_directory=home_dir)
        folder = await self.main_window.dialog(dialog)
        if folder:
            self.path_input.value = folder

    def show_info_dialog(self, title, message):
        async def _show():
            dialog = InfoDialog(title=title, message=message)
            await self.main_window.dialog(dialog)

        asyncio.create_task(_show())

    def on_integration_settings(self, widget):
        open_integration_settings(self)

    def on_plot_settings(self, widget):
        open_plot_settings(self)

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

        self.logging_setup(self.path_input.value)
        logging.info("The inputted folder is valid")

        settings = load_integration_settings()
        settings["folder_path"] = folder
        plot_settings = load_plot_settings()

        compounds = plot_settings.pop("compounds")
        compounds = compounds.split(",")

        plot_settings = {
            "headers": [plot_settings["time_header"],   plot_settings["signal_header"]],
            "window_bounds": [  plot_settings["min_window"],   plot_settings["max_window"]],
            "compounds": compounds
        }

        for setting in plot_settings:
            settings[setting] = plot_settings[setting]

        try:
            logging.info("Running HPLC integration.")
            result = hplc_integration_gen(**settings)

            if result[0] == "aborted":
                self.error_label.text = "Integration aborted by user. Partial completion."
                logging.warning(f"Integration aborted by user at sample: {result[1]}")
                return

            if result[0] == "compound_error":
                self.error_label.text = "Number of peak clicks weren't equal to the number of compounds."
                logging.error(f"The number of peak clicks weren't equal to the number of compounds for sample: {result[1]}")
                return

            if result[0] == "success":
                logging.info("HPLC integration completed successfully.")

            self.show_info_dialog("Done", "HPLC integration completed successfully.")

        except Exception as e:
            self.error_label.text = f"Error: {e}"
            logging.error(f"Error: {e}", exc_info=True)

def main():
    return ChromatoPyApp("ChromatoPy", "com.GerardOtiniano.chromatopy")