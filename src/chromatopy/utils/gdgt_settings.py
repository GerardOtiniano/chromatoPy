import toga
from .GDGT_configuration import open_gdgt_selector

class GDGTSettings(toga.App):

    def startup(self):
        self.main_window = toga.MainWindow(title="GDGT Settings", size=(700, 700), resizable=True)
        self.open_gdgt_settings()

    def open_gdgt_settings(self):
        open_gdgt_selector(self)

# def main():
#     return GDGTSettings("ChromatoPy", "com.GerardOtiniano.chromatopy")
#
# if __name__ == "__main__":
#     main().main_loop()

def run_ui():
    """Launch the settings window and block until closed."""
    app = GDGTSettings("ChromatoPy", "com.GerardOtiniano.chromatopy")
    app.main_loop()

def main():
    return GDGTSettings("ChromatoPy", "com.GerardOtiniano.chromatopy")

if __name__ == "__main__":
    run_ui()