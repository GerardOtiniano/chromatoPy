import toga
from .Integration_configuration import open_settings

class IntegrationSettings(toga.App):

    def startup(self):
        self.main_window = toga.MainWindow(title="Integration Settings", size=(700, 700), resizable=True)
        self.open_integration_settings()

    def open_integration_settings(self):
        open_settings(self)

def main():
    return IntegrationSettings("ChromatoPy", "com.GerardOtiniano.chromatopy")

if __name__ == "__main__":
    main().main_loop()
