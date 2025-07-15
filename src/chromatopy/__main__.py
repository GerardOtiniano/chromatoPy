from .chromatoPy_front_ui_gen import ChromatoPyApp
def main() -> None:
    app = ChromatoPyApp("ChromatoPy", "com.GerardOtiniano.chromatopy")
    app.main_loop()

if __name__ == "__main__":
    main()