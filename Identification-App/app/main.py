"""
Hyrax-ID desktop app entry point.

Run with:
    .venv\\Scripts\\python.exe app\\main.py
"""
import multiprocessing
import os
import sys

# Must happen before numba is imported anywhere downstream (via librosa in
# the vendored engine) so its JIT cache never tries to write into a
# possibly read-only Nuitka --standalone install directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import user_data_dir  # noqa: E402

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(user_data_dir(), "numba_cache"))


def main():
    # The spectrogram viewer's batch runs on worker *processes*. On Windows (and in a frozen exe) a worker
    # starts by re-importing this module, so everything heavy -- Qt, the whole GUI and with it torch -- is
    # imported here, inside main(), and not at module level: a worker then only needs the light top part.
    multiprocessing.freeze_support()

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    import paths
    from main_window import MainWindow

    app = QApplication(sys.argv)
    icon_path = os.path.join(paths.ASSETS_DIR, "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
