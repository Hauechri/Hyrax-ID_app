"""
Hyrax-ID desktop app entry point.

Run with:
    .venv\\Scripts\\python.exe app\\main.py
"""
import os
import sys

# Must happen before numba is imported anywhere downstream (via librosa in
# the vendored engine) so its JIT cache never tries to write into a
# possibly read-only Nuitka --standalone install directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import user_data_dir  # noqa: E402

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(user_data_dir(), "numba_cache"))

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtGui import QIcon  # noqa: E402

import paths  # noqa: E402
from main_window import MainWindow  # noqa: E402


def main():
    app = QApplication(sys.argv)
    icon_path = os.path.join(paths.ASSETS_DIR, "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
