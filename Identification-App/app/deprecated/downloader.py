"""
Downloads the denoiser model (01_ACS.pk) on demand, off the GUI thread.
Not bundled by Nuitka -- see build_exe.ps1 -- so this path always runs,
including in dev.
"""
import os

import requests
from PySide6.QtCore import QObject, Signal

# Direct download endpoint. The share-page URL from the original project
# notes (.../getlink/<id>/...) returns an HTML landing page, not the file
# itself -- confirmed by inspection; this /dl/ URL is the real one.
DENOISER_URL = "https://faubox.rrze.uni-erlangen.de/dl/fi93Jos1YBfsvkeAwkRMwv/01_ACS.pk"


def denoiser_cache_path():
    import paths

    return os.path.join(paths.user_data_dir(), "01_ACS.pk")


def denoiser_is_cached():
    return os.path.isfile(denoiser_cache_path())


class DownloadWorker(QObject):
    progress = Signal(int, int)  # bytes_read, total_bytes (-1 if unknown)
    finished = Signal()
    error = Signal(str)

    def run(self):
        final_path = denoiser_cache_path()
        part_path = final_path + ".part"
        try:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            with requests.get(DENOISER_URL, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", -1))
                read = 0
                with open(part_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        f.write(chunk)
                        read += len(chunk)
                        self.progress.emit(read, total)

            # Only replace the final file once the download completed fully,
            # so a crash/interrupt never leaves a corrupt file mistaken for valid.
            os.replace(part_path, final_path)
        except Exception as exc:
            if os.path.exists(part_path):
                os.remove(part_path)
            self.error.emit(str(exc))
            return
        finally:
            self.finished.emit()
