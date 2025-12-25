import os
import subprocess
import sys

def open_folder(path: str):
    if not os.path.isdir(path):
        return False, "Path does not exist"

    try:
        if sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", path])
        elif sys.platform.startswith("darwin"):
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("win"):
            os.startfile(path)
        else:
            return False, "Unsupported OS"
    except Exception as e:
        return False, str(e)

    return True, None
