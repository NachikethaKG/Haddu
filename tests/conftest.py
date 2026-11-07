"""Pytest global configuration and environment initialization for Haddu test suite."""

import os
import sys

# On Windows, ensure GTK / Pango / GObject DLL directories are discoverable for WeasyPrint
if sys.platform == "win32":
    for _path in os.environ.get("PATH", "").split(os.pathsep):
        if (
            _path
            and os.path.isdir(_path)
            and any(
                os.path.exists(os.path.join(_path, dll))
                for dll in ("libgobject-2.0-0.dll", "gobject-2.0-0.dll", "libpango-1.0-0.dll")
            )
        ):
            try:
                os.add_dll_directory(_path)
            except (OSError, ValueError):
                pass
