"""Automated document and legal evidence dossier generation module for Haddu."""

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

from haddu.reporting.generator import (
    compile_dossier_pdf,
    find_nearest_village,
    generate_dossier_visuals,
    render_dossier_html,
)

__all__ = [
    "compile_dossier_pdf",
    "find_nearest_village",
    "generate_dossier_visuals",
    "render_dossier_html",
]
