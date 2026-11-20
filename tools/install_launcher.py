"""Register this checkout in the current user's Linux application menu."""

import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
applications = data / "applications"
applications.mkdir(parents=True, exist_ok=True)
executable = str(root / "run.sh")
for char in ("\\", '"', "`", "$"):
    executable = executable.replace(char, "\\" + char)
launcher = applications / "portrait-processor.desktop"
launcher.write_text(
    "[Desktop Entry]\n"
    "Type=Application\n"
    "Name=Portrait Processor\n"
    "Comment=Remove backgrounds and prepare symmetric 948×1048 portraits\n"
    f'Exec="{executable}" %F\n'
    f"Icon={root / 'assets' / 'portrait-processor.svg'}\n"
    "Terminal=false\n"
    "Categories=Graphics;Photography;\n"
    "StartupNotify=true\n",
    encoding="utf-8",
)
print(f"Application menu shortcut: {launcher}")
