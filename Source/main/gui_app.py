"""
Backward-compatibility shim.
The GUI application has been organized into Source/ui/gui_app.py.
This shim ensures that any existing shortcut, batch file, or script calling
`Source/main/gui_app.py` continues to work seamlessly.
"""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CURRENT_DIR.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ui.gui_app import App, main

if __name__ == "__main__":
    main()
