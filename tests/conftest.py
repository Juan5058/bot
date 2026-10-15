import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("JARVIS_MODE", "simulated")
os.environ.setdefault("JARVIS_AI_ENABLED", "false")
os.environ.setdefault("JARVIS_LOG_LEVEL", "INFO")
