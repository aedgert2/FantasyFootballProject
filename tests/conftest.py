import sys
from pathlib import Path

# The pipeline is a set of scripts, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
