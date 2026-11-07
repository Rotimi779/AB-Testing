import sys
from pathlib import Path

# Make the project root importable when running `pytest` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
