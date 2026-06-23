from pathlib import Path

# Project root is the directory that contains this package (src/).
# Resolving via __file__ makes every path independent of the current
# working directory, so the project can be run from anywhere.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

CONFIG_FILE_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH: Path = PROJECT_ROOT / "params.yaml"
