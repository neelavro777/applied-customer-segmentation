import os
import yaml
from pathlib import Path
import logging
from src.constants import PROJECT_ROOT

def read_yaml(path_to_yaml: Path) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        logging.info(f"yaml file: {path_to_yaml} loaded successfully")
        return content

def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")

def resolve_path(path) -> Path:
    """Resolve a path relative to the project root.

    Absolute paths are returned unchanged; relative paths are anchored to
    PROJECT_ROOT so behaviour does not depend on the current working dir.
    """
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
