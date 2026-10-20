"""Load small experiment overlays while keeping existing standalone YAMLs valid."""

from pathlib import Path

from omegaconf import OmegaConf


def load_config(path, _stack=()):
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError(f"Cyclic _base_ configuration: {path}")
    config = OmegaConf.load(path)
    base = config.pop("_base_", None)
    if base is None:
        return config
    return OmegaConf.merge(load_config(path.parent / base, (*_stack, path)), config)
