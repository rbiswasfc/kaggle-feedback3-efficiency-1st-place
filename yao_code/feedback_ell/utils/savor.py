"""
@created by: heyao
@created at: 2022-09-21 18:48:30
"""
from omegaconf import OmegaConf


def save_yaml(config, to_filename):
    s = OmegaConf.to_yaml(config)
    with open(to_filename, "w") as f:
        f.write(s)
    return True
