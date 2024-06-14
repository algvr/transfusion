import os
from pathlib import Path

SPLITS_DIR = "splits"
data_roots = {
    "egtea": Path(os.path.expandvars("$DATA/EGTEAp/")),
    "epic": Path(os.path.expandvars("$DATA/EK/")),
    "epicv": Path(os.path.expandvars("$DATA/EK/")),
    "ego4d": Path(os.path.expandvars("$DATA/Ego4d/v1")),
    "ego4djpg": Path(os.path.expandvars("$DATA/Ego4d/v1")),
    "ego4djpgv2": Path(os.path.expandvars("$DATA/Ego4d/v2")),
}
paper_actors = set(
    [
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
        "P07",
        "P08",
        "P10",
        "P12",
        "P13",
        "P14",
        "P15",
        "P16",
        "P17",
        "P19",
        "P20",
        "P21",
        "P22",
        "P23",
        "P24",
        "P25",
        "P26",
        "P27",
        "P28",
        "P29",
        "P30",
        "P31",
        "P31",
    ]
)
