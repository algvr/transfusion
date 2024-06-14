import datetime
import json
import logging
import os
import shutil
from pathlib import Path


runs_path = Path(os.environ["RUNS"])
plots_dir = "plots"


DEBUG_ACTORS = {
    "ego4d": [
        "59815805-de31-4993-8f5e-f12b1537bcfc",
        "134a4c63-583a-4e64-8cf9-002b6d26cdf1"
    ],
}
DEBUG_ACTORS["ego4djpg"] = DEBUG_ACTORS["ego4d"]
DEBUG_ACTORS["ego4djpgv2"] = DEBUG_ACTORS["ego4d"]

ALL_ACTORS = {
    "epic": None,
    "epicv": None,
    "egtea": None,
    "ego4d": None,
    "ego4djpg": None,
    "ego4djpgv2": None,
}


def get_datasets_from_name(dataset_name):
    datasets = dataset_name.split("_")
    return datasets


def get_label_merging(label_merging_path):
    if label_merging_path:
        with open(label_merging_path, "r") as fp:
            label_merging = json.load(fp)

        label_merging["ego4d"]["fine"] = label_merging["epic"]["fine"].copy()
        label_merging["ego4d"]["fine"].update(label_merging["egtea"]["fine"].copy())
        label_merging["ego4djpg"] = label_merging["ego4d"]
        label_merging["ego4djpgv2"] = label_merging["ego4d"]
    else:
        label_merging = {}

    return label_merging


def copy_src(path_from, path_to):
    """
    Saves files under path_from to a zip file in path_to.

    There should not be large files (e.g. checkpoints or images) in path_from.
    Args:
        path_from: folder containing the source code that should be saved.
        path_to: folder where the resulting zip file will be saved.
    """
    assert os.path.isdir(path_from)
    # Collect all files and folders that contain python files
    tmp_folder = os.path.join(path_to, "src/")
    os.mkdir(tmp_folder)

    from_folder = os.path.basename(path_from)
    shutil.copy2(path_from, os.path.join(tmp_folder, from_folder), overwrite=True)
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    path_archive = os.path.join(path_to, "src_{}".format(time_str))
    shutil.make_archive(path_archive, "zip", tmp_folder)
    try:
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        raise Warning("Something went wrong; could not delete {}".format(tmp_folder))
        pass
    logging.info("Copied folder {} to {}".format(path_from, path_archive))
    return path_archive


def get_run_id(experiment):
    exp_path = runs_path.joinpath(experiment)
    exp_path.mkdir(parents=True, exist_ok=True)

    runs_so_far = sorted([int(x.name) for x in exp_path.iterdir()])
    no_runs = runs_so_far[-1] if runs_so_far else 1
    return no_runs + 1


def make_run_dir(run_id, experiment):
    run_dir = runs_path.joinpath(experiment, str(run_id))
    run_dir.mkdir(parents=True, exist_ok=False)
    plot_dir = run_dir.joinpath(plots_dir)
    plot_dir.mkdir(parents=True, exist_ok=False)
    plot_dir.joinpath("train").mkdir(parents=True, exist_ok=False)
    plot_dir.joinpath("val").mkdir(parents=True, exist_ok=False)
    return run_dir, plot_dir
