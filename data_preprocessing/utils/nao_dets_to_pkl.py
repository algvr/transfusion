import logging
import numpy as np
import pandas as pd

from data_preprocessing.utils.path_utils import (
    data_roots,
    get_path_to_actor,
    read_detections_csv,
    read_detections_pkl,
)


def compare_dets(dataset):
    data_root = data_roots[dataset]

    actors = ["P01"]
    # actors = get_actors(data_root, dataset, "all")

    concat_dfs = []
    for actor in actors:
        logging.info(f"Working on {actor=}")
        actor_path = get_path_to_actor(data_root, dataset, actor)
        actor_nao_paths = actor_path.rglob(f"{actor}*_nao.csv")

        for path in actor_nao_paths:
            if path.stem != f"{actor}_nao.csv":
                nao_df = read_detections_csv(path)
                concat_dfs.append(nao_df)

        concat_df = pd.concat(concat_dfs)

        full_df = read_detections_csv(actor_path.joinpath(f"{actor}_nao.csv"))
        full_df_pkl = read_detections_pkl(actor_path.joinpath(f"{actor}_nao.pkl"))


if __name__ == "__main__":
    dataset = "epic"
    # nao_dets_to_pkl(dataset)
    compare_dets(dataset)
