from data_preprocessing.datasets.egonao_datasets import EgoNaoDataset
from data_preprocessing.datasets.snao_datasets import MergedNaoDataset
from data_preprocessing.utils.path_utils import data_roots
from modeling.hand_pos_dataset import HandPosDataset
from runner.nao.snao_data import get_narr_dataset_wrapper
from runner.utils.utils import get_datasets_from_name, get_label_merging, ALL_ACTORS, DEBUG_ACTORS


def get_egonao_datasets(config):
    dataset_cfg = config["dataset"]
    dataset_args = dataset_cfg["args"]
    run_args = config["run"]
    debug = config["debug"]
    label_merging = get_label_merging(dataset_args["label_merging"])
    hand_args = run_args.get("hand_args", {"use": False})

    if debug:
        actors = DEBUG_ACTORS
    else:
        actors = ALL_ACTORS

    dataset_names = get_datasets_from_name(dataset_cfg["name"])
    if len(dataset_names) > 1:
        datasets = {}
        for dataset_name in dataset_names:
            dataset = EgoNaoDataset(
                root_data_path=data_roots[dataset_name],
                subset=None,
                offset_s=dataset_args["offset_s"],
                actors=actors[dataset_name],
                source=dataset_name,
                label_merging=label_merging[dataset_name],
                label_cutoff=dataset_args["label_cutoff"],
                nao_version=dataset_args["nao_version"],
                coarse=dataset_args["coarse"],
                take_double=dataset_args["take_double"],
                narr_structure=dataset_args.get("narr_structure", "{gt_narr}"),
                narr_external_paths=dataset_args.get("narr_external_paths", []),
                use_external_label_mapping=dataset_args.get("use_external_label_mapping", False),
                verb_bg=run_args.get("verb_bg", False),
                hand_cache_path=dataset_args.get("hand_cache_path")
            )
            datasets[dataset_name] = dataset

        for d_name in datasets.keys():
            datasets[d_name] = get_narr_dataset_wrapper(datasets[d_name], [datasets[d_name]], run_args, d_name)[0]

            if hand_args["use"]:
                hand_args["device"] = "cpu" if run_args["devices"]["acc"] == "cpu" else f"cuda:{run_args['devices']['devices'][0]}"
                datasets[d_name] = HandPosDataset(datasets[d_name], hand_args)

        raw_nao_dataset = MergedNaoDataset(datasets)
    else:  # if we load just 1 dataset
        dset_type = dataset_cfg["name"]
        dataset_args["label_merging"] = label_merging[dset_type]
        root_data_path = data_roots[dset_type]
        raw_nao_dataset = EgoNaoDataset(
            root_data_path,
            subset=None,
            offset_s=dataset_args["offset_s"],
            actors=actors[dset_type],
            source=dset_type,
            label_merging=label_merging[dset_type],
            label_cutoff=dataset_args["label_cutoff"],
            nao_version=dataset_args["nao_version"],
            take_double=dataset_args["take_double"],
            coarse=dataset_args["coarse"],
            narr_structure=dataset_args.get("narr_structure", "{gt_narr}"),
            narr_external_paths=dataset_args.get("narr_external_paths", []),
            use_external_label_mapping=dataset_args.get("use_external_label_mapping", False),
            verb_bg=run_args.get("verb_bg", False),
        )
        raw_nao_dataset = get_narr_dataset_wrapper(raw_nao_dataset, [raw_nao_dataset], run_args, dset_type)[0]

        if hand_args["use"]:
            hand_args["device"] = "cpu" if run_args["devices"]["acc"] == "cpu" else f"cuda:{run_args['devices']['devices'][0]}"
            raw_nao_dataset = HandPosDataset(raw_nao_dataset, hand_args)

    return raw_nao_dataset
