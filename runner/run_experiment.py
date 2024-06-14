

import argparse
from datetime import datetime
import getpass
import logging
from matplotlib import pyplot as plt
import os
from os.path import abspath, basename, dirname, expandvars, isdir, isfile, join, realpath
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import resource
import sys
import ssl
import time
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
import wandb
import warnings
import yaml

base = dirname(dirname(realpath(__file__)))
sys.path.append(base)
sys.path.append(dirname(base))
sys.path.append(join(base, "detectron2"))
sys.path.append(join(base, "detectron2", "detectron2"))
sys.path.append(join(base, "detectron2", "detectron2", "checkpoint"))
sys.path.append(join(base, "detectron2", "checkpoint"))

from data_preprocessing.utils.cfg_utils import setup_logger
from modeling.model_factory import get_fusion_model, get_model
from modeling.narration_embeds.collate_wrapper_utils import get_collate_fn
from modeling.narration_embeds.narr_pooling_layers import LEARNABLE_LM
from runner.nao.snao_data import get_train_test_split
from runner.utils.callbacks import fig2img
from runner.utils.data_transforms import set_transforms_to_dsets
from runner.utils.envyaml_wrapper import EnvYAMLWrapper
from runner.utils.factories import get_callbacks, get_datasets, get_train_module
from runner.utils.utils import make_run_dir

LANG_MODEL_FEATURE_SIZES = {
    "all-distilroberta-v1": 768,
    "all-MiniLM-L12-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "distilgpt2": 768,
    "t5-small": 512,
    "flan-t5-large": 1024,
    "flan-t5-small": 512,
    "slowfast": 2304,
}
LM_TO_TEXT_POOLING = {
    "all-distilroberta-v1": "sbert_finetune",
    "all-MiniLM-L12-v2": "sbert_finetune",
    "all-MiniLM-L6-v2": "sbert_finetune",
    "distilgpt2": "gpt2",
    "t5-small": "t5-wikihow",
    "flan-t5-small": "t5-wikihow",
    "flan-t5-large": "t5-wikihow",
}
DEBUG_BS = 10

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.multiprocessing.set_sharing_strategy("file_system")


def update_config(config, args):
    timestamp = datetime.now()
    date_time = timestamp.strftime("%m/%d/%Y, %H:%M:%S")
    config.set("date", date_time)
    config.set("debug", config["debug"] or args.debug)
    config.set("force_wandb_logging", config.get("force_wandb_logging", False))

    narr_f_config_path = config["run"]["narr_fusion"]["config"]
    narr_f_config = EnvYAMLWrapper(narr_f_config_path).yaml_config
    config["run"]["narr_fusion"].update(narr_f_config)
    config["run"]["resumed_from"] = args.resume_from or ""
    config["run"]["resumed_from_name"] = args.resume_from_name or ""

    if config["run"]["devices"]["acc"] == "gpu":
        if args.gpu is not None:
            config["run"]["devices"]["devices"] = args.gpu

    # seting embed sizes based on different hyperparams
    run_args = config["run"]
    config["run"]["narration_embeds"]["args"]["text_pooling"] = LM_TO_TEXT_POOLING[
        run_args["narration_embeds"]["args"]["model_v"]
    ]
    if config["run"]["narration_embeds"].get("slowfast_f", False):
        config["run"]["narration_embeds"]["args"]["text_pooling"] = "slowfast"
        config["run"]["narration_embeds"]["args"]["model_v"] = "slowfast"

    if (
        config["run"]["narration_embeds"]["args"]["pooling"] == "sbert"
        or config["run"]["narration_embeds"]["args"]["text_pooling"] in LEARNABLE_LM
    ):

        if config["run"]["narration_embeds"]["args"]["out_mlp"]:
            run_args["narr_fusion"]["args"]["input_f_size"] = config["run"]["narration_embeds"]["args"]["out_mlp"]
            config["run"]["narration_embeds"]["args"]["size"] = LANG_MODEL_FEATURE_SIZES[
                run_args["narration_embeds"]["args"]["model_v"]
            ]
        else:

            run_args["narr_fusion"]["args"]["input_f_size"] = LANG_MODEL_FEATURE_SIZES[
                run_args["narration_embeds"]["args"]["model_v"]
            ]

            config["run"]["narration_embeds"]["args"]["size"] = LANG_MODEL_FEATURE_SIZES[
                run_args["narration_embeds"]["args"]["model_v"]
            ]

        # if we have cross_f_shared model, we take the token embedding
        if config["run"]["narr_fusion"]["model"] == "cross_f_shared":
            run_args["narr_fusion"]["args"]["back_to_img_fn"] = "token"

        if config["run"]["criterion"].get("multivar_n", 0):
            run_args["narr_fusion"]["args"]["back_to_img_fn"] = "token"

    else:
        run_args["narr_fusion"]["args"]["input_f_size"] = config["run"]["narration_embeds"]["args"]["size"]

    # needed to enable running, cannot finetune both
    if config["run"]["narration_embeds"]["args"]["text_pooling"] in LEARNABLE_LM:
        run_args["narration_embeds"]["args"]["finetune"] = False

    # if we use kl div we need a heatmap that resembles a distribution
    if config["run"]["criterion"].get("kl_div", 0):
        run_args["hmap_scaling"] = "sum"
        run_args["criterion"]["agg"] = "sum"
        run_args["heatmap_type"] = "gaussian_dist"

    # for mutlivar we also need a distribution like target
    if config["run"]["criterion"].get("multivar_n", 0):
        run_args["heatmap_type"] = "gaussian_dist"
        run_args["metric_norm"] = True
    if config["run"]["criterion"].get("mae", 0):
        run_args["heatmap_type"] = "gaussian"

    run_args["resize_spec"] = config["aug"]["resize_spec"]
    run_args["channel_order"] = config["aug"].get("channel_order", "RGB")
    run_args["dataset"] = config["dataset"]["name"]

    model_config_path = config["model"]
    model_config = get_model_config_from_path(model_config_path, config)
    model_config["verb_classifier"]["hand_args"] = run_args.get("hand_args", {"use": False})
    config.set("model", model_config)

    run_args["experiment"] = config["experiment"]
    config.set("run", run_args)

    files_to_log = {
        "config": args.config,
        "narr_fusion_config": narr_f_config_path,
        "model_config": model_config_path,
        **{
            f"external_narrs_{idx}": path
            for idx, path in enumerate(config["dataset"]["args"].get("narr_external_paths", []))
        },
    }
    return config, files_to_log


def get_model_config_from_path(model_config_path, other_config):
    model_config = EnvYAMLWrapper(model_config_path).yaml_config
    if "pretrained" not in model_config:
        model_config["pretrained"] = other_config["pretrained"]
    if "finetune" not in model_config:
        model_config["finetune"] = other_config["finetune"]

    if other_config["experiment"] in {"snao", "vsnao"}:
        if "head_k_size" not in model_config:
            model_config["head_k_size"] = other_config["head_k_size"]
        if "hmap_head_upscale" not in model_config:
            model_config["hmap_head_upscale"] = other_config["hmap_head_upscale"]

    try:
        model_config["init_0"] = other_config["init_0"]
    except:
        pass
    try:
        model_config["keep_blocks"] = other_config["keep_blocks"]
    except:
        pass

    return model_config


def get_resume_run_info(args):
    api = wandb.Api()

    init_step = args.initial_step if "initial_step" in dir(args) else None

    if (
        "." in args.resume_from
        and not args.resume_from.lower().startswith("http:")
        and not args.resume_from.lower().startswith("https:")
    ):
        # local checkpoint
        config = EnvYAMLWrapper(args.config)
        return config, args.resume_from, basename(args.resume_from).rsplit(".", 1)[0], init_step or 0

    # else: resume from wandb

    have_v = len(args.resume_from.split(":")) > 1
    run_id = args.resume_from.split(":")[0]

    if have_v:
        model_v = args.resume_from.split(":")[-1]
    else:
        model_v = "latest"

    if run_id.lower().startswith("http:") or run_id.lower().startswith("https:"):
        run_id = list(filter(str.__len__, run_id.split("/")))[-1]
    if "-" in run_id:
        raise ValueError("Please input the run ID or URL instead of the run name")

    run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{run_id}")  #f"supremap/transfusion/{run_id}")
    if init_step is None:
        init_step = run.lastHistoryStep

    config = api.artifact(f"{args.wandb_entity}/{args.wandb_project}/{run.name}_config:latest")
    model_config = api.artifact(f"{args.wandb_entity}/{args.wandb_project}/{run.name}_model_config:latest")
    narr_fusion_config = api.artifact(f"{args.wandb_entity}/{args.wandb_project}/{run.name}_narr_fusion_config:latest")
    model_checkpoint = api.artifact(f"{args.wandb_entity}/{args.wandb_project}/model-{run_id}:{model_v}")

    # create new dir to download this run into

    target_dir = join(base, "checkpoints", run_id)
    os.makedirs(target_dir, exist_ok=True)

    dl_paths = {}
    model_ckpt_name = f"model_checkpoint:{model_v}"
    for artifact_name, artifact in {
        "config": config,
        "model_config": model_config,
        "narr_fusion_config": narr_fusion_config,
        model_ckpt_name: model_checkpoint,
    }.items():
        artifact_target_path = join(target_dir, artifact_name)
        if not isdir(artifact_target_path):
            artifact.download(root=artifact_target_path)
        for root, _, files in os.walk(artifact_target_path):
            for file in files:
                dl_paths[artifact_name] = join(artifact_target_path, file)
                break

    # patch the config so that the right files get loaded
    config = EnvYAMLWrapper(dl_paths["config"])
    config.set("model", dl_paths["model_config"])
    config["run"]["narr_fusion"]["config"] = dl_paths["narr_fusion_config"]
    return config, dl_paths[model_ckpt_name], run.name, init_step


if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
    os.environ["WANDB_CACHE_DIR"] = expandvars("${CODE}/cache/wandb/")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--config",
        type=str,
        help="config file to run the program",
        default="./nao/configs/ego_nao_res50_ego4dv2.yml",
    )
    parser.add_argument(
        "--gpu", type=int, action="append", default=None, help="GPUs to use (omit to use setting in config file)"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Add to run with reduced number of samples")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="URL of wandb run or path to local checkpoint from which to resume training",
    )
    parser.add_argument("--run-test", action="store_true", default=False)
    parser.add_argument("--run-val", action="store_true", default=False)
    parser.add_argument("--do-not-log-checkpoints", action="store_true", default=False)
    parser.add_argument("--wandb-entity", default="algvr")
    parser.add_argument("--wandb-project", default="transfusion")
    parser.add_argument("--skip-ssl-verification", action="store_true")
    args = parser.parse_args()

    if args.do_not_log_checkpoints:
        print("\n!!! Warning: No checkpoints will be logged for this run! !!!\n")
        time.sleep(5)

    if args.resume_from not in [None, ""]:
        config, model_checkpoint_path, resume_from_name, init_step = get_resume_run_info(args)
        setattr(args, "resume_from_name", resume_from_name)
    else:
        config, model_checkpoint_path, resume_from_name, init_step = EnvYAMLWrapper(args.config), None, "", 0
        setattr(args, "resume_from_name", "")

    if args.run_test:
        config["run"]["run_test"] = args.run_test

    if args.run_val:
        config["run"]["run_val"] = args.run_val

    config, files_to_log = update_config(config, args)
    if config["no_threads"] > 0:
        torch.set_num_threads(config["no_threads"])
    
    if args.skip_ssl_verification:
        print("WARNING: skipping SSL verification for all requests")
        time.sleep(2)
        ssl._create_default_https_context = ssl._create_unverified_context

    wandb_run = wandb.init(
        project=args.wandb_project,
        config=config.yaml_config,
        entity=args.wandb_entity,
        mode="disabled" if (config["debug"] and not config["force_wandb_logging"]) else None,
        dir=Path(expandvars("$RUNS")),
        save_code=True,
    )
    wandb.run.log_code(".")

    for log_name, log_path in files_to_log.items():
        try:
            wandb.run.log_artifact(abspath(log_path), name=f"{wandb.run.name}_{log_name}", type="yaml")
        except ValueError:
            print(f"{log_path=} not found")
            raise ValueError

    run_cfg = config["run"]
    train_bs = run_cfg["train_bs"]
    val_bs = run_cfg["val_bs"]
    run_test = run_cfg["run_test"]
    run_val = run_cfg.get("run_val", False) or config["split"].get("all_samples_as_val", False)
    experiment = config["experiment"]
    run_dir, plot_dir = make_run_dir(wandb.run.name, experiment)
    setup_logger(run_dir)
    logging.warning(f"Run directory {run_dir}")
    with open(run_dir.joinpath("config.yml"), "w") as fp:
        yaml.dump(config.yaml_config, fp, default_flow_style=False)
    wandb_logger = WandbLogger(log_model=False if args.do_not_log_checkpoints else "all", save_dir=run_dir)

    raw_dataset = get_datasets(config["experiment"])(config)

    no_classes = {"noun": raw_dataset.get_no_nouns(), "verb": raw_dataset.get_no_verbs()}
    logging.info(f"Working on {raw_dataset.get_no_nouns()} noun classes and {raw_dataset.get_no_verbs()} verb classes")

    train_dataset, val_dataset, test_dataset, train_test_split = get_train_test_split(raw_dataset, config)
    if config["debug"]:
        train_dataset.nao_annots = train_dataset.nao_annots.iloc[:2000]

    train_test_split.to_csv(f"{run_dir}/split.csv")
    artifact = wandb.Artifact(f"{wandb.run.name}_train_test_split", type="split")
    artifact.add_file(f"{run_dir}/split.csv")
    wandb_run.log_artifact(artifact)

    class_infos = {
        "weights": {
            "noun": train_dataset.get_b_class_weights("noun"),
            "verb": train_dataset.get_b_class_weights("verb")
        },
        "no_classes": no_classes,
    }

    set_transforms_to_dsets(
        train_dataset,
        [val_dataset, test_dataset],
        run_cfg,
        config["aug"],
        config["experiment"],
    )
    collate_fn = get_collate_fn(config["experiment"], run_cfg["narration_embeds"], raw_dataset)

    train_bs = train_bs // len(run_cfg["devices"]["devices"])
    val_bs = val_bs // len(run_cfg["devices"]["devices"])

    train_loader = DataLoader(
        train_dataset,
        train_bs if not config["debug"] else DEBUG_BS,
        shuffle=True,
        num_workers=config["no_workers"],
        pin_memory=False,
        drop_last=args.debug,
        collate_fn=collate_fn,
        prefetch_factor=3 if config["no_workers"] else None,
    )
    val_loader = DataLoader(
        val_dataset,
        val_bs if not config["debug"] else DEBUG_BS,
        shuffle=False,
        num_workers=config["no_workers"] or 0,
        pin_memory=False,
        collate_fn=collate_fn,
        prefetch_factor=1 if config["no_workers"] else None,
        drop_last=args.debug
    )
    logging.info(f"Sample count for train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    model_cfg = config["model"]
    model = get_model(config["experiment"], model_cfg, run_cfg, class_infos["no_classes"],
                      train_noun_verb_frequencies=train_dataset.get_noun_verb_frequencies())
    model = get_fusion_model(model, model_cfg, run_cfg, class_infos["no_classes"])

    module = get_train_module(config["experiment"])(run_cfg, class_infos, model)
    graph = wandb_logger.watch(model, log="all", log_graph=True, log_freq=50)
    callbacks_fn = get_callbacks(config["experiment"])
    callbacks = callbacks_fn(
        train_dataset.get_samples(20, run_cfg["seed"], train_loader.collate_fn),
        val_dataset.get_samples(20, run_cfg["seed"], train_loader.collate_fn),
        plot_dir,
        run_dir,
        model_cfg["type"],
        run_cfg,
        raw_dataset.get_noun_mapping(),
        raw_dataset.get_verb_mapping(),
    )

    if run_cfg.get("replace_heads", False) == "all" and model_checkpoint_path:
        checkpoint = torch.load(model_checkpoint_path)
        
        checkpoint["state_dict"]["noun_criterion.weight"] = module.state_dict()["noun_criterion.weight"]
        checkpoint["state_dict"]["verb_criterion.weight"] = module.state_dict()["verb_criterion.weight"]
        
        # replace the heads for finetuning
        to_replace = []
        for k in checkpoint["state_dict"]:
            if "classifier" in k or "box_regressor" in k:
                to_replace.append(k)
        for k in to_replace:
            checkpoint["state_dict"][k] = module.state_dict()[k]
        print(f"Replaced {to_replace} weights")
        print("---------------------------------------------")
        print(f"Proceed with finetuning")

        module.load_state_dict(checkpoint["state_dict"])
        model_checkpoint_path = None

    trainer = pl.Trainer(
        devices=run_cfg["devices"]["devices"],
        accelerator=run_cfg["devices"]["acc"],
        max_epochs=run_cfg["epochs"],
        logger=wandb_logger,
        val_check_interval=run_cfg["val_every"],
        auto_lr_find=True,
        accumulate_grad_batches=run_cfg["accumulate_grad_batches"],
        gradient_clip_val=run_cfg["grad_clip"],
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        enable_checkpointing=run_cfg["save_every"] > 0,
        num_sanity_val_steps=5 if not run_val and not config["split"].get("all_samples_as_val", False) else -1,
        precision=run_cfg["precision"],
        resume_from_checkpoint=model_checkpoint_path or None,
        strategy="ddp" if len(run_cfg["devices"]["devices"]) > 1 else None,
        num_nodes=1,
    )

    if run_cfg["tune_lr"] and not config["debug"]:
        res = trainer.tune(module, train_dataloaders=train_loader)["lr_find"]
        fig = res.plot(suggest=True, show=False)
        fig.suptitle(f"Learning rate:{res.suggestion()}")
        plt.show()
        plt.savefig(f"{run_dir}/lr_finder.jpg", pad_inches=0, bbox_inches="tight")
        wandb.log({"lr_finder": wandb.Image(fig2img(fig))})

    if not run_test:
        res = trainer.fit(
            module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    else:
        logging.info("Running model on test set")

        test_loader = DataLoader(
            test_dataset,
            val_bs if not config["debug"] else DEBUG_BS,
            shuffle=False,
            num_workers=config["no_workers"],
            pin_memory=False,
            collate_fn=collate_fn,
            prefetch_factor=1 if config["no_workers"] else None,
            drop_last=False,
            # multiprocessing_context="spawn" if config["no_workers"] else None,
        )
        res = trainer.validate(
            module,
            val_dataloaders=test_loader,
            ckpt_path=model_checkpoint_path,
            verbose=True,
        )
        print(res)
