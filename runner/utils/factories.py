from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torchvision import transforms
from torchvision.transforms.transforms import ToPILImage

from runner.nao.ego_nao_data import get_egonao_datasets
from runner.nao.ego_nao_trainer import EgoNAOTrainer
from runner.utils.callbacks import BboxPlotterCallback
from runner.utils.data_transforms import get_denormalize, get_norm_mean_std


def get_datasets(experiment):
    if experiment == "egonao":
        return get_egonao_datasets
    raise NotImplementedError()


def get_train_module(experiment):
    if experiment == "egonao":
        return EgoNAOTrainer
    raise NotImplementedError()


monitor_modes = {"heatmap_acc_val": "min", "noun_val_acc": "max", "verb_val_acc": "max"}


def get_callbacks(experiment):
    if experiment in {"ego_nao", "egonao"}:
        checkpoint_callbacks = get_callbacks_for_ego
    else:
        raise NotImplementedError()
    return checkpoint_callbacks


def get_callbacks_for_ego(train_samples, val_samples, plot_dir, run_dir, model_type, run_cfg, noun_mapping, verb_mapping):
    criterion_dict = run_cfg["criterion"]
    if criterion_dict["bbox"] and criterion_dict["noun"] and criterion_dict["verb"] and criterion_dict["ttc"]:
        m_checkpoint = ModelCheckpoint(
                monitor="map_box_noun_verb_ttc_val",
                mode="max",
                dirpath=run_dir,
                save_top_k=-1,
                filename=f"{model_type}-" + "{epoch:02d}"+"_{map_box_noun_verb_ttc_val:.2f}",
            )
    elif criterion_dict["bbox"] and criterion_dict["noun"] and criterion_dict["verb"]:
        m_checkpoint = ModelCheckpoint(
                monitor="map_box_noun_verb_val",
                mode="max",
                dirpath=run_dir,
                save_top_k=-1,
                filename=f"{model_type}-" + "{epoch:02d}"+"_{map_box_noun_verb_val:.2f}",
            )
    elif criterion_dict["bbox"] and criterion_dict["noun"]:
        m_checkpoint = ModelCheckpoint(
                monitor="map_box_noun_val",
                mode="max",
                dirpath=run_dir,
                save_top_k=-1,
                filename=f"{model_type}-" + "{epoch:02d}"+"_{map_box_noun_val:.2f}",
            )
    else:
        raise NotImplementedError()

    callbacks = [m_checkpoint]

    mean, std = get_norm_mean_std(run_cfg["normalization"], run_cfg["dataset"])
    inverse_img = transforms.Compose([get_denormalize(mean, std), ToPILImage(mode="RGB")])
    plot_callback = BboxPlotterCallback(plot_dir, train_samples, val_samples, inverse_img,noun_mapping, verb_mapping, callback_per=1, 
                add_verbs=run_cfg["criterion"]["verb"],
                add_ttcs=run_cfg["criterion"]["ttc"]
                )
    callbacks.append(plot_callback)

    return callbacks
