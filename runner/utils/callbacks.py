from copy import copy
import io

import numpy as np
import PIL
import PIL.Image
import torch
import wandb
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pytorch_lightning.callbacks import Callback
from textwrap import wrap
from matplotlib.patches import Rectangle

matplotlib.use("Agg")


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        pad_inches=0,
        bbox_inches="tight",
    )
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def get_countour(heatmap, tol=1e-2):
    cont = torch.where(
        heatmap - heatmap.std(dim=[-1, -2], unbiased=False, keepdim=True) < tol,
        1,
        0,
    )
    return cont.squeeze().cpu().numpy()


def get_rectangles_for_many(bboxes, edgecolor, linewidth=2):
    """A list of list of bboxes. The first entry is the sample idx, each sample can have multiple boxes."""
    rectangles = []
    for annot_boxes in bboxes:
        annot_rectangles = get_rectangles_for_one(annot_boxes, edgecolor, linewidth)

        rectangles.append(annot_rectangles)

    return rectangles


def get_rectangles_for_one(bboxes, edgecolor, linewidth=2):
    annot_rectangles = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        rectangle = Rectangle(
            xy=(x1, y1), width=(x2 - x1), height=(y2 - y1), linewidth=linewidth, edgecolor=edgecolor, facecolor="none"
        )
        annot_rectangles.append(rectangle)

    return annot_rectangles


def dict_to_device(dicty, device):
    for key, value in dicty.items():
        try:
            if isinstance(value, list):
                dicty[key] = [item.to(device) for item in value]
            else:
                dicty[key] = value.to(device)
        except AttributeError:
            continue
    return dicty


class HmapPlotterCallback(Callback):
    def __init__(
        self,
        plot_dir,
        train_samples,
        val_samples,
        inverse_img,
        inverse_hmap,
        noun_mapping,
        verb_mapping,
        callback_per=2,
        tol=1e-2,
        pred_key="heatmap",
    ):
        self.plot_dir = plot_dir
        self.callback_per = callback_per
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.title_cols = ["nao_narration", "nao_clip_id", "frame", "eta"]
        self.train_titles = self.get_titles(train_samples, self.title_cols)
        self.val_titles = self.get_titles(val_samples, self.title_cols)
        self.tol = tol
        self.pred_key = pred_key
        self.noun_mapping = noun_mapping
        self.verb_mapping = verb_mapping
        self.idx_to_noun_mapping = {v: k for k, v in self.noun_mapping.items()}
        self.idx_to_verb_mapping = {v: k for k, v in self.verb_mapping.items()}

        self.setup_gts(
            self.train_samples["inputs"],
            self.val_samples["inputs"],
        )

        self.inverse_img = inverse_img
        self.inverse_hmap = inverse_hmap

        self.plot_array(
            self.train_samples["inputs"]["image"],
            self.train_contours,
            self.train_samples["inputs"][self.pred_key],
            "train",
            0,
            self.train_titles,
            desc="gt_images",
        )
        self.plot_array(
            self.val_samples["inputs"]["image"],
            self.val_contours,
            self.val_samples["inputs"][self.pred_key],
            "val",
            0,
            self.val_titles,
            desc="gt_images",
        )

        train_table = wandb.Table(
            columns=self.title_cols, data=np.array([self.train_samples[k].tolist() for k in self.title_cols]).T
        )
        val_table = wandb.Table(
            columns=self.title_cols, data=np.array([self.val_samples[k].tolist() for k in self.title_cols]).T
        )
        wandb.log({"train_samples": train_table})
        wandb.log({"val_samples": val_table})

    def setup_gts(self, train_gts, val_gts):
        self.train_contours = get_countour(train_gts[self.pred_key], self.tol)
        self.val_contours = get_countour(val_gts[self.pred_key], self.tol)

    def get_titles(self, samples, title_cols):
        entries = samples[title_cols[0]]
        titles = np.char.array(["" for _ in entries])
        for title_col in title_cols[::-1]:
            if title_col == "eta":
                samples[title_col] = samples[title_col].apply(lambda x: round(x, 3))

            titles = title_col + ":" + np.char.array(samples[title_col].values.astype(str)) + " " + titles
        return titles

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.callback_per == 0:
            pl_module.eval()
            with torch.no_grad():
                inputs = dict_to_device(self.train_samples["inputs"], pl_module.device)
                outs = pl_module(inputs)
                pred_hmaps = outs["heatmap"].to("cpu")
                if pl_module.prob_mode:
                    pred_hmaps = pred_hmaps > 0

                if pl_module.ttc_pred_on:
                    ttcs = outs["ttc"].to("cpu").numpy()
                else:
                    ttcs = None

                self.plot_array(
                    self.train_samples["inputs"]["image"],
                    self.train_contours,
                    pred_hmaps,
                    "train",
                    pl_module.current_epoch + 1,
                    self.train_titles,
                    "pred_images",
                    ttcs=ttcs,
                )

                inputs = dict_to_device(self.val_samples["inputs"], pl_module.device)
                outs = pl_module(inputs)
                pred_hmaps = outs["heatmap"].to("cpu")

                if pl_module.prob_mode:
                    pred_hmaps = pred_hmaps > 0

                if pl_module.ttc_pred_on:
                    ttcs = outs["ttc"].to("cpu").numpy()

                self.plot_array(
                    self.val_samples["inputs"]["image"],
                    self.val_contours,
                    pred_hmaps,
                    "val",
                    pl_module.current_epoch + 1,
                    self.val_titles,
                    "pred_images",
                    ttcs=ttcs,
                )

    def plot_array(self, imgs, contours, hmaps, subset, epoch, titles, desc="gt_images", ncols=5, ttcs=None):
        wandb_imgs = []
        fig = plt.figure(figsize=(20, 16))
        grid = ImageGrid(fig, 111, nrows_ncols=(len(imgs) // ncols, ncols), axes_pad=(0.1, 0.6))

        for i in range(len(imgs)):
            img = self.inverse_img(imgs[i])
            mapped_hmap = self.inverse_hmap(hmaps[i])
            grid[i].imshow(img)
            grid[i].contour(contours[i], cmap="Oranges")
            grid[i].imshow(mapped_hmap, alpha=0.4)
            grid[i].set_title("\n".join(wrap(titles[i], 35)), fontsize=11)
            if ttcs:
                grid[i].text(0, 20, f"PRed TTC: {ttcs[i]}", color="white", fontsize=12)

        plt.show()
        plt.savefig(
            f"{self.plot_dir}/{subset}/{desc}_{epoch}.jpg",
            pad_inches=0,
            bbox_inches="tight",
        )
        to_show = fig2img(fig)
        plt.close()
        wandb_imgs.append(wandb.Image(to_show))

        wandb.log({f"{desc}_{subset}": wandb_imgs})


def get_list_from_outputs(outputs, key, top_k=None):
    if top_k:
        bboxes = [output[key][:top_k].cpu().numpy() for output in outputs]
    else:
        bboxes = [output[key].cpu().numpy() for output in outputs]
    return bboxes


class BboxPlotterCallback(HmapPlotterCallback):
    def __init__(
        self,
        plot_dir,
        train_samples,
        val_samples,
        inverse_img,
        noun_mapping,
        verb_mapping,
        inverse_hmap=lambda x: x,
        callback_per=2,
        tol=0.01,
        add_verbs=False,
        add_ttcs=False,
    ):
        self.add_verbs = add_verbs
        self.add_ttcs = add_ttcs
        self.title_cols = ["nao_narration", "nao_clip_id", "eta"]

        super().__init__(
            plot_dir,
            train_samples,
            val_samples,
            inverse_img,
            inverse_hmap,
            noun_mapping,
            verb_mapping,
            callback_per,
            tol,
            pred_key="targets",
        )

    def setup_gts(self, train_gts, val_gts):
        train_bboxes = get_list_from_outputs(train_gts["targets"], key="boxes")
        self.train_contours = get_rectangles_for_many(train_bboxes, edgecolor="green", linewidth=3)

        val_bboxes = get_list_from_outputs(val_gts["targets"], key="boxes")
        self.val_contours = get_rectangles_for_many(val_bboxes, edgecolor="green", linewidth=3)

    def plot_array(
        self, imgs, rectangles, preds, subset, epoch, titles, desc="gt_images", ncols=5, ttcs=None, language_f=None
    ):
        wandb_imgs = []
        fig = plt.figure(figsize=(20, 16))
        grid = ImageGrid(fig, 111, nrows_ncols=(len(imgs) // ncols, ncols), axes_pad=(0.25, 0.75))

        pred_boxes = get_list_from_outputs(preds, key="boxes", top_k=5)
        try:
            pred_nouns = get_list_from_outputs(preds, key="labels", top_k=5)
        except:
            pred_nouns = get_list_from_outputs(preds, key="nouns", top_k=5)
        if self.add_verbs:
            pred_verbs = get_list_from_outputs(preds, key="verbs", top_k=5)
        if self.add_ttcs:
            pred_ttcs = get_list_from_outputs(preds, key="ttcs", top_k=5)
        pred_rectangles = get_rectangles_for_many(pred_boxes, edgecolor="red")

        # i goes over the sample idx, usually 20 samples
        for i in range(len(imgs)):
            img = self.inverse_img(imgs[i])
            grid[i].imshow(img)
            for rectangle in rectangles[i]:
                grid[i].add_patch(copy(rectangle))

            pred_box = pred_boxes[i]
            # j goes over the predictions for each sample, where we might have multiple bboxes with their labels
            for j, rectangle in enumerate(pred_rectangles[i]):
                grid[i].add_patch(copy(rectangle))
                predy_nouns = self.idx_to_noun_mapping[pred_nouns[i][j]]
                grid[i].text(pred_box[j][0], pred_box[j][1] + 5, predy_nouns, color="white", fontsize=9)
                if self.add_verbs:
                    predy_verbs = self.idx_to_verb_mapping[pred_verbs[i][j]]
                    grid[i].text(pred_box[j][0], pred_box[j][1] - 10, predy_verbs, color="white", fontsize=9)
                if self.add_ttcs:
                    grid[i].text(
                        pred_box[j][0] + 7, pred_box[j][1] + 5, f"TTC: {pred_ttcs[i][j]:.2f}s", color="white", fontsize=8
                    )

            if language_f:
                if  type(language_f[i]) == str:
                    grid[i].text(0, 20, "\n".join(wrap(language_f[i], 40)), size=9, color="red")
            grid[i].set_title("\n".join(wrap(titles[i], 35)), fontsize=10)

        plt.show()
        plt.savefig(
            f"{self.plot_dir}/{subset}/{desc}_{epoch}.jpg",
            pad_inches=0,
            bbox_inches="tight",
        )
        to_show = fig2img(fig)
        plt.close()
        wandb_imgs.append(wandb.Image(to_show))
        wandb.log({f"{desc}_{subset}": wandb_imgs})

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.current_epoch == 0:
            pl_module.eval()
            with torch.no_grad():
                inputs = dict_to_device(self.train_samples["inputs"], pl_module.device)
                #try:
                outs = pl_module.model.forward_w_dets(inputs)
                #except:
                #    outs = pl_module.model(inputs)["detections"]

                self.plot_array(
                    imgs=self.train_samples["inputs"]["image"],
                    rectangles=self.train_contours,
                    preds=outs,
                    subset="train",
                    epoch="before_train",
                    titles=self.train_titles,
                    desc="pred_images",
                    ttcs=None,
                    language_f=self.train_samples["inputs"].get("language_f", None),
                )

                inputs = dict_to_device(self.val_samples["inputs"], pl_module.device)
                try:
                    outs = pl_module.model.forward_w_dets(inputs)
                except:
                    outs = pl_module.model(inputs)["detections"]

                self.plot_array(
                    imgs=self.val_samples["inputs"]["image"],
                    rectangles=self.val_contours,
                    preds=outs,
                    subset="val",
                    epoch="before_train",
                    titles=self.val_titles,
                    desc="pred_images",
                    ttcs=None,
                    language_f=self.val_samples["inputs"].get("language_f", None),
                )

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.callback_per == 0:
            pl_module.eval()
            with torch.no_grad():
                inputs = dict_to_device(self.train_samples["inputs"], pl_module.device)
                try:
                    outs = pl_module.model.forward_w_dets(inputs)
                except:
                    outs = pl_module.model(inputs)["detections"]

                self.plot_array(
                    imgs=self.train_samples["inputs"]["image"],
                    rectangles=self.train_contours,
                    preds=outs,
                    subset="train",
                    epoch=pl_module.current_epoch + 1,
                    titles=self.train_titles,
                    desc="pred_images",
                    ttcs=None,
                    language_f=self.train_samples["inputs"].get("language_f", None),
                )

                inputs = dict_to_device(self.val_samples["inputs"], pl_module.device)
                try:
                    outs = pl_module.model.forward_w_dets(inputs)
                except:
                    outs = pl_module.model(inputs)["detections"]

                self.plot_array(
                    imgs=self.val_samples["inputs"]["image"],
                    rectangles=self.val_contours,
                    preds=outs,
                    subset="val",
                    epoch=pl_module.current_epoch + 1,
                    titles=self.val_titles,
                    desc="pred_images",
                    ttcs=None,
                    language_f=self.val_samples["inputs"].get("language_f", None),
                )
