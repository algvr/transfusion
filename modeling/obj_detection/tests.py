from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from modeling.obj_detection.wrapper_utils import *
from modeling.obj_detection.wrapper_utils import is_torch_18v
from PIL import Image
from torchvision.models.detection import *
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.utils import draw_bounding_boxes

from data_preprocessing.datasets.egonao_datasets import EgoNaoDataset
from torchvision import transforms

import torch.nn.functional as Fnn

from data_preprocessing.utils.path_utils import data_roots
from runner.utils.utils import DEBUG_ACTORS
from runner.utils.data_transforms import IMNET_MEAN, IMNET_STD, get_snao_resize_transforms
from modeling.obj_detection.dual_stream_rcnn_wrapper import get_rcnn_model, rcnn_dict
from runner.utils.callbacks import get_rectangles_for_many

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == "__main__":
    rcnn_type = "mobilenet_320"
    bs = 3
    
    try:
        image_path = Path("/local/home/rpasca/Thesis/Ego4d_all/forecast/tools/short_term_anticipation/data/object_frames")
        images = list(image_path.glob("*.jpg"))
        
        img_idxs = [5]

        for i, image_p in enumerate(images):
            if image_p.name == "f7aec252-bd4f-4696-8de5-ef7b871e2194_0010203.jpg":
                img_idxs.append(i)
                break


        image_pil = [Image.open(str(images[idx])) for idx in img_idxs]

        images_raw = [torch.Tensor(np.array(image)).permute(2, 0, 1,) for image in image_pil]
        images_raw.append(Fnn.interpolate (images_raw[-1][None], size=(480,640), align_corners=True, mode="bilinear")[0])
        images = [image/255 for image in images_raw] 
            
        norm_transform = transforms.Normalize(IMNET_MEAN, IMNET_STD) 
        images_norm = [norm_transform(image) for image in images]

    except FileNotFoundError:
        images = torch.randint(0,255, (bs, 3, 480, 640))/255

    no_boxes = [11, 13, 2]
    boxes = [torch.rand(no_boxes[i], 4) for i in range(bs)]
    for box in boxes:
        box[:, 2:4] = box[:, 0:2] + box[:, 2:4]

    labelinos = [torch.randint(1, 5, (1, no_boxes[i])) for i in range(len(no_boxes)) ]

    targets = []
    for i in range(bs):
        d = {}
        d["boxes"] = boxes[i]
        d["labels"] = labelinos[i].squeeze(0)
        d["verbs"] = labelinos[i].squeeze(0)
        targets.append(d)

    my_model = get_rcnn_model(rcnn_type, True, 0, num_classes={"noun":10, "verb":10})
    base_rcnn_clzz = rcnn_dict[rcnn_type]
    base_rcnn = base_rcnn_clzz(pretrained=True, trainable_backbone_layers=0)

    with torch.no_grad():
        my_out = my_model({"image": images_norm}, targets)
        # print(my_out)
        loss_objectness, loss_rpn_box_reg = my_model.rcnn_to_wrap.rpn.compute_loss(
            my_out["proposals"]["objectness"],
            my_out["proposals"]["pred_bbox_deltas"],
            my_out["proposals"]["labels"],
            my_out["proposals"]["reg_targets"],
        )
        loss_classifier, loss_box_reg = fastrcnn_loss(
            my_out["roi_outputs"]["class_logits"],
            my_out["roi_outputs"]["box_regression"],
            my_out["roi_outputs"]["labels"],
            my_out["roi_outputs"]["reg_targets"],
        )
        base_out = base_rcnn(images, targets)
        # assert loss_objectness == base_out["loss_objectness"]
        # assert loss_rpn_box_reg == base_out["loss_rpn_box_reg"]
        # assert loss_classifier == base_out["loss_classifier"]
        # assert loss_box_reg == base_out["loss_box_reg"]

        my_model.eval()
        my_out = my_model.forward_w_dets({"image": images_norm}, targets)
        print("Forward with dets version")
        my_res = draw_bounding_boxes(images_raw[0].to(torch.uint8), my_out[0]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        plt.savefig(f"eval_my_res_boxes_{img_idxs[0]}.jpg")

        my_res = draw_bounding_boxes(images_raw[1].to(torch.uint8), my_out[1]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        plt.savefig(f"eval_my_res_boxes_{img_idxs[1]}.jpg")

        my_res = draw_bounding_boxes(images_raw[2].to(torch.uint8), my_out[2]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        plt.savefig(f"eval_my_res_boxes_{img_idxs[1]}_mizerie.jpg")

        my_model.train()
        outs = my_model.forward({"image": images_norm}, targets)
        my_out = my_model.dets_from_outs(outs)
        my_out_direct = my_model.forward_w_dets({"image": images_norm}, targets)
        
        # print("Dets version")
        assert torch.all(my_out[0]["nouns"] == my_out_direct[0]["nouns"])
        assert torch.all(my_out[0]["boxes"] == my_out_direct[0]["boxes"])
        my_res = draw_bounding_boxes(images_raw[0].to(torch.uint8), my_out[0]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        plt.savefig(f"train_my_res_boxes_{img_idxs[0]}.jpg")

        my_res = draw_bounding_boxes(images_raw[1].to(torch.uint8), my_out[1]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        plt.savefig(f"train_my_res_boxes_{img_idxs[1]}.jpg")

        my_res = draw_bounding_boxes(images_raw[2].to(torch.uint8), my_out[2]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        plt.savefig(f"train_my_res_boxes_{img_idxs[1]}_mizerie.jpg")

        base_rcnn.eval()
        base_out = base_rcnn(images, targets)
        # print(base_out[0]["labels"])
        # print(base_out[0]["scores"])
        base_res = draw_bounding_boxes(images_raw[0].to(torch.uint8), base_out[0]["boxes"], width=5)
        plt.imshow(F.to_pil_image(base_res))
        plt.savefig(f"eval_base_res_boxes_{img_idxs[0]}.jpg")

        base_res = draw_bounding_boxes(images_raw[1].to(torch.uint8), base_out[1]["boxes"], width=5)
        plt.imshow(F.to_pil_image(base_res))
        plt.savefig(f"eval_base_res_boxes_{img_idxs[1]}.jpg")

        base_res = draw_bounding_boxes(images_raw[2].to(torch.uint8), base_out[2]["boxes"], width=5)
        plt.imshow(F.to_pil_image(base_res))
        plt.savefig(f"eval_base_res_boxes_{img_idxs[1]}_mizerie.jpg")

        # assert (my_out[0]["scores"] - base_out[0]["scores"]).max() == 0
        # assert (my_out[0]["scores"] - base_out[0]["scores"]).min() == 0
        # ------------------------------------------------

        print("Testing on the datasets")

        dataset = "ego4d"
        dataset_args = {"label_cutoff":None,
        "nao_version":1, 
        "coarse":False,
        "take_double":False
        }

        dataset = EgoNaoDataset(
            root_data_path=data_roots[dataset],
            subset=None,
            offset_s=0.4,
            actors=DEBUG_ACTORS[dataset],
            source=dataset,
            label_merging={"verb":{}, "noun":{}},
            label_cutoff=dataset_args["label_cutoff"],
            nao_version=dataset_args["nao_version"],
            coarse=dataset_args["coarse"],
            take_double=dataset_args["take_double"],
    )


        no_nouns = len(dataset.noun_mapping)
        no_verbs = len(dataset.verb_mapping)
        my_model = get_rcnn_model(rcnn_type, True, 0, num_classes={"noun":no_nouns, "verb":no_verbs})

        item = dataset[0]
        targetino = [{"boxes":item["bboxes"], "labels":item["labels"]}]
        imagino = item["image"].permute(2,0,1)
        my_model.train()
        outs = my_model.forward({"image": [imagino]}, 
                                targets=targetino)
        my_out = my_model.dets_from_outs(outs)
        my_out_direct = my_model.forward_w_dets({"image": [imagino]}, targets= targetino)

        image = torch.from_numpy(dataset.get_raw_item(0)["image"].copy()).permute(2,0,1)
        my_res = draw_bounding_boxes(image.to(torch.uint8), my_out[0]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        ax = plt.gca()
        rectangles = get_rectangles_for_many([dataset.get_raw_item(0)["bboxes"]],linewidth=5, 
                                    edgecolor="green")[0]
        for rect in rectangles:
            ax.add_patch(rect,)
        plt.savefig(f"255_dataset_my_res_boxes_0.jpg")


        #Set input transform to be 0-1
        dataset.set_input_transforms(transforms.ToTensor())
        item = dataset[0]
        targetino = [{"boxes":item["bboxes"], "labels":item["labels"]}]
        imagino = item["image"]
        my_model.train()
        outs = my_model.forward({"image": [imagino]}, targets=targetino)
        my_out = my_model.dets_from_outs(outs)
        my_out_direct = my_model.forward_w_dets({"image": [imagino]}, targets= targetino)

        image = torch.from_numpy(dataset.get_raw_item(0)["image"].copy()).permute(2,0,1)
        my_res = draw_bounding_boxes(image.to(torch.uint8), my_out[0]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        ax = plt.gca()
        rectangles = get_rectangles_for_many([dataset.get_raw_item(0)["bboxes"]],linewidth=5, 
                                    edgecolor="green")[0]
        for rect in rectangles:
            ax.add_patch(rect,)
        plt.savefig(f"01_dataset_my_res_boxes_0.jpg")
        plt.close()

        #set normalization as well
        my_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ])
        dataset.set_input_transforms(my_transforms)
        item = dataset[0]
        imagino = item["image"]
        my_model.train()
        outs = my_model.forward({"image": [imagino]}, targets=targetino)
        my_out = my_model.dets_from_outs(outs)
        my_out_direct = my_model.forward_w_dets({"image": [imagino]}, targets= targetino)

        image = torch.from_numpy(dataset.get_raw_item(0)["image"].copy()).permute(2,0,1)
        my_res = draw_bounding_boxes(image.to(torch.uint8), my_out[0]["boxes"], width=5)
        ax = plt.gca()
        rectangles = get_rectangles_for_many([dataset.get_raw_item(0)["bboxes"]],linewidth=5, 
                                    edgecolor="green")[0]
        for rect in rectangles:
            ax.add_patch(rect,)
        plt.imshow(F.to_pil_image(my_res))
        plt.savefig(f"norm_dataset_my_res_boxes_0.jpg")
        plt.close()

        my_res = draw_bounding_boxes(image.to(torch.uint8), my_out_direct[0]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        ax = plt.gca()
        rectangles = get_rectangles_for_many([dataset.get_raw_item(0)["bboxes"]],linewidth=5, 
                                    edgecolor="green")[0]
        for rect in rectangles:
            ax.add_patch(rect,)
        plt.savefig(f"norm_dataset_my_res_direct_boxes_0.jpg")
        plt.close()
        
        
        print("Testing with resizing")
        resize_fn = get_snao_resize_transforms({"resize_spec":[224, 224]})
        dataset.set_resize_fn(resize_fn)

        item = dataset[0]
        targetino = [{"boxes":item["bboxes"], "labels":item["labels"]}]
        imagino = item["image"]
        my_model.train()
        outs = my_model.forward({"image": [imagino]}, targets=targetino)
        my_out = my_model.dets_from_outs(outs)
        my_out_direct = my_model.forward_w_dets({"image": [imagino]}, targets= targetino)

        image = torch.from_numpy(dataset.get_raw_item(0)["image"].copy()).permute(2,0,1)
        my_res = draw_bounding_boxes(image.to(torch.uint8), my_out[0]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        ax = plt.gca()
        rectangles =  get_rectangles_for_many([dataset.get_raw_item(0)["bboxes"]],linewidth=5, 
                                    edgecolor="green")[0]
        for rect in rectangles:
            ax.add_patch(rect,)
        plt.savefig(f"resize_norm_dataset_my_res_boxes_0.jpg")
        plt.close()

        my_res = draw_bounding_boxes(image.to(torch.uint8), my_out_direct[0]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        ax = plt.gca()
        rectangles =  get_rectangles_for_many([dataset.get_raw_item(0)["bboxes"]],linewidth=5, 
                                    edgecolor="green")[0]
        for rect in rectangles:
            ax.add_patch(rect)
        plt.savefig(f"resize_norm_dataset_my_res_direct_boxes_0.jpg")
        plt.close()

        print("Testing with dropouts")
        dataset.set_resize_fn(None)
        my_model = get_rcnn_model(rcnn_type, True, 0, 
                                num_classes={"noun":no_nouns, "verb":no_verbs},
                                box_1_dropout=0.2,
                                box_2_dropout=0.1,
                                classif_dropout=0.2)

        item = dataset[0]
        targetino = [{"boxes":item["bboxes"], "labels":item["labels"]}]
        imagino = item["image"]
        
        my_model.train()
        outs = my_model.forward({"image": [imagino]}, targets=targetino)
        my_out = my_model.dets_from_outs(outs)

        image = torch.from_numpy(dataset.get_raw_item(0)["image"].copy()).permute(2,0,1)
        my_res = draw_bounding_boxes(image.to(torch.uint8), my_out[0]["boxes"], width=5)
        plt.imshow(F.to_pil_image(my_res))
        ax = plt.gca()
        rectangles =  get_rectangles_for_many([dataset.get_raw_item(0)["bboxes"]], 
                                    edgecolor="green")[0]
        for rect in rectangles:
            ax.add_patch(rect)
        plt.savefig(f"dropout_norm_dataset_my_res_boxes_0.jpg")
        plt.close()


