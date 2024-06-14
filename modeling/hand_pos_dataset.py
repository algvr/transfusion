import math
import numpy as np
import pickle
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


feature_key_to_args_map = {
    "boxes": "box",
    "relevant_coords": "relevant_coord",
    "scores": "score",
    "contacts": "contact",
}


class HandPosDataset(Dataset):
    def __init__(
        self,
        to_wrap_dset,
        hand_pos_args,
    ) -> None:
        super().__init__()
        self.to_wrap_dset = to_wrap_dset
        self.hand_pos_args = hand_pos_args
        self.num_steps = hand_pos_args["num_steps"]
        self.step = hand_pos_args["step"]
        self.source = self.to_wrap_dset.to_wrap_dset.source

        self.readers = self.to_wrap_dset.readers
        self.hand_feat_dim = 21 * 3

        print("Loading hand data...")
        with open(self.hand_pos_args["path"], "rb") as f:
            self.hand_cache = pickle.load(f)
        print("Hand data loaded")

    def get_frame_box_vec(self, step_idx, hand_idx, frame_hand_data):
        side_name = ["left_hand", "right_hand"][hand_idx]
        box_xy = torch.from_numpy(frame_hand_data["hand_bbox_list"][0][side_name][:2])
        box_wh = torch.from_numpy(frame_hand_data["hand_bbox_list"][0][side_name][2:])
        img_width, img_height = frame_hand_data["image_width"], frame_hand_data["image_height"]
        return torch.cat((box_xy, box_xy + box_wh)) / torch.tensor([img_width, img_height, img_width, img_height])
    
    def get_frame_pose_vec(self, step_idx, hand_idx, frame_hand_data):
        side_name = ["left_hand", "right_hand"][hand_idx]
        side_data = frame_hand_data["pred_output_list"][0][side_name]
        img_width, img_height = frame_hand_data["image_width"], frame_hand_data["image_height"]
        return (torch.from_numpy(side_data["pred_joints_img"]) / torch.tensor([img_width, img_height, 100.])).flatten()

    def get_missing_pose_vec(self, step_idx, hand_idx):
        return torch.zeros(self.hand_feat_dim)
    
    def get_missing_box_vec(self, step_idx, hand_idx):
        return torch.zeros(4)

    def __getitem__(self, idx):
        example = self.to_wrap_dset[idx]
        annot = self.to_wrap_dset.to_wrap_dset.get_annot_by_idx(idx)
        video_id = annot.get("video_id", annot.get("video_uid"))
        frame_idx = annot.name
        frame_id = f'{video_id}_{"%07i" % frame_idx}'
        
        hand_poses = torch.stack([self.get_missing_pose_vec(step_idx, hand_idx) for hand_idx in [0, 1] for step_idx in range(self.num_steps)])
        hand_boxes = torch.stack([self.get_missing_box_vec(step_idx, hand_idx) for hand_idx in [0, 1] for step_idx in range(self.num_steps)])
        
        if video_id in self.hand_cache:
            # check which frames to use
            # TODO: use neighboring frames within tolerance if not found (debatable if will improve since extracted with stride anyway)
            frames_to_use = [max(0, frame_idx - step_idx * self.step) for step_idx in range(self.num_steps)]
            for hand_idx in [0, 1]:
                for step_idx, frame_idx_to_use in enumerate(frames_to_use):
                    if frame_idx_to_use in self.hand_cache[video_id]:
                        frame_hand_data = self.hand_cache[video_id][frame_idx_to_use]
                        if ("pred_output_list" in frame_hand_data
                            and len(frame_hand_data["pred_output_list"]) == 1
                            and frame_hand_data["pred_output_list"][0].get(["left_hand", "right_hand"][hand_idx]) not in [None, {}]):
                            hand_pos = self.num_steps * hand_idx + step_idx
                            hand_poses[hand_pos] = self.get_frame_pose_vec(step_idx, hand_idx, self.hand_cache[video_id][frame_idx_to_use])
                            hand_boxes[hand_pos] = self.get_frame_box_vec(step_idx, hand_idx, self.hand_cache[video_id][frame_idx_to_use])

        return {**example, "hand_poses": hand_poses, "hand_boxes": hand_boxes}

    def __len__(self):
        return len(self.to_wrap_dset)

    def get_nao_annots(self):
        return self.to_wrap_dset.nao_annots

    def get_raw_item(self, idx):
        return self.to_wrap_dset.get_raw_item(idx)

    def get_ignore_labels(self, col):
        return self.to_wrap_dset.get_ignore_labels(col)

    def get_label_cutoff(self):
        return self.to_wrap_dset.get_label_cutoff()

    def drop_cutoff_classes(self):
        return self.to_wrap_dset.drop_cutoff_classes()

    def get_verb_mapping(self):
        return self.to_wrap_dset.to_wrap_dset.verb_mapping

    def get_noun_mapping(self):
        return self.to_wrap_dset.to_wrap_dset.noun_mapping

    def set_input_transforms(self, transforms):
        self.to_wrap_dset.set_input_transforms(transforms)

    def set_hmap_transforms(self, hmap_transforms):
        self.to_wrap_dset.set_hmap_transforms(hmap_transforms)

    def set_resize_fn(self, resize_fn):
        self.to_wrap_dset.set_resize_fn(resize_fn)

    def get_narration_embeds(self):
        nar_embeds = np.array([self.narration_embeds[k] for k in sorted(self.narration_embeds.keys())])
        nar_embeds = np.append(nar_embeds, np.zeros((1, nar_embeds.shape[-1])), axis=0)
        return nar_embeds


    def get_no_nouns(self):
        return self.to_wrap_dset.get_no_nouns()

    def get_no_verbs(self):
        return self.to_wrap_dset.get_no_verbs()

    def get_classes(self, clzz):
        return self.to_wrap_dset.get_classes(clzz)

    def set_resize_fn(self, resize_fn):
        self.to_wrap_dset.set_resize_fn(resize_fn)

    def set_input_transforms(self, transforms):
        self.to_wrap_dset.set_input_transforms(transforms)

    def set_localiz_transforms(self, transforms):
        self.to_wrap_dset.set_localiz_transforms(transforms)

    def set_flow_transform(self, transform):
        self.to_wrap_dset.set_flow_transform(transform)
        
