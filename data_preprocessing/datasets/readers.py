import cv2
import detectron2.data.transforms as T
import os
import io
import lmdb
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List
from torch.utils.data import Dataset
import torch

from data_preprocessing.utils.path_utils import data_roots, get_path_to_actor


FLOW_IMG_SHAPE_UP = (2, 360, 480)
JPG_FLOW_IMG_SHAPE_UP = (2, 480, 640)
FLOW_IMG_SHAPE_DOWN = (2, FLOW_IMG_SHAPE_UP[1] // 8, FLOW_IMG_SHAPE_UP[2] // 8)


def get_image_reader(dataset):
    if str.lower(dataset) == "ego4d":
        return Ego4dDataReader
    elif str.lower(dataset) == "ego4djpg":
        return Ego4dJpgReader 
    elif str.lower(dataset) == "ego4djpgv2":
        return Ego4dJpgReader
    raise NotImplementedError(f"{dataset=} type does not exist")


class Ego4dJpgReader(Dataset):
    def __init__(self, data_path, actor_id) -> None:
        super().__init__()
        self.actor_id = actor_id
        self.video_id = actor_id
        self.data_path = data_path
        self.frame_template = "{actor_id:s}_{frame_number:07d}.jpg"

    def __getitem__(self, index):
        uid = self.frame_template.format(actor_id=self.actor_id, frame_number=index)
        img = np.asarray(Image.open(self.data_path / uid))
        return img

    def get_frame(self, idx):
        return self[idx]

    def get_img_shape(self):
        rand_img_path = list(self.data_path.glob(f"{self.actor_id}*"))[0]
        img = np.asarray(Image.open(rand_img_path))
        return img.shape


class Ego4dDataReaderMp4(Dataset):
    """Used to read a single frame/clip from ego4d videos(mp4). Also computes offset for videos with blank beginning."""

    def __init__(self, actor_path, video, frame_stride=1, overwrite_path=False):
        self.video = video
        if overwrite_path:
            self.actor_path = actor_path
            self.to_load = video
        else:    
            self.actor_path = actor_path.parent
            self.to_load = self.actor_path.joinpath(self.video)
            self.path_to_root = actor_path.parent

        assert self.to_load.exists(), f"{self.to_load} video path does not exists"
        # self.video_f = open(self.to_load, "rb").read()
        video_h = cv2.VideoCapture(str(self.to_load))
        self.fps = video_h.get(cv2.CAP_PROP_FPS)
        _, img = video_h.read()
        self.img_shape = img.shape
        video_h.release()
        self.offset = 0

    def get_img_shape(self):
        return self.img_shape

    def __len__(self):
        video_h = cv2.VideoCapture(str(self.to_load))
        leny = int(video_h.get(cv2.CAP_PROP_FRAME_COUNT)) - self.offset
        video_h.release()
        return leny

    def get_frame(self, frame):
        video_h = cv2.VideoCapture(str(self.to_load))
        video_h.set(cv2.CAP_PROP_POS_FRAMES, frame)
        succ, img = video_h.read()
        video_h.release()
        return img[:, :, ::-1]

    def get_clip(self, idxs):
        """Read succesively from the video frames specified by idxs which should be sorted in increasing order for fast reading."""
        video_h = cv2.VideoCapture(str(self.to_load))
        delta = idxs[1] - idxs[0]
        video_h.set(cv2.CAP_PROP_POS_FRAMES, idxs[0])

        read = 0
        imgs = []
        while read <= delta * (len(idxs) - 1):
            succ, img = video_h.read()
            if not succ:
                print(f"Error reading from video reader:{self.to_load},{idxs=}")
            if read % delta == 0:
                imgs.append(img[:, :, ::-1])
            read += 1

        video_h.release()
        return imgs

    def get_ms(self, ms):
        video_h = cv2.VideoCapture(str(self.to_load))
        video_h.set(cv2.CAP_PROP_POS_MSEC, ms)
        succ, img = video_h.read()
        assert succ, f"Could not read at {ms=}, {self.to_load=}"
        video_h.release()
        return img[:, :, ::-1]

    def __getitem__(self, idx):
        nr_frame = idx + self.offset

        video_h = cv2.VideoCapture(str(self.to_load))
        video_h.set(cv2.CAP_PROP_POS_FRAMES, nr_frame)
        succ, img = video_h.read()
        video_h.release()

        return img[:, :, ::-1], nr_frame


class Ego4dDataReader:
    def __init__(
        self,
        path_to_root: Path,
        video: str,
        readonly=True,
        lock=False,
        frame_template="{video_id:s}_{frame_number:010d}",
        map_size=1099511627776,
        jpg=True,
    ) -> None:

        self.path_to_root = path_to_root.parent
        if isinstance(self.path_to_root, str):
            self.path_to_root = Path(self.path_to_root)

        self.readonly = readonly
        self.lock = lock
        self.map_size = map_size
        self.frame_template = frame_template
        self.video_id = path_to_root.name
        self.jpg = jpg
        self.load_img_shape()

    def _get_parent(self, parent: str) -> lmdb.Environment:
        return lmdb.open(
            str(self.path_to_root / parent), map_size=self.map_size, readonly=self.readonly, lock=self.lock
        )

    def get_img_shape(self):
        return self.img_shape

    def get_frame(self, frame: int) -> np.ndarray:
        with self._get_parent(self.video_id) as env:
            with env.begin(write=False) as txn:
                data = txn.get(self.frame_template.format(video_id=self.video_id, frame_number=frame).encode())
                if self.jpg:
                    file_bytes = np.asarray(bytearray(io.BytesIO(data).read()), dtype=np.uint8)
                    out = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)[:, :, ::-1]
                else:
                    out = np.fromstring(data, dtype=np.float32).reshape((-1,) + self.img_shape)
                return out

    def get_clip(self, frames: List[int]) -> List[np.ndarray]:
        out = []
        with self._get_parent(self.video_id) as env:
            with env.begin(write=False) as txn:
                if self.jpg:
                    for frame in frames:
                        data = txn.get(self.frame_template.format(video_id=self.video_id, frame_number=frame).encode())
                        file_bytes = np.asarray(bytearray(io.BytesIO(data).read()), dtype=np.uint8)
                        val = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)[:, :, ::-1]
                        out.append(val)
                else:
                    for frame in frames:
                        data = txn.get(self.frame_template.format(video_id=self.video_id, frame_number=frame).encode())
                        val = np.fromstring(data, dtype=np.float32).reshape((-1,) + self.img_shape)
                        out.append(val)
            return out

    def load_img_shape(self):
        with self._get_parent(self.video_id) as env:
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                cursor.first()
                data = cursor.value()
                if self.jpg:
                    file_bytes = np.asarray(bytearray(io.BytesIO(data).read()), dtype=np.uint8)
                    self.img_shape = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR).shape
                cursor.close()
        # self.img_shape = 480, 640, 3

    def set_img_shape(self, img_shape):
        self.img_shape = img_shape

    def get_existing_keys(self):
        keys = []
        for parent in self.path_to_root.iterdir():
            with self._get_parent(parent.name) as env:
                with env.begin() as txn:
                    keys += list(txn.cursor().iternext(values=False))
        return keys


class FlowDataReader:
    """Same as Ego4D data reader but reads floating point numpy arrays that require knowing the shape in advance"""

    def __init__(
        self,
        path_to_root: Path,
        video: str,
        readonly=True,
        lock=False,
        frame_template="{video_id:s}_{frame_number:010d}",
        map_size=4099511627776,
        array_load_shape=FLOW_IMG_SHAPE_DOWN,
        img_shape=FLOW_IMG_SHAPE_UP,
    ) -> None:

        self.path_to_root = path_to_root
        if isinstance(self.path_to_root, str):
            self.path_to_root = Path(self.path_to_root)

        self.readonly = readonly
        self.lock = lock
        self.map_size = map_size
        self.frame_template = frame_template
        self.video_id = video
        self.array_load_shape = array_load_shape
        self.img_shape = img_shape
        self.keys = self.get_existing_keys()

    def _get_parent(self, parent: str) -> lmdb.Environment:
        return lmdb.open(
            str(self.path_to_root / parent), map_size=self.map_size, readonly=self.readonly, lock=self.lock
        )

    def get_img_shape(self):
        return self.img_shape

    def val_from_bytes(self, data):
        # val = torch.from_numpy(np.fromstring(data, dtype=np.float32).reshape(self.array_load_shape)).cpu()
        # return upflow8_shape(val[None], self.img_shape[:2]).numpy()[0].transpose(1, 2, 0)
        raise NotImplementedError()

    def get_frame(self, frame: int) -> np.ndarray:
        with self._get_parent(self.video_id) as env:
            with env.begin(write=False) as txn:
                data = txn.get(self.frame_template.format(video_id=self.video_id, frame_number=frame).encode())
                val = self.val_from_bytes(data)
                return val

    def get_clip(self, frames: List[int]) -> List[np.ndarray]:
        out = []
        with self._get_parent(self.video_id) as env:
            with env.begin(write=False) as txn:
                for frame in frames:
                    data = txn.get(self.frame_template.format(video_id=self.video_id, frame_number=frame).encode())
                    try:
                        val = self.val_from_bytes(data)
                        out.append(val)
                    except:
                        raise Exception(f"Error at frame {frame}, actor: {self.video_id}, fn get_clip")
            return out

    def set_img_shape(self, img_shape):
        self.img_shape = img_shape

    def get_existing_keys(self):
        keys = []
        with self._get_parent(self.video_id) as env:
            with env.begin() as txn:
                keys += list(txn.cursor().iternext(values=False))
        keys = [x.decode("ascii") for x in keys]

        self.keys = set(keys)

        return self.keys

    def check_frames_exist(self, frames):
        keys = self.get_existing_keys()
        for frame in frames:
            key = self.frame_template.format(video_id=self.video_id, frame_number=frame)
            if key not in keys:
                return False

        return True


class FlowDataReaderJpg(FlowDataReader):
    def __init__(
        self,
        path_to_root: Path,
        video: str,
        readonly=True,
        lock=False,
        frame_template="{video_id:s}_{frame_number:010d}",
        map_size=4099511627776,
        array_load_shape=JPG_FLOW_IMG_SHAPE_UP,
        img_shape=FLOW_IMG_SHAPE_UP,
    ) -> None:
        super().__init__(
            path_to_root, f"{video}_jpg", readonly, lock, frame_template, map_size, array_load_shape, img_shape
        )

    def val_from_bytes(self, data):
        file_bytes = np.asarray(bytearray(io.BytesIO(data).read()), dtype=np.uint8)
        try:
            val = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE).reshape(self.array_load_shape).transpose(1, 2, 0)
        except:
            val = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE).reshape(FLOW_IMG_SHAPE_UP).transpose(1, 2, 0)
        if val.shape[:2] != self.img_shape[:2][::-1]:
            val = cv2.resize(val, dsize=self.img_shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        return val


class SFastFeaturesReader(FlowDataReader):
    def __init__(
        self,
        path_to_root: Path,
        video: str,
        readonly=True,
        lock=False,
        frame_template="{video_id:s}_{frame_number:010d}",
        map_size=4099511627776,
        array_load_shape=FLOW_IMG_SHAPE_DOWN,
        img_shape=FLOW_IMG_SHAPE_UP,
        no_segs_to_read=2,
    ) -> None:
        super().__init__(path_to_root, video, readonly, lock, frame_template, map_size, array_load_shape, img_shape)
        self.no_segs_to_read = no_segs_to_read
        self.sfast_f_dim = 2304
        self.no_bytes_needed = self.sfast_f_dim * 4 * self.no_segs_to_read

    def val_from_bytes(self, data):
        # file_bytes = np.asarray(bytearray(io.BytesIO(data).read()), dtype=np.float32)
        val = np.fromstring(data[:self.no_bytes_needed], dtype=np.float32).reshape((self.no_segs_to_read, self.sfast_f_dim))
        return val


class ImageDataset(Dataset):
    def __init__(self, cfg, data_reader):
        self.data_reader = data_reader
        self.cfg = cfg
        self.aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

    def __len__(self):
        return len(self.data_reader)

    def __getitem__(self, idx):
        img, metadata = self.data_reader[idx]
        h, w = img.shape[:2]
        img = self.aug.get_transform(img).apply_image(img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

        return {
            "image": img,
            "frames": torch.as_tensor(metadata),
            "height": torch.as_tensor(h),
            "width": torch.as_tensor(w),
        }


def collate_unidet_input(batch):
    return [{"image": x["image"], "height": x["height"], "width": x["width"]} for x in batch], {
        "frames": torch.stack([x["frames"] for x in batch])
    }

