import numpy as np
from torch.utils.data.dataset import Dataset

from data_preprocessing.datasets.readers import Ego4dDataReader


def get_snao_video_adapter(dataset):
    if str.lower(dataset) == "epic":
        return EpicImageToVideoAdapter
    elif str.lower(dataset) == "egtea":
        return VideoReader
    elif str.lower(dataset) == "epicv":
        return VideoReader
    elif str.lower(dataset) in {"ego4d", "ego4djpg", "ego4djpgv2"}:
        return VideoReader


def pad_clip(clip, target_len):
    dif = target_len - len(clip)
    if dif > 0:
        clip = np.pad(clip, ((dif, 0), (0, 0), (0, 0), (0, 0)), constant_values=0)
    return clip


def get_clip_frame_idxs(stop_idx, sample_rate, num_frames, block=None, allow_zero=True):
    # subtract one for the final that must be included
    pos_no_frames = stop_idx // sample_rate

    start = stop_idx - min(sample_rate * pos_no_frames, (num_frames - 1) * sample_rate)
    if start <= 0 and not allow_zero:
        while start <= 0:
            start += sample_rate

    idxs = np.arange(start=start, stop=stop_idx + 1, step=sample_rate)
    # assert idxs[-1] == stop_idx
    return idxs


def get_clip_frame_idxs_block(stop_idx, sample_rate, num_frames, block, allow_zero=False):
    dilated_idxs = get_clip_frame_idxs(
        stop_idx - block - sample_rate + 1, sample_rate, num_frames - block, allow_zero=allow_zero
    )

    block_idxs = np.arange(stop_idx - block + 1, stop_idx + 1)

    return np.append(dilated_idxs, block_idxs)


class EpicImageToVideoAdapter(Dataset):
    """Used to read rgb frames and assemble them into a clip."""

    def __init__(self, image_reader, num_frames, sample_rate) -> None:
        super().__init__()
        self.image_reader = image_reader
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.clip_l = num_frames * sample_rate

    def __getitem__(self, index):
        idxs = get_clip_frame_idxs(index, self.sample_rate, self.num_frames, allow_zero=False)
        clip = np.array([self.image_reader[idx][0] for idx in idxs])
        clip = pad_clip(clip, self.num_frames)
        return clip

    def __len__(self):
        return len(self.image_reader)

    def get_frame(self, frame):
        idxs = get_clip_frame_idxs(frame, self.sample_rate, self.num_frames, allow_zero=False)
        clip = np.array([self.image_reader.get_frame(idx) for idx in idxs])
        clip = pad_clip(clip, self.num_frames)
        return clip

    def get_img_shape(self):
        return self.image_reader.get_img_shape()


class VideoReader(Dataset):
    def __init__(self, image_reader, num_frames, sample_rate) -> None:
        super().__init__()
        assert isinstance(image_reader, Ego4dDataReader)
        self.image_reader = image_reader
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.clip_l = num_frames * sample_rate

    def __getitem__(self, index):
        nr_frame = index + self.image_reader.offset
        return self.get_frame(nr_frame)

    def __len__(self):
        return len(self.image_reader)

    def get_frame(self, frame_no):
        idxs = get_clip_frame_idxs(frame_no, self.sample_rate, self.num_frames)
        clip = np.array(self.image_reader.get_clip(idxs))

        clip = pad_clip(clip, self.num_frames)
        return clip

    def get_img_shape(self):
        return self.image_reader.get_img_shape()


if __name__ == "__main__":
    idxs = get_clip_frame_idxs(29, 3, 3)
    assert np.all(idxs == np.array([23, 26, 29]))

    idxs = get_clip_frame_idxs(29, 3, 8)
    assert np.all(idxs == np.array([8, 11, 14, 17, 20, 23, 26, 29]))

    idxs = get_clip_frame_idxs(29, 3, 10)
    assert np.all(idxs == np.array([2, 5, 8, 11, 14, 17, 20, 23, 26, 29]))

    idxs = get_clip_frame_idxs(29, 3, 11)
    assert np.all(idxs == np.array([2, 5, 8, 11, 14, 17, 20, 23, 26, 29]))

    idxs = get_clip_frame_idxs(30, 3, 11)
    assert np.all(idxs == np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]))

    idxs = get_clip_frame_idxs(30, 3, 11, allow_zero=False)
    assert np.all(idxs == np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30]))

    idxs = get_clip_frame_idxs(30, 3, 10, allow_zero=True)
    assert np.all(idxs == np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30]))

    idxs = get_clip_frame_idxs(30, 3, 10, allow_zero=True)
    assert np.all(idxs == np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30]))

    idxs = get_clip_frame_idxs(30, 3, 9, allow_zero=True)
    assert np.all(idxs == np.array([6, 9, 12, 15, 18, 21, 24, 27, 30]))
