import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .rgbt_dataset import TRAIN_SCALES, random_crop_pad_to_shape


class MFNetSegmentationDataset(Dataset):
    """MFNet split in the original four-channel PNG release format."""

    def __init__(self, data_root, split="test", training=False, crop_size=None):
        self.data_root = os.path.join(data_root, "MFNet")
        self.split = split
        self.training = training
        self.crop_size = crop_size
        self.images_dir = os.path.join(self.data_root, "images")
        self.labels_dir = os.path.join(self.data_root, "labels")
        self.samples = self._read_split()

    def _read_split(self):
        split_path = os.path.join(self.data_root, f"{self.split}.txt")
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"MFNet split file does not exist: {split_path}")
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(
                "MFNet requires 'images/' and 'labels/' under "
                f"{self.data_root}"
            )

        with open(split_path, "r", encoding="utf-8") as file:
            samples = [line.strip() for line in file if line.strip()]
        if not samples:
            raise ValueError(f"MFNet split file is empty: {split_path}")

        for sample_id in samples:
            image_path = os.path.join(self.images_dir, f"{sample_id}.png")
            label_path = os.path.join(self.labels_dir, f"{sample_id}.png")
            if not os.path.isfile(image_path) or not os.path.isfile(label_path):
                raise FileNotFoundError(
                    f"Incomplete MFNet sample '{sample_id}': expected "
                    f"{image_path} and {label_path}"
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id = self.samples[index]
        image_path = os.path.join(self.images_dir, f"{sample_id}.png")
        label_path = os.path.join(self.labels_dir, f"{sample_id}.png")

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to read MFNet image: {image_path}")
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(
                f"MFNet image must have shape [H, W, 4], got {image.shape}: {image_path}"
            )

        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        thermal = np.repeat(image[:, :, 3:4], 3, axis=2)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(f"Unable to read MFNet label: {label_path}")
        if label.ndim != 2:
            raise ValueError(
                f"MFNet label must be a single-channel class-index image, got "
                f"{label.shape}: {label_path}"
            )
        if self.training:
            rgb, thermal, label = self._augment(rgb, thermal, label)

        return {
            "rgb": self._to_tensor(rgb),
            "thermal": self._to_tensor(thermal),
            "label": torch.from_numpy(np.ascontiguousarray(label.astype(np.int64))),
            "filename": f"{sample_id}.png",
        }

    def _augment(self, rgb, thermal, label):
        # Same Cv-style augment as RGB-T / test/our MotherData.
        crop_size = rgb.shape[:2] if self.crop_size is None else tuple(self.crop_size)
        if random.random() >= 0.5:
            rgb = cv2.flip(rgb, 1)
            thermal = cv2.flip(thermal, 1)
            label = cv2.flip(label, 1)
        scale = random.choice(TRAIN_SCALES)
        height, width = rgb.shape[:2]
        scaled_size = (max(int(width * scale), 1), max(int(height * scale), 1))
        rgb = cv2.resize(rgb, scaled_size, interpolation=cv2.INTER_LINEAR)
        thermal = cv2.resize(thermal, scaled_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, scaled_size, interpolation=cv2.INTER_NEAREST)
        return random_crop_pad_to_shape(rgb, thermal, label, crop_size, pad_label_value=0)

    @staticmethod
    def _to_tensor(image):
        image = image.astype(np.float32) / 255.0
        image = (image - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def build_mfnet_loader(
    data_root,
    split="test",
    batch_size=1,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    training=False,
    crop_size=None,
):
    dataset = MFNetSegmentationDataset(data_root, split, training, crop_size)
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def build_mfnet_test_loader(data_root, num_workers=4):
    return build_mfnet_loader(data_root, "test", 1, False, num_workers)
