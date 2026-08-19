import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


DATASET_CONFIGS = {
    "FMB": {
        "num_classes": 15,
        # "mean_class_count": 14,
        # "exclude_eval_classes": [0],
        "label_list": [
            "unlabeled", "Road", "Sidewalk", "Building", "Traffic Lamp",
            "Traffic Sign", "Vegetation", "Sky", "Person", "Car", "Truck",
            "Bus", "Motorcycle", "Bicycle", "Pole",
        ],
        "label_color": [
            [0, 0, 0], [179, 228, 228], [181, 57, 133], [67, 162, 177],
            [200, 178, 50], [132, 45, 199], [66, 172, 84], [179, 73, 79],
            [76, 99, 166], [66, 121, 253], [6, 6, 6], [12, 12, 12],
            [105, 153, 140], [222, 215, 158], [135, 113, 90],
        ],
    },
    "PST": {
        "num_classes": 5,
        "label_list": [
            "Background", "Fire-extinguisher", "Backpack", "Hand-drill", "Survivor",
        ],
        "label_color": [
            [0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255],
        ],
    },
    "MFNet": {
        "num_classes": 9,
        "label_list": [
            "unlabeled", "car", "person", "bike", "curve", "car_stop",
            "guardrail", "color_cone", "bump",
        ],
        "label_color": [
            [0, 0, 0], [64, 0, 128], [64, 64, 0], [0, 128, 192],
            [0, 0, 192], [128, 128, 0], [64, 64, 128], [192, 128, 0],
            [192, 64, 0],
        ],
    },
}

TRAIN_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]


def read_rgb_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _pad_image_to_shape(img, shape, value):
    """Center-pad to shape; matches test/our pad_image_to_shape."""
    target_h, target_w = shape
    pad_h = max(target_h - img.shape[0], 0)
    pad_w = max(target_w - img.shape[1], 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)


def _generate_random_crop_pos(ori_size, crop_size):
    """Matches test/our generate_random_crop_pos (including its randint range)."""
    h, w = ori_size
    crop_h, crop_w = crop_size
    pos_h, pos_w = 0, 0
    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)
    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)
    return pos_h, pos_w


def random_crop_pad_to_shape(rgb, thermal, label, crop_size, pad_label_value=0):
    """Crop then center-pad to crop_size; matches test/our MotherData (Cv)."""
    crop_h, crop_w = crop_size
    start_h, start_w = _generate_random_crop_pos(rgb.shape[:2], crop_size)
    rgb = rgb[start_h:start_h + crop_h, start_w:start_w + crop_w]
    thermal = thermal[start_h:start_h + crop_h, start_w:start_w + crop_w]
    label = label[start_h:start_h + crop_h, start_w:start_w + crop_w]
    rgb = _pad_image_to_shape(rgb, crop_size, 0)
    thermal = _pad_image_to_shape(thermal, crop_size, 0)
    label = _pad_image_to_shape(label, crop_size, pad_label_value)
    return rgb, thermal, label


class RGBTSegmentationDataset(torch.utils.data.Dataset):
    """RGB-T split with ImageNet normalization.

    Training augmentation follows test/our MotherData (dataloader_type: Cv):
    flip -> multi-scale resize -> crop/pad back to the pre-scale size
    (or to an explicit crop_size when provided).
    """

    def __init__(self, data_root, dataset_name, split="test", training=False, crop_size=None):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.split = split
        self.training = training
        self.crop_size = crop_size
        self.config = DATASET_CONFIGS[dataset_name]
        self.rgb_dir, self.thermal_dir, self.label_dir = self._split_paths(data_root)
        self.samples = self._find_samples()

    def _split_paths(self, data_root):
        split_root = os.path.join(data_root, self.dataset_name, self.split)
        if self.dataset_name == "FMB":
            return (
                os.path.join(split_root, "Visible"),
                os.path.join(split_root, "Infrared"),
                os.path.join(split_root, "Label"),
            )
        return (
            os.path.join(split_root, "rgb"),
            os.path.join(split_root, "thermal"),
            os.path.join(split_root, "labels"),
        )

    def _find_samples(self):
        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(f"RGB {self.split} directory does not exist: {self.rgb_dir}")
        samples = sorted(
            filename
            for filename in os.listdir(self.rgb_dir)
            if filename.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not samples:
            raise ValueError(f"No {self.split} images found in: {self.rgb_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename = self.samples[index]
        rgb_path = os.path.join(self.rgb_dir, filename)
        thermal_path = os.path.join(self.thermal_dir, filename)
        label_path = os.path.join(self.label_dir, filename)

        rgb = read_rgb_image(rgb_path)
        thermal = read_rgb_image(thermal_path)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(f"Unable to read label: {label_path}")
        if label.ndim > 2:
            label = label[:, :, 0]
        if self.training:
            rgb, thermal, label = self._augment(rgb, thermal, label)

        return {
            "rgb": self._to_tensor(rgb),
            "thermal": self._to_tensor(thermal),
            "label": torch.from_numpy(np.ascontiguousarray(label.astype(np.int64))),
            "filename": filename,
        }

    def _augment(self, rgb, thermal, label):
        # Match test/our MotherData: crop target is original HxW when crop_size is None.
        if self.crop_size is None:
            crop_size = rgb.shape[:2]
        else:
            crop_size = tuple(self.crop_size)

        # 1) random horizontal flip
        if random.random() >= 0.5:
            rgb = cv2.flip(rgb, 1)
            thermal = cv2.flip(thermal, 1)
            label = cv2.flip(label, 1)

        # 2) random multi-scale resize
        scale = random.choice(TRAIN_SCALES)
        height, width = rgb.shape[:2]
        scaled_size = (max(int(width * scale), 1), max(int(height * scale), 1))
        rgb = cv2.resize(rgb, scaled_size, interpolation=cv2.INTER_LINEAR)
        thermal = cv2.resize(thermal, scaled_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, scaled_size, interpolation=cv2.INTER_NEAREST)

        # 3) crop/pad back to crop_size (label pad=0, same as our Cv loader)
        return random_crop_pad_to_shape(rgb, thermal, label, crop_size, pad_label_value=0)

    @staticmethod
    def _to_tensor(image):
        image = image.astype(np.float32) / 255.0
        image = (image - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def build_rgbt_loader(
    data_root,
    dataset_name,
    split="test",
    batch_size=1,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    training=False,
    crop_size=None,
):
    dataset = RGBTSegmentationDataset(data_root, dataset_name, split, training, crop_size)
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def build_test_loader(data_root, dataset_name, num_workers=4):
    return build_rgbt_loader(data_root, dataset_name, "test", 1, False, num_workers)
