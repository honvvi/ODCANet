import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


DATASET_CONFIGS = {
    "FMB": {
        "num_classes": 15,
        "mean_class_count": 14,
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
    "MH": {
        "num_classes": 9,
        "label_list": [
            "unlabeled", "car", "person", "bike", "curve", "car_stop",
            "guardrail", "color_cone", "bump",
        ],
        "label_color": [
            [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
        ],
    },
}


def read_rgb_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class RGBTSegmentationDataset(torch.utils.data.Dataset):
    """RGB-T test split with ImageNet normalization."""

    def __init__(self, data_root, dataset_name):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.rgb_dir, self.thermal_dir, self.label_dir = self._split_paths(data_root)
        self.samples = self._find_samples()

    def _split_paths(self, data_root):
        test_root = os.path.join(data_root, self.dataset_name, "test")
        if self.dataset_name == "FMB":
            return (
                os.path.join(test_root, "Visible"),
                os.path.join(test_root, "Infrared"),
                os.path.join(test_root, "Label"),
            )
        if self.dataset_name == "MH":
            return test_root, test_root, test_root
        return (
            os.path.join(test_root, "rgb"),
            os.path.join(test_root, "thermal"),
            os.path.join(test_root, "labels"),
        )

    def _find_samples(self):
        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(f"RGB test directory does not exist: {self.rgb_dir}")
        if self.dataset_name == "MH":
            samples = []
            for filename in sorted(os.listdir(self.rgb_dir)):
                if os.path.splitext(filename)[0][-3:] != "_th":
                    continue
                sample_id = filename.split("_")[0]
                rgb_path = os.path.join(self.rgb_dir, f"{sample_id}_rgb.png")
                label_path = os.path.join(self.label_dir, f"{sample_id}.png")
                if not os.path.isfile(rgb_path) or not os.path.isfile(label_path):
                    raise FileNotFoundError(
                        f"Incomplete MH sample '{sample_id}': expected "
                        f"{sample_id}_rgb.png, {sample_id}_th.png, and {sample_id}.png"
                    )
                samples.append(sample_id)
            return samples
        return sorted(
            filename
            for filename in os.listdir(self.rgb_dir)
            if filename.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename = self.samples[index]
        if self.dataset_name == "MH":
            rgb_path = os.path.join(self.rgb_dir, f"{filename}_rgb.png")
            thermal_path = os.path.join(self.thermal_dir, f"{filename}_th.png")
            label_path = os.path.join(self.label_dir, f"{filename}.png")
            output_filename = f"{filename}.png"
        else:
            rgb_path = os.path.join(self.rgb_dir, filename)
            thermal_path = os.path.join(self.thermal_dir, filename)
            label_path = os.path.join(self.label_dir, filename)
            output_filename = filename

        rgb = self._to_tensor(read_rgb_image(rgb_path))
        thermal = self._to_tensor(read_rgb_image(thermal_path))
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(f"Unable to read label: {label_path}")

        return {
            "rgb": rgb,
            "thermal": thermal,
            "label": torch.from_numpy(label.astype(np.int64)),
            "filename": output_filename,
        }

    @staticmethod
    def _to_tensor(image):
        image = image.astype(np.float32) / 255.0
        image = (image - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return torch.from_numpy(image.transpose(2, 0, 1)).float()


def build_test_loader(data_root, dataset_name, num_workers=4):
    dataset = RGBTSegmentationDataset(data_root, dataset_name)
    return dataset, DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
