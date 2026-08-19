from .mfnet_dataset import MFNetSegmentationDataset, build_mfnet_loader
from .rgbt_dataset import DATASET_CONFIGS, RGBTSegmentationDataset, build_rgbt_loader


def build_data_loader(
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
    if dataset_name == "MFNet":
        return build_mfnet_loader(
            data_root, split, batch_size, shuffle, num_workers, drop_last, training, crop_size
        )
    return build_rgbt_loader(
        data_root, dataset_name, split, batch_size, shuffle, num_workers,
        drop_last, training, crop_size
    )


def build_test_loader(data_root, dataset_name, num_workers=4):
    return build_data_loader(data_root, dataset_name, "test", 1, False, num_workers)


__all__ = [
    "DATASET_CONFIGS",
    "MFNetSegmentationDataset",
    "RGBTSegmentationDataset",
    "build_data_loader",
    "build_test_loader",
]
