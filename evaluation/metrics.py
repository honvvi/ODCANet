import numpy as np


class SegmentationMetrics:
    """Accumulate semantic-segmentation metrics from predicted labels."""

    def __init__(self, num_classes, mean_class_count=None, exclude_eval_classes=None):
        self.num_classes = num_classes
        self.exclude_eval_classes = set(exclude_eval_classes or [])
        self.eval_class_indices = [
            class_index
            for class_index in range(num_classes)
            if class_index not in self.exclude_eval_classes
        ]
        # Kept for API compatibility; averaging uses nanmean, not this denominator.
        if mean_class_count is not None:
            self.mean_class_count = mean_class_count
        else:
            self.mean_class_count = len(self.eval_class_indices)
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, prediction, target):
        valid = (target >= 0) & (target < self.num_classes)
        for excluded_class in self.exclude_eval_classes:
            valid &= target != excluded_class
        labels = self.num_classes * target[valid].astype(int) + prediction[valid].astype(int)
        counts = np.bincount(labels, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def get_metrics(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            diagonal = np.diag(self.confusion_matrix)
            union = (
                self.confusion_matrix.sum(axis=1)
                + self.confusion_matrix.sum(axis=0)
                - diagonal
            )
            iou = diagonal / union
            accuracy_per_class = diagonal / self.confusion_matrix.sum(axis=1)

        iou_eval = iou[self.eval_class_indices]
        accuracy_eval = accuracy_per_class[self.eval_class_indices]

        return {
            "mIoU": float(np.nanmean(iou_eval)),
            "IoU_per_class": iou,
            "Pixel_Accuracy": float(diagonal.sum() / (self.confusion_matrix.sum() + 1e-10)),
            "mAccuracy": float(np.nanmean(accuracy_eval)),
            "Accuracy_per_class": accuracy_per_class,
        }


def build_segmentation_metrics(dataset_config):
    """Create metrics from a dataset config entry."""
    return SegmentationMetrics(
        dataset_config["num_classes"],
        dataset_config.get("mean_class_count"),
        dataset_config.get("exclude_eval_classes"),
    )
