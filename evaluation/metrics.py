import numpy as np


class SegmentationMetrics:
    """Accumulate semantic-segmentation metrics from predicted labels."""

    def __init__(self, num_classes, mean_class_count=None):
        self.num_classes = num_classes
        self.mean_class_count = mean_class_count or num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, prediction, target):
        valid = (target >= 0) & (target < self.num_classes)
        labels = self.num_classes * target[valid].astype(int) + prediction[valid].astype(int)
        counts = np.bincount(labels, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def get_metrics(self):
        diagonal = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1)
            + self.confusion_matrix.sum(axis=0)
            - diagonal
        )
        iou = diagonal / (union + 1e-10)
        accuracy_per_class = diagonal / (self.confusion_matrix.sum(axis=1) + 1e-10)

        return {
            "mIoU": np.nansum(iou) / self.mean_class_count,
            "IoU_per_class": iou,
            "Pixel_Accuracy": diagonal.sum() / (self.confusion_matrix.sum() + 1e-10),
            "mAccuracy": np.nansum(accuracy_per_class) / self.mean_class_count,
            "Accuracy_per_class": accuracy_per_class,
        }
