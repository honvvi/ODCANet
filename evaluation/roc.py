import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve


class SegmentationBenchmark:
    """Collect ROC data during the validation pass."""

    def __init__(self, output_dir, label_list, roc_pixels_per_image=4096):
        self.output_dir = output_dir
        self.label_list = label_list
        self.roc_pixels_per_image = roc_pixels_per_image
        self.probabilities = []
        self.labels = []
        os.makedirs(output_dir, exist_ok=True)

    def add_roc_batch(self, logits, labels):
        probabilities = logits.softmax(dim=1).detach()
        labels = labels.detach()

        for probability, label in zip(probabilities, labels):
            valid = (label >= 0) & (label < probability.shape[0])
            probability = probability.permute(1, 2, 0)[valid]
            label = label[valid]
            if self.roc_pixels_per_image and label.numel() > self.roc_pixels_per_image:
                indices = torch.linspace(
                    0, label.numel() - 1, self.roc_pixels_per_image, device=label.device
                ).long()
                probability = probability[indices]
                label = label[indices]
            self.probabilities.append(probability.cpu().numpy())
            self.labels.append(label.cpu().numpy())

    def save_roc(self):
        if not self.probabilities:
            return {}

        probabilities = np.concatenate(self.probabilities, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        curves = {}
        aucs = {}
        plt.figure(figsize=(7, 5))

        for class_id, class_name in enumerate(self.label_list):
            binary_labels = (labels == class_id).astype(np.uint8)
            if binary_labels.min() == binary_labels.max():
                continue
            fpr, tpr, _ = roc_curve(binary_labels, probabilities[:, class_id])
            class_auc = auc(fpr, tpr)
            curves[class_name] = {"fpr": fpr, "tpr": tpr}
            aucs[class_name] = float(class_auc)
            plt.plot(fpr, tpr, lw=1.2, label=f"{class_name} (AUC={class_auc:.4f})")

        if not curves:
            plt.close()
            return {}

        plt.plot([0, 1], [0, 1], "k--", lw=0.8)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "roc_curve.png"), dpi=300)
        plt.close()
        np.savez(
            os.path.join(self.output_dir, "roc_curves.npz"),
            labels=labels,
            probabilities=probabilities,
            **{f"{name}_fpr": curve["fpr"] for name, curve in curves.items()},
            **{f"{name}_tpr": curve["tpr"] for name, curve in curves.items()},
        )
        return {
            "macro_auc": float(np.mean(list(aucs.values()))),
            "per_class_auc": aucs,
            "roc_sampled_pixels": int(labels.size),
            "roc_pixels_per_image": self.roc_pixels_per_image,
        }

    def save_summary(self, summary):
        output_path = os.path.join(self.output_dir, "summary.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=True)
        return output_path
