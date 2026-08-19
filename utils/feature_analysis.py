"""Feature-level statistics for OFDM common/unique decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


STAGE_COUNT = 4
EPS = 1e-8


@dataclass
class StageAccumulator:
    pixel_count: int = 0
    sum_abs_cos_rgb: float = 0.0
    sum_abs_cos_thermal: float = 0.0
    sum_abs_cos_rgb_common: float = 0.0
    sum_abs_cos_unique_cross: float = 0.0
    sum_sq_rgb: float = 0.0
    sum_sq_thermal: float = 0.0
    sum_dot_rgb: float = 0.0
    sum_dot_thermal: float = 0.0
    cka_samples: List[torch.Tensor] = field(default_factory=list)


@dataclass
class RunAccumulator:
    sample_count: int = 0
    stages: List[StageAccumulator] = field(
        default_factory=lambda: [StageAccumulator() for _ in range(STAGE_COUNT)]
    )


def extract_ofdm_features(
    nets,
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    device: torch.device,
    dataset_name: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Run encoder + OFDM and return per-stage common/unique tensors."""
    rgb_batch = rgb.unsqueeze(0).to(device, non_blocking=True)
    thermal_batch = thermal.unsqueeze(0).to(device, non_blocking=True)
    feature_maps = {}

    rgb_features = nets["Shared_Encoder"](rgb_batch, feature_maps, dataset_name)
    thermal_features = nets["Shared_Encoder"](thermal_batch, feature_maps, dataset_name)
    _, r_unique, r_common, t_unique, t_common = nets["OFDM"](
        rgb_features, thermal_features, feature_maps, dataset_name
    )
    return r_unique, r_common, t_unique, t_common


def _flatten_channels(features: torch.Tensor) -> torch.Tensor:
    """Convert [B, C, H, W] to [C, N]."""
    batch, channels, height, width = features.shape
    return features.reshape(batch, channels, -1)[0]


def mean_abs_cosine(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    """Match the orthogonality loss: mean absolute cosine over spatial pixels."""
    cosine = F.cosine_similarity(features_a, features_b, dim=1, eps=EPS)
    return float(torch.mean(torch.abs(cosine)).item())


def mean_pearson_correlation(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    """Average Pearson correlation across channels on flattened spatial tokens."""
    flat_a = _flatten_channels(features_a)
    flat_b = _flatten_channels(features_b)
    channels = flat_a.shape[0]
    correlations = []
    for channel_idx in range(channels):
        vector_a = flat_a[channel_idx]
        vector_b = flat_b[channel_idx]
        centered_a = vector_a - vector_a.mean()
        centered_b = vector_b - vector_b.mean()
        denominator = centered_a.norm() * centered_b.norm()
        if float(denominator.item()) <= EPS:
            continue
        correlations.append(float((centered_a * centered_b).sum().item() / denominator.item()))
    if not correlations:
        return 0.0
    return sum(correlations) / len(correlations)


def _subsample_columns(matrix: torch.Tensor, max_samples: int) -> torch.Tensor:
    if matrix.shape[1] <= max_samples:
        return matrix
    indices = torch.linspace(0, matrix.shape[1] - 1, steps=max_samples, device=matrix.device).long()
    return matrix[:, indices]


def linear_cka(features_a: torch.Tensor, features_b: torch.Tensor, max_samples: int = 4096) -> float:
    """Linear CKA between two feature maps, with optional pixel subsampling."""
    matrix_a = _subsample_columns(_flatten_channels(features_a), max_samples)
    matrix_b = _subsample_columns(_flatten_channels(features_b), max_samples)

    matrix_a = matrix_a - matrix_a.mean(dim=1, keepdim=True)
    matrix_b = matrix_b - matrix_b.mean(dim=1, keepdim=True)

    gram_a = matrix_a @ matrix_a.transpose(0, 1)
    gram_b = matrix_b @ matrix_b.transpose(0, 1)
    numerator = torch.trace(gram_a @ gram_b)
    denominator = torch.sqrt(torch.trace(gram_a @ gram_a) * torch.trace(gram_b @ gram_b))
    if float(denominator.item()) <= EPS:
        return 0.0
    return float((numerator / denominator).item())


def _update_stage_accumulator(
    stage_acc: StageAccumulator,
    r_unique: torch.Tensor,
    r_common: torch.Tensor,
    t_unique: torch.Tensor,
    t_common: torch.Tensor,
    collect_cka: bool,
) -> int:
    pixel_count = int(r_unique.shape[2] * r_unique.shape[3])
    stage_acc.pixel_count += pixel_count
    stage_acc.sum_abs_cos_rgb += mean_abs_cosine(r_unique, r_common) * pixel_count
    stage_acc.sum_abs_cos_thermal += mean_abs_cosine(t_unique, t_common) * pixel_count
    stage_acc.sum_abs_cos_rgb_common += mean_abs_cosine(r_common, t_common) * pixel_count
    stage_acc.sum_abs_cos_unique_cross += mean_abs_cosine(r_unique, t_unique) * pixel_count

    stage_acc.sum_sq_rgb += mean_pearson_correlation(r_unique, r_common) ** 2 * pixel_count
    stage_acc.sum_dot_rgb += mean_pearson_correlation(r_unique, r_common) * pixel_count
    stage_acc.sum_sq_thermal += mean_pearson_correlation(t_unique, t_common) ** 2 * pixel_count
    stage_acc.sum_dot_thermal += mean_pearson_correlation(t_unique, t_common) * pixel_count

    if collect_cka:
        stage_acc.cka_samples.append(
            torch.tensor(
                [
                    linear_cka(r_unique, r_common),
                    linear_cka(t_unique, t_common),
                    linear_cka(r_common, t_common),
                    linear_cka(r_unique, t_unique),
                ],
                dtype=torch.float32,
            )
        )
    return pixel_count


def update_run_accumulator(
    run_acc: RunAccumulator,
    r_unique: Iterable[torch.Tensor],
    r_common: Iterable[torch.Tensor],
    t_unique: Iterable[torch.Tensor],
    t_common: Iterable[torch.Tensor],
    collect_cka: bool = False,
) -> None:
    run_acc.sample_count += 1
    for stage_idx, (unique_rgb, common_rgb, unique_t, common_t) in enumerate(
        zip(r_unique, r_common, t_unique, t_common)
    ):
        _update_stage_accumulator(
            run_acc.stages[stage_idx],
            unique_rgb,
            common_rgb,
            unique_t,
            common_t,
            collect_cka,
        )


def _weighted_mean(total: float, count: int) -> Optional[float]:
    if count <= 0:
        return None
    return total / count


def _stage_summary(stage_acc: StageAccumulator) -> Dict[str, Optional[float]]:
    pixel_count = stage_acc.pixel_count
    summary = {
        "pixel_count": pixel_count,
        "mean_abs_cosine_rgb": _weighted_mean(stage_acc.sum_abs_cos_rgb, pixel_count),
        "mean_abs_cosine_thermal": _weighted_mean(stage_acc.sum_abs_cos_thermal, pixel_count),
        "mean_abs_cosine_all": _weighted_mean(
            stage_acc.sum_abs_cos_rgb + stage_acc.sum_abs_cos_thermal,
            pixel_count * 2 if pixel_count > 0 else 0,
        ),
        "mean_abs_cosine_common_cross_modal": _weighted_mean(
            stage_acc.sum_abs_cos_rgb_common, pixel_count
        ),
        "mean_abs_cosine_unique_cross_modal": _weighted_mean(
            stage_acc.sum_abs_cos_unique_cross, pixel_count
        ),
        "mean_pearson_rgb": _weighted_mean(stage_acc.sum_dot_rgb, pixel_count),
        "mean_pearson_thermal": _weighted_mean(stage_acc.sum_dot_thermal, pixel_count),
    }
    if stage_acc.cka_samples:
        cka_tensor = torch.stack(stage_acc.cka_samples, dim=0)
        summary["linear_cka_rgb_common_unique"] = float(cka_tensor[:, 0].mean().item())
        summary["linear_cka_thermal_common_unique"] = float(cka_tensor[:, 1].mean().item())
        summary["linear_cka_common_cross_modal"] = float(cka_tensor[:, 2].mean().item())
        summary["linear_cka_unique_cross_modal"] = float(cka_tensor[:, 3].mean().item())
    return summary


def summarize_run(label: str, run_acc: RunAccumulator) -> Dict:
    stage_summaries = []
    overall = {
        "pixel_count": 0,
        "sum_abs_cos_rgb": 0.0,
        "sum_abs_cos_thermal": 0.0,
        "sum_abs_cos_common_cross_modal": 0.0,
        "sum_abs_cos_unique_cross_modal": 0.0,
        "sum_dot_rgb": 0.0,
        "sum_dot_thermal": 0.0,
        "cka_samples": [],
    }

    for stage_idx, stage_acc in enumerate(run_acc.stages, start=1):
        stage_summary = _stage_summary(stage_acc)
        stage_summary["stage"] = stage_idx
        stage_summaries.append(stage_summary)

        overall["pixel_count"] += stage_acc.pixel_count
        overall["sum_abs_cos_rgb"] += stage_acc.sum_abs_cos_rgb
        overall["sum_abs_cos_thermal"] += stage_acc.sum_abs_cos_thermal
        overall["sum_abs_cos_common_cross_modal"] += stage_acc.sum_abs_cos_rgb_common
        overall["sum_abs_cos_unique_cross_modal"] += stage_acc.sum_abs_cos_unique_cross
        overall["sum_dot_rgb"] += stage_acc.sum_dot_rgb
        overall["sum_dot_thermal"] += stage_acc.sum_dot_thermal
        overall["cka_samples"].extend(stage_acc.cka_samples)

    pixel_count = overall["pixel_count"]
    overall_summary = {
        "sample_count": run_acc.sample_count,
        "mean_abs_cosine_rgb": _weighted_mean(overall["sum_abs_cos_rgb"], pixel_count),
        "mean_abs_cosine_thermal": _weighted_mean(overall["sum_abs_cos_thermal"], pixel_count),
        "mean_abs_cosine_all": _weighted_mean(
            overall["sum_abs_cos_rgb"] + overall["sum_abs_cos_thermal"],
            pixel_count * 2 if pixel_count > 0 else 0,
        ),
        "mean_abs_cosine_common_cross_modal": _weighted_mean(
            overall["sum_abs_cos_common_cross_modal"], pixel_count
        ),
        "mean_abs_cosine_unique_cross_modal": _weighted_mean(
            overall["sum_abs_cos_unique_cross_modal"], pixel_count
        ),
        "mean_pearson_rgb": _weighted_mean(overall["sum_dot_rgb"], pixel_count),
        "mean_pearson_thermal": _weighted_mean(overall["sum_dot_thermal"], pixel_count),
    }
    if overall["cka_samples"]:
        cka_tensor = torch.stack(overall["cka_samples"], dim=0)
        overall_summary["linear_cka_rgb_common_unique"] = float(cka_tensor[:, 0].mean().item())
        overall_summary["linear_cka_thermal_common_unique"] = float(cka_tensor[:, 1].mean().item())
        overall_summary["linear_cka_common_cross_modal"] = float(cka_tensor[:, 2].mean().item())
        overall_summary["linear_cka_unique_cross_modal"] = float(cka_tensor[:, 3].mean().item())

    return {
        "label": label,
        "overall": overall_summary,
        "per_stage": stage_summaries,
    }


def compare_runs(run_summaries: List[Dict], reference_label: Optional[str] = None) -> Dict:
    """Compare decoupling metrics between runs. Lower common-unique cosine is better."""
    if len(run_summaries) < 2:
        return {}

    label_to_summary = {item["label"]: item for item in run_summaries}
    if reference_label is None:
        reference_label = run_summaries[0]["label"]
    if reference_label not in label_to_summary:
        reference_label = run_summaries[0]["label"]

    reference = label_to_summary[reference_label]["overall"]
    comparisons = {}
    for label, summary in label_to_summary.items():
        if label == reference_label:
            continue
        candidate = summary["overall"]
        comparisons[f"{label} vs {reference_label}"] = {
            "delta_mean_abs_cosine_all": _safe_delta(
                candidate.get("mean_abs_cosine_all"),
                reference.get("mean_abs_cosine_all"),
            ),
            "delta_mean_abs_cosine_rgb": _safe_delta(
                candidate.get("mean_abs_cosine_rgb"),
                reference.get("mean_abs_cosine_rgb"),
            ),
            "delta_mean_abs_cosine_thermal": _safe_delta(
                candidate.get("mean_abs_cosine_thermal"),
                reference.get("mean_abs_cosine_thermal"),
            ),
            "reduction_pct_mean_abs_cosine_all": _safe_reduction_pct(
                candidate.get("mean_abs_cosine_all"),
                reference.get("mean_abs_cosine_all"),
            ),
        }
    return comparisons


def _safe_delta(candidate: Optional[float], reference: Optional[float]) -> Optional[float]:
    if candidate is None or reference is None:
        return None
    return candidate - reference


def _safe_reduction_pct(candidate: Optional[float], reference: Optional[float]) -> Optional[float]:
    if candidate is None or reference is None or reference == 0:
        return None
    return (reference - candidate) / reference * 100.0


def format_summary_table(run_summaries: List[Dict]) -> str:
    header = (
        f"{'Label':<24} {'Samples':>8} {'|cos| RGB':>10} {'|cos| T':>10} "
        f"{'|cos| All':>10} {'|cos| UxM':>10} {'Pearson RGB':>12}"
    )
    lines = [header, "-" * len(header)]
    for summary in run_summaries:
        overall = summary["overall"]
        lines.append(
            f"{summary['label']:<24} "
            f"{overall.get('sample_count', 0):>8d} "
            f"{overall.get('mean_abs_cosine_rgb', float('nan')):>10.4f} "
            f"{overall.get('mean_abs_cosine_thermal', float('nan')):>10.4f} "
            f"{overall.get('mean_abs_cosine_all', float('nan')):>10.4f} "
            f"{overall.get('mean_abs_cosine_unique_cross_modal', float('nan')):>10.4f} "
            f"{overall.get('mean_pearson_rgb', float('nan')):>12.4f}"
        )
    return "\n".join(lines)
