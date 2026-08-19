"""ODCANet evaluation entry point."""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data import DATASET_CONFIGS, build_test_loader
from evaluation.metrics import build_segmentation_metrics
from evaluation.roc import SegmentationBenchmark
from model import build_total_model
from utils.checkpoint import load_model_part
from utils.inference import run_inference


BENCHMARK_ROOT = "./results"
ROC_PIXELS_PER_IMAGE = 8192
WARMUP_ITERATIONS = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ODCANet")
    parser.add_argument("--config", default="./config/train_cfg.yaml")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="FMB")
    parser.add_argument(
        "--shared_encoder_path",
        "--encoder_path",
        dest="shared_encoder_path",
        required=True,
    )
    parser.add_argument("--ofdm_path", "--fusion_path", dest="ofdm_path", required=True)
    parser.add_argument(
        "--hcsam_decoder_path",
        "--decoder_path",
        dest="hcsam_decoder_path",
        required=True,
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:2")
    return parser.parse_args()


def load_config(path, dataset_name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.setdefault("Data", {})["dataset_list"] = [dataset_name]
    config["dataset"] = dataset_name
    return config


def prediction_from_logits(logits):
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [C, H, W], got {tuple(logits.shape)}")
    return logits.argmax(dim=0).cpu().numpy()


def print_results(dataset_name, sample_count, results, label_list, mean_class_count):
    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {sample_count}")
    if mean_class_count != len(label_list):
        print(f"Mean Class Count: {mean_class_count} / {len(label_list)}")
    print("-" * 50)
    print(f"mIoU:           {results['mIoU']:.4f} ({results['mIoU'] * 100:.2f}%)")
    print(f"mAccuracy:      {results['mAccuracy']:.4f} ({results['mAccuracy'] * 100:.2f}%)")
    print(f"Pixel Accuracy: {results['Pixel_Accuracy']:.4f} ({results['Pixel_Accuracy'] * 100:.2f}%)")
    print("-" * 50)
    print("Per-class IoU:")
    for class_name, iou in zip(label_list, results["IoU_per_class"]):
        print(f"  {class_name:15s}: {iou:.4f} ({iou * 100:.2f}%)")
    print("Per-class Accuracy:")
    for class_name, accuracy in zip(label_list, results["Accuracy_per_class"]):
        print(f"  {class_name:15s}: {accuracy:.4f} ({accuracy * 100:.2f}%)")


class ODCANetInferenceWrapper(nn.Module):
    """Expose the three-component model as one module for FLOPs profiling."""

    def __init__(self, nets, dataset_name):
        super().__init__()
        self.shared_encoder = nets["Shared_Encoder"]
        self.ofdm = nets["OFDM"]
        self.hcsam_decoder = nets["HCSAM_Decoder"]
        self.dataset_name = dataset_name

    def forward(self, rgb, thermal):
        feature_maps = {}
        rgb_features = self.shared_encoder(rgb, feature_maps, self.dataset_name)
        thermal_features = self.shared_encoder(thermal, feature_maps, self.dataset_name)
        fused_features, _, _, _, _ = self.ofdm(
            rgb_features, thermal_features, feature_maps, self.dataset_name
        )
        return self.hcsam_decoder(rgb, fused_features, feature_maps, self.dataset_name)


def profile_complexity(nets, rgb, thermal, dataset_name):
    try:
        from thop import profile
    except ImportError as error:
        raise RuntimeError("FLOPs profiling requires the 'thop' package.") from error

    wrapper = ODCANetInferenceWrapper(nets, dataset_name).eval()
    with torch.no_grad():
        flops, params = profile(wrapper, inputs=(rgb, thermal), verbose=False)
    complexity = {
        "params_m": float(params / 1e6),
        "flops_g": float(flops / 1e9),
    }
    print(
        "Complexity: "
        f"Params={complexity['params_m']:.2f}M, "
        f"FLOPs={complexity['flops_g']:.2f}G"
    )
    return complexity


def run_validation_pass(
    loader,
    nets,
    device,
    dataset_name,
    metrics,
    benchmark,
):
    total_latency_ms = 0.0
    total_images = 0
    wall_start = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", leave=False):
            rgb = batch["rgb"].squeeze(0).to(device, non_blocking=True)
            thermal = batch["thermal"].squeeze(0).to(device, non_blocking=True)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            logits = run_inference(rgb, thermal, nets, device, dataset_name)

            end_event.record()
            torch.cuda.synchronize(device)
            total_latency_ms += start_event.elapsed_time(end_event)
            total_images += 1

            prediction = prediction_from_logits(logits)
            target = batch["label"].squeeze(0).numpy()
            metrics.update(prediction, target)
            benchmark.add_roc_batch(logits.unsqueeze(0), batch["label"].to(device))

    wall_time_s = time.perf_counter() - wall_start
    return {
        "inference_time_s": total_latency_ms / 1000.0,
        "latency_ms_per_image": total_latency_ms / total_images,
        "fps": total_images / (total_latency_ms / 1000.0) if total_latency_ms else None,
        "test_runtime_s": wall_time_s,
    }


def evaluate(args):
    dataset_config = DATASET_CONFIGS[args.dataset]
    label_list = dataset_config["label_list"]
    mean_class_count = dataset_config.get("mean_class_count", len(label_list))
    if not torch.cuda.is_available():
        raise RuntimeError("ODCANet evaluation requires CUDA for efficiency measurement.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}, Classes: {len(label_list)}")

    model_config = load_config(args.config, args.dataset)
    nets = build_total_model(model_config, ptflops=False, f_maps=False, pretrain=False)
    load_model_part(nets, "Shared_Encoder", args.shared_encoder_path, device)
    load_model_part(nets, "OFDM", args.ofdm_path, device)
    load_model_part(nets, "HCSAM_Decoder", args.hcsam_decoder_path, device)
    for model in nets.values():
        model.eval()

    dataset, loader = build_test_loader(args.data_root, args.dataset, args.num_workers)
    benchmark_dir = os.path.join(BENCHMARK_ROOT, args.dataset)
    benchmark = SegmentationBenchmark(
        benchmark_dir, label_list, ROC_PIXELS_PER_IMAGE
    )

    first_batch = next(iter(loader))
    sample_rgb = first_batch["rgb"].to(device)
    sample_thermal = first_batch["thermal"].to(device)
    complexity = profile_complexity(nets, sample_rgb, sample_thermal, args.dataset)
    warmup_rgb = sample_rgb.squeeze(0)
    warmup_thermal = sample_thermal.squeeze(0)
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            run_inference(
                warmup_rgb,
                warmup_thermal,
                nets,
                device,
                args.dataset,
            )
    torch.cuda.synchronize(device)

    metrics = build_segmentation_metrics(dataset_config)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    timing = run_validation_pass(
        loader,
        nets,
        device,
        args.dataset,
        metrics,
        benchmark,
    )
    results = metrics.get_metrics()

    print_results(args.dataset, len(dataset), results, label_list, mean_class_count)
    summary = {
        "dataset": args.dataset,
        "device": torch.cuda.get_device_name(device),
        "samples": len(dataset),
        "metrics": {
            "mIoU": float(results["mIoU"]),
            "mAccuracy": float(results["mAccuracy"]),
            "pixel_accuracy": float(results["Pixel_Accuracy"]),
            "per_class_iou": {
                class_name: float(iou)
                for class_name, iou in zip(label_list, results["IoU_per_class"])
            },
            "per_class_accuracy": {
                class_name: float(accuracy)
                for class_name, accuracy in zip(
                    label_list, results["Accuracy_per_class"]
                )
            },
        },
        "complexity": complexity,
        "efficiency": {
            "peak_memory_gb": float(
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            ),
            **timing,
        },
        "roc": benchmark.save_roc(),
    }
    with open(
        os.path.join(benchmark_dir, "efficiency_records.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary["efficiency"], file, indent=2)
    print(
        "Efficiency: "
        f"Latency={summary['efficiency']['latency_ms_per_image']:.3f} ms/image, "
        f"FPS={summary['efficiency']['fps']:.2f}, "
        f"Peak Memory={summary['efficiency']['peak_memory_gb']:.3f} GB"
    )
    print(f"Benchmark results saved to: {benchmark.save_summary(summary)}")
    return results


if __name__ == "__main__":
    evaluate(parse_args())
