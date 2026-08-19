"""Analyze OFDM common/unique feature decoupling on a test split."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data import DATASET_CONFIGS, build_test_loader
from model import build_total_model
from utils.checkpoint import load_model_part
from utils.feature_analysis import (
    RunAccumulator,
    compare_runs,
    extract_ofdm_features,
    format_summary_table,
    summarize_run,
    update_run_accumulator,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute OFDM feature-level decoupling statistics."
    )
    parser.add_argument("--config", default="./config/train_cfg.yaml")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="MFNet")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label:shared_encoder_path:ofdm_path:hcsam_decoder_path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of test images per run.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference run label used for comparison (defaults to the first --run).",
    )
    parser.add_argument(
        "--compute-cka",
        action="store_true",
        help="Also compute linear CKA (slower, uses more memory).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the JSON file that stores all statistics.",
    )
    return parser.parse_args()


def load_config(path: str, dataset_name: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.setdefault("Data", {})["dataset_list"] = [dataset_name]
    config["dataset"] = dataset_name
    return config


def parse_run_spec(run_spec: str) -> tuple[str, str, str, str]:
    parts = run_spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid --run format: {run_spec}. Expected "
            "label:shared_encoder_path:ofdm_path:hcsam_decoder_path"
        )
    label, shared_encoder_path, ofdm_path, decoder_path = parts
    for path in (shared_encoder_path, ofdm_path, decoder_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return label, shared_encoder_path, ofdm_path, decoder_path


def build_eval_model(config: dict, device: torch.device):
    nets = build_total_model(config, ptflops=False, f_maps=False, pretrain=False)
    for model in nets.values():
        model.eval()
    return nets


def load_run_weights(
    nets,
    shared_encoder_path: str,
    ofdm_path: str,
    decoder_path: str,
    device: torch.device,
):
    load_model_part(nets, "Shared_Encoder", shared_encoder_path, device)
    load_model_part(nets, "OFDM", ofdm_path, device)
    load_model_part(nets, "HCSAM_Decoder", decoder_path, device)


def analyze_single_run(
    nets,
    loader,
    device: torch.device,
    dataset_name: str,
    max_samples: int | None,
    compute_cka: bool,
) -> RunAccumulator:
    run_acc = RunAccumulator()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Feature analysis", leave=False)):
            if max_samples is not None and batch_idx >= max_samples:
                break
            rgb = batch["rgb"].squeeze(0)
            thermal = batch["thermal"].squeeze(0)
            r_unique, r_common, t_unique, t_common = extract_ofdm_features(
                nets, rgb, thermal, device, dataset_name
            )
            update_run_accumulator(
                run_acc,
                r_unique,
                r_common,
                t_unique,
                t_common,
                collect_cka=compute_cka,
            )
    return run_acc


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Feature analysis currently requires CUDA.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = load_config(args.config, args.dataset)
    _, loader = build_test_loader(args.data_root, args.dataset, args.num_workers)

    run_specs = [parse_run_spec(spec) for spec in args.run]
    run_summaries = []

    for label, shared_encoder_path, ofdm_path, decoder_path in run_specs:
        print(f"\nAnalyzing run: {label}")
        nets = build_eval_model(config, device)
        load_run_weights(nets, shared_encoder_path, ofdm_path, decoder_path, device)
        run_acc = analyze_single_run(
            nets,
            loader,
            device,
            args.dataset,
            args.max_samples,
            args.compute_cka,
        )
        run_summaries.append(summarize_run(label, run_acc))
        del nets
        torch.cuda.empty_cache()

    comparison = compare_runs(run_summaries, reference_label=args.reference)
    payload = {
        "dataset": args.dataset,
        "split": "test",
        "max_samples": args.max_samples,
        "metrics_description": {
            "mean_abs_cosine_rgb": "Mean absolute cosine similarity between RGB unique/common features.",
            "mean_abs_cosine_thermal": "Mean absolute cosine similarity between thermal unique/common features.",
            "mean_abs_cosine_all": "Average of RGB and thermal common-unique absolute cosine similarities.",
            "mean_abs_cosine_common_cross_modal": "Absolute cosine similarity between RGB and thermal common features.",
            "mean_abs_cosine_unique_cross_modal": "Absolute cosine similarity between RGB and thermal unique features.",
            "mean_pearson_rgb": "Mean Pearson correlation between RGB unique/common features.",
            "mean_pearson_thermal": "Mean Pearson correlation between thermal unique/common features.",
            "linear_cka_*": "Optional linear CKA; lower common-unique CKA indicates weaker linear alignment.",
        },
        "runs": run_summaries,
        "comparison": comparison,
    }

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print("\n" + "=" * 72)
    print("OFDM Feature Decoupling Summary")
    print("=" * 72)
    print(format_summary_table(run_summaries))
    if comparison:
        print("\nComparison (negative delta / positive reduction => better decoupling):")
        for pair_name, values in comparison.items():
            delta = values.get("delta_mean_abs_cosine_all")
            reduction = values.get("reduction_pct_mean_abs_cosine_all")
            if delta is None or reduction is None:
                continue
            print(
                f"  {pair_name}: delta |cos| all = {delta:+.4f}, "
                f"reduction = {reduction:+.2f}%"
            )
    print(f"\nSaved detailed statistics to: {args.output}")


if __name__ == "__main__":
    main()
