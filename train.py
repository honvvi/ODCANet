"""ODCANet training entry point."""

import argparse
import os

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from data import DATASET_CONFIGS, build_data_loader
from evaluation.metrics import build_segmentation_metrics
from model import build_total_model
from utils.checkpoint import load_model_part


CHECKPOINT_ROOT = "./Checkpoint"


def parse_args():
    parser = argparse.ArgumentParser(description="Train ODCANet")
    parser.add_argument("--config", default="./config/train_cfg.yaml")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="MFNet")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--crop_height", type=int, default=None)
    parser.add_argument("--crop_width", type=int, default=None)
    parser.add_argument("--orth_weight", type=float, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--shared_encoder_path", "--encoder_path", dest="shared_encoder_path")
    parser.add_argument("--ofdm_path", "--fusion_path", dest="ofdm_path")
    parser.add_argument("--hcsam_decoder_path", "--decoder_path", dest="hcsam_decoder_path")
    return parser.parse_args()


def load_config(path, dataset_name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.setdefault("Data", {})["dataset_list"] = [dataset_name]
    config["dataset"] = dataset_name
    return config


def forward_batch(nets, rgb, thermal, dataset_name):
    feature_maps = {}
    rgb_features = nets["Shared_Encoder"](rgb, feature_maps, dataset_name)
    thermal_features = nets["Shared_Encoder"](thermal, feature_maps, dataset_name)
    fused_features, r_unique, r_common, t_unique, t_common = nets["OFDM"](
        rgb_features, thermal_features, feature_maps, dataset_name
    )
    logits = nets["HCSAM_Decoder"](rgb, fused_features, feature_maps, dataset_name)
    return logits, r_unique, r_common, t_unique, t_common


def orthogonality_loss(unique_features, common_features, stage_weights):
    loss = 0.0
    for weight, unique, common in zip(stage_weights, unique_features, common_features):
        stage_loss = torch.mean(torch.abs(F.cosine_similarity(unique, common, dim=1, eps=1e-8)))
        loss = loss + weight * stage_loss
    return loss


def get_orth_weights(args, train_cfg, stage_count=4):
    if args.orth_weight is not None:
        return [float(args.orth_weight)] * stage_count
    config_value = train_cfg.get("orth_weight")
    if config_value is None:
        return [0.0] * stage_count
    if isinstance(config_value, (int, float)):
        return [float(config_value)] * stage_count
    if len(config_value) != stage_count:
        raise ValueError(f"orth_weight must contain {stage_count} values, got {config_value}")
    return [float(value) for value in config_value]


def set_train_mode(nets, training):
    for model in nets.values():
        model.train(training)


def set_learning_rate(optimizer, base_lr, current_iter, total_iters, warm_iters, lr_power):
    if warm_iters > 0 and current_iter < warm_iters:
        lr = base_lr * float(current_iter + 1) / float(warm_iters)
    else:
        progress = (current_iter - warm_iters) / max(total_iters - warm_iters, 1)
        scaled_ratio = (1.0 - 0.1) * (1.0 - progress) + 0.1
        lr = base_lr * (scaled_ratio ** lr_power)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_one_epoch(loader, nets, optimizer, device, dataset_name, args, train_cfg, epoch, total_iters):
    set_train_mode(nets, True)
    total_loss = 0.0
    base_lr = args.learning_rate or train_cfg.get("learning_rate", 3.5e-5)
    warm_epoch = int(train_cfg.get("warm_epoch") or 0)
    lr_power = float(train_cfg.get("lr_power") or 0.9)
    warm_iters = warm_epoch * len(loader)
    orth_weights = get_orth_weights(args, train_cfg)

    for iteration, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)):
        current_iter = (epoch - 1) * len(loader) + iteration
        lr = set_learning_rate(optimizer, base_lr, current_iter, total_iters, warm_iters, lr_power)
        rgb = batch["rgb"].to(device, non_blocking=True)
        thermal = batch["thermal"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, r_unique, r_common, t_unique, t_common = forward_batch(
            nets, rgb, thermal, dataset_name
        )
        ce_loss = F.cross_entropy(logits, labels, ignore_index=255)
        orth_loss = (
            orthogonality_loss(r_unique, r_common, orth_weights)
            + orthogonality_loss(t_unique, t_common, orth_weights)
        )
        loss = ce_loss + orth_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())

    return total_loss / max(len(loader), 1)


def validate(loader, nets, device, dataset_name, dataset_config):
    set_train_mode(nets, False)
    metrics = build_segmentation_metrics(dataset_config)
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            rgb = batch["rgb"].to(device, non_blocking=True)
            thermal = batch["thermal"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits, _, _, _, _ = forward_batch(nets, rgb, thermal, dataset_name)
            total_loss += float(F.cross_entropy(logits, labels, ignore_index=255))
            predictions = logits.argmax(dim=1).cpu().numpy()
            targets = batch["label"].numpy()
            for prediction, target in zip(predictions, targets):
                metrics.update(prediction, target)

    results = metrics.get_metrics()
    results["loss"] = total_loss / max(len(loader), 1)
    return results


def save_component_weights(nets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "Shared_Encoder": os.path.join(output_dir, "shared_encoder.pth"),
        "OFDM": os.path.join(output_dir, "ofdm.pth"),
        "HCSAM_Decoder": os.path.join(output_dir, "hcsam_decoder.pth"),
    }
    for name, path in paths.items():
        torch.save(nets[name].state_dict(), path)
    return paths


def train(args):
    dataset_config = DATASET_CONFIGS[args.dataset]
    train_cfg = load_config(args.config, args.dataset).get("Train", {})
    epochs = args.epochs or int(train_cfg.get("epoch", 300))
    batch_size = args.batch_size or int(train_cfg.get("train_batch_size", 2))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(train_cfg.get("num_workers", 4))
    output_dir = args.output_dir or os.path.join(CHECKPOINT_ROOT, args.dataset)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}, Classes: {dataset_config['num_classes']}")

    model_config = load_config(args.config, args.dataset)
    nets = build_total_model(
        model_config, ptflops=False, f_maps=False, pretrain=not args.no_pretrain
    )
    for component_name, path in (
        ("Shared_Encoder", args.shared_encoder_path),
        ("OFDM", args.ofdm_path),
        ("HCSAM_Decoder", args.hcsam_decoder_path),
    ):
        if path:
            load_model_part(nets, component_name, path, device)
    for model in nets.values():
        model.to(device)

    if (args.crop_height is None) ^ (args.crop_width is None):
        raise ValueError("Provide both --crop_height and --crop_width, or neither.")
    crop_size = (
        (args.crop_height, args.crop_width)
        if args.crop_height is not None
        else None
    )
    train_dataset, train_loader = build_data_loader(
        args.data_root, args.dataset, args.train_split, batch_size, True,
        num_workers, True, True, crop_size
    )
    optimizer = torch.optim.AdamW(
        [parameter for model in nets.values() for parameter in model.parameters()],
        lr=args.learning_rate or train_cfg.get("learning_rate", 3.5e-5),
        weight_decay=0.01,
    )

    total_iters = epochs * len(train_loader)
    print(f"Training samples: {len(train_dataset)}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            train_loader, nets, optimizer, device, args.dataset, args, train_cfg,
            epoch, total_iters
        )
    paths = save_component_weights(nets, output_dir)
    print(f"Saved checkpoint to {output_dir}: {paths}")

    final_miou = None
    eval_after_train = bool(train_cfg.get("Eval_after_train", True))
    if eval_after_train:
        _, eval_loader = build_data_loader(
            args.data_root, args.dataset, args.eval_split, 1, False, num_workers
        )
        eval_results = validate(
            eval_loader, nets, device, args.dataset, dataset_config
        )
        final_miou = float(eval_results["mIoU"])
        print(
            f"mIoU={final_miou:.4f}, "
            f"mAcc={eval_results['mAccuracy']:.4f}"
        )

    if final_miou is None:
        print("Training finished. weights saved.")
    else:
        print(f"Training finished. mIoU: {final_miou:.4f}")


if __name__ == "__main__":
    train(parse_args())
