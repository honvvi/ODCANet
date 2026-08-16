import torch


LEGACY_MODEL_NAME_MAP = {
    "Teacher_Encoder": "Shared_Encoder",
    "Teacher_Fusion": "OFDM",
    "Teacher_Decoder": "HCSAM_Decoder",
}
WRAPPER_KEYS = ("state_dict", "model", "net", "network", "module")
LEGACY_STATE_KEY_PREFIX_MAP = {
    "OFDM": (
        ("fusion.", "ofdm_stages."),
    ),
    "HCSAM_Decoder": (
        ("decoder.msaa_skip_1.", "decoder.hcsam_blocks.0."),
        ("decoder.msaa_skip_2.", "decoder.hcsam_blocks.1."),
        ("decoder.msaa_skip_3.", "decoder.hcsam_blocks.2."),
        ("decoder.msaa_skip_4.", "decoder.hcsam_blocks.3."),
        ("decoder.cnn_up_cat_blocks.", "decoder.pub_blocks."),
        ("decoder.hcsam_blocks.0.msaa.", "decoder.hcsam_blocks.0.multi_scale_refiner."),
        ("decoder.hcsam_blocks.1.msaa.", "decoder.hcsam_blocks.1.multi_scale_refiner."),
        ("decoder.hcsam_blocks.2.msaa.", "decoder.hcsam_blocks.2.multi_scale_refiner."),
        ("decoder.hcsam_blocks.3.msaa.", "decoder.hcsam_blocks.3.multi_scale_refiner."),
    ),
}


def _looks_like_state_dict(value):
    return isinstance(value, dict) and any(torch.is_tensor(item) for item in value.values())


def _state_dict_candidates(checkpoint, component_name):
    legacy_name = next(
        (old for old, new in LEGACY_MODEL_NAME_MAP.items() if new == component_name),
        None,
    )
    component_names = [name for name in (component_name, legacy_name) if name]
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint for {component_name} must be a dict, got {type(checkpoint).__name__}"
        )

    candidates = []
    containers = [("checkpoint", checkpoint)]
    containers += [
        (key, checkpoint[key])
        for key in WRAPPER_KEYS
        if isinstance(checkpoint.get(key), dict)
    ]
    for container_name, container in containers:
        for name in component_names:
            if name in container and _looks_like_state_dict(container[name]):
                candidates.append((f"{container_name}.{name}", container[name]))
        if _looks_like_state_dict(container):
            candidates.append((container_name, container))
    return candidates


def _strip_wrapper_prefix(key):
    changed = True
    while changed:
        changed = False
        for prefix in WRAPPER_KEYS:
            prefix = f"{prefix}."
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _normalize_weight_key(key, component_name):
    key = _strip_wrapper_prefix(key)
    for old_name, new_name in LEGACY_MODEL_NAME_MAP.items():
        key = key.replace(old_name, new_name)

    component_prefix = f"{component_name}."
    if component_prefix in key:
        key = key.split(component_prefix, 1)[1]
    key = _strip_wrapper_prefix(key)
    for old_prefix, new_prefix in LEGACY_STATE_KEY_PREFIX_MAP.get(component_name, ()):
        if key.startswith(old_prefix):
            key = new_prefix + key[len(old_prefix):]
    if component_name == "OFDM":
        key = key.replace(".rgb_projector.", ".rgb_opb.")
        key = key.replace(".t_projector.", ".t_opb.")
    return key


def _format_items(items, limit=50):
    return "\n".join(f"  {item}" for item in list(items)[:limit])


def load_compatible_weights(model, checkpoint, component_name):
    candidates = _state_dict_candidates(checkpoint, component_name)
    if not candidates:
        raise ValueError(f"No tensor state_dict found for {component_name}")

    target_state = model.state_dict()
    best = None
    for candidate_name, state_dict in candidates:
        converted = {}
        unused_keys = []
        shape_mismatches = []
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            new_key = _normalize_weight_key(key, component_name)
            if new_key not in target_state:
                unused_keys.append(key)
                continue
            if target_state[new_key].shape != value.shape:
                shape_mismatches.append(
                    (key, new_key, tuple(value.shape), tuple(target_state[new_key].shape))
                )
                continue
            converted[new_key] = value

        candidate = (len(converted), candidate_name, converted, unused_keys, shape_mismatches)
        best = candidate if best is None or candidate[0] > best[0] else best

    matched_count, candidate_name, converted, unused_keys, shape_mismatches = best
    if matched_count == 0:
        raise RuntimeError(f"No weights matched {component_name} after legacy-name conversion")
    if shape_mismatches:
        details = _format_items(
            f"{old_key} -> {new_key}: checkpoint {old_shape}, model {model_shape}"
            for old_key, new_key, old_shape, model_shape in shape_mismatches
        )
        raise RuntimeError(f"Shape mismatch while loading {component_name}:\n{details}")

    incompatible = model.load_state_dict(converted, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"Missing weights for {component_name} after conversion:\n"
            f"{_format_items(incompatible.missing_keys)}"
        )

    legacy_name = next(
        (old for old, new in LEGACY_MODEL_NAME_MAP.items() if new == component_name),
        None,
    )
    component_aliases = [name for name in (component_name, legacy_name) if name]
    legacy_unused = [
        key for key in unused_keys if any(alias in key for alias in component_aliases)
    ]
    if legacy_unused:
        raise RuntimeError(
            f"Unconverted legacy keys remain for {component_name}:\n"
            f"{_format_items(legacy_unused)}"
        )

    print(f"Loaded {component_name} weights from {candidate_name}: {matched_count} tensors")


def load_model_part(nets, component_name, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)
    load_compatible_weights(nets[component_name], checkpoint, component_name)
    nets[component_name] = nets[component_name].to(device)
    print(f"Loaded {component_name} weights")
