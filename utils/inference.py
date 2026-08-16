def run_inference(rgb, thermal, nets, device, dataset_name):
    """Run a single RGB-T sample through the three ODCANet components."""
    rgb = rgb.unsqueeze(0).to(device)
    thermal = thermal.unsqueeze(0).to(device)
    feature_maps = {}

    rgb_features = nets["Shared_Encoder"](rgb, feature_maps, dataset_name)
    thermal_features = nets["Shared_Encoder"](thermal, feature_maps, dataset_name)
    fused_features, _, _, _, _ = nets["OFDM"](
        rgb_features, thermal_features, feature_maps, dataset_name
    )
    return nets["HCSAM_Decoder"](rgb, fused_features, feature_maps, dataset_name).squeeze(0)
