import copy
import torch
import torch.nn as nn
from .shared_encoder import SharedConvNeXtV2Encoder
from .ofdm import OFDM
from .hcsam import HCSAMDecoder


# Default dataset configurations
DATASET_CONFIGS = {
    'FMB': {'num_classes': 15},
    'PST': {'num_classes': 5},
    'MH': {'num_classes': 9},
}


def build_total_model(FLAGS, ptflops=False, f_maps=False, pretrain=True):
    """
    Build the complete model with the shared encoder, OFDM fusion, and decoder.

    Args:
        FLAGS: Configuration dictionary
        ptflops: Whether to compute FLOPs
        f_maps: Whether to return feature maps
        pretrain: Whether to load pretrained weights

    Returns:
        dict: Dictionary containing the three model components
    """
    # IMPORTANT: Copy FLAGS['Model'] to match original runner's behavior
    M_FLAGS = copy.deepcopy(FLAGS['Model'])

    if ptflops:
        M_FLAGS['ptflops_tensors'] = {'ODCANet': {}}

    # Promote model-specific configs to root level for compatibility
    for key in ['Shared_Encoder', 'OFDM', 'HCSAM_Decoder', 'weights_root', 'input_resize']:
        if key in M_FLAGS:
            FLAGS[key] = M_FLAGS[key]

    # Add dataset configs to M_FLAGS for SegHead
    # Dataset is specified via FLAGS['Data']['dataset_list'] (from config) or FLAGS['dataset'] (from command line)
    if 'dataset_list' in FLAGS.get('Data', {}):
        dataset_list = FLAGS['Data']['dataset_list']
    elif 'dataset' in FLAGS:
        dataset_list = [FLAGS['dataset']]
    else:
        dataset_list = []

    M_FLAGS['dataset_list'] = dataset_list

    for dataset in dataset_list:
        if dataset not in M_FLAGS:
            M_FLAGS[dataset] = {}
        if 'num_classes' not in M_FLAGS[dataset]:
            M_FLAGS[dataset]['num_classes'] = DATASET_CONFIGS.get(dataset, {'num_classes': 15})['num_classes']

    models_to_build = M_FLAGS.get('models_need_build', [])
    if not models_to_build:
        print("Warning: 'models_need_build' is empty. No models will be built.")
        return {}

    built_models = {}
    built_models['Shared_Encoder'] = SharedConvNeXtV2Encoder(
        FLAGS=M_FLAGS, ptflops=ptflops, f_maps=f_maps, pretrain=pretrain
    )
    built_models['OFDM'] = OFDM(FLAGS=M_FLAGS, ptflops=ptflops, f_maps=f_maps)
    built_models['HCSAM_Decoder'] = HCSAMDecoder(
        FLAGS=M_FLAGS, ptflops=ptflops, f_maps=f_maps
    )

    return built_models


def compute_ptflops(FLAGS):
    """Compute FLOPs for the model"""
    built_models = build_total_model(FLAGS=FLAGS, ptflops=True, f_maps=False, pretrain=False)
    if not built_models:
        print("No models were built, skipping ptflops calculation.")
        return

    input_size = FLAGS['Model'].get('FLOPs_size', [1, 3, 480, 640])

    # Get dataset from config or command line
    dataset_list = FLAGS.get('Data', {}).get('dataset_list', [])
    if not dataset_list and 'dataset' in FLAGS:
        dataset_list = [FLAGS['dataset']]
    dataset_name = dataset_list[0] if dataset_list else 'FMB'

    try:
        encoded_features = built_models['Shared_Encoder'](torch.randn(input_size), {}, dataset_name)
        fused_features, _, _, _, _ = built_models['OFDM'](encoded_features, encoded_features, {}, dataset_name)
        logits = built_models['HCSAM_Decoder'](torch.randn(input_size), fused_features, {}, dataset_name)
        print("FLOPs computation completed successfully.")
    except Exception as e:
        print(f"FLOPs computation failed: {e}")

    del built_models
