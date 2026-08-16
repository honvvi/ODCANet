import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    def __init__(self, MODEL_FLAGS=None, depths=None, dims=None, pre_train_path=None, pre_train=False):
        super().__init__()
        self.MODEL_FLAGS = MODEL_FLAGS

        drop_path_rate = 0.

        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.downsample_layers = nn.ModuleList([stem])
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

        if pre_train and pre_train_path:
            self.load_pre_train_weights(pre_train_path)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def load_pre_train_weights(self, pre_train_path):
        try:
            pretrain_weights = torch.load(pre_train_path, map_location='cpu')
            if 'model' in pretrain_weights:
                pretrain_weights = pretrain_weights['model']
            miss_keys, unexpected_keys = self.load_state_dict(pretrain_weights, strict=False)
            print(f'ConvNeXtV2 weights loaded, miss: {miss_keys}, unexpected: {unexpected_keys}')
        except Exception as e:
            print(f'Failed to load pretrained weights: {e}')

    def forward(self, x, f_maps):
        if self.MODEL_FLAGS and self.MODEL_FLAGS.get('input_resize') is not None:
            x = nn.functional.interpolate(x, self.MODEL_FLAGS['input_resize'], mode="bilinear", align_corners=True)

        en_stem = self.downsample_layers[0](x)
        en_st1 = self.stages[0](en_stem)

        en_st2 = self.downsample_layers[1](en_st1)
        en_st2 = self.stages[1](en_st2)

        en_st3 = self.downsample_layers[2](en_st2)
        en_st3 = self.stages[2](en_st3)

        en_st4 = self.downsample_layers[3](en_st3)
        en_st4 = self.stages[3](en_st4)

        return x, en_st1, en_st2, en_st3, en_st4, f_maps


class SharedConvNeXtV2Encoder(nn.Module):
    def __init__(self, FLAGS, ptflops=False, f_maps=False, pretrain=True):
        super().__init__()
        self.FLAGS = FLAGS
        self.ptflops = ptflops
        self.f_maps = f_maps

        backbone_type = FLAGS['Shared_Encoder']['backbone']
        backbone_configs = {
            'Convnextv2_atto': {'depths': [2, 2, 6, 2], 'dims': [40, 80, 160, 320]},
            'Convnextv2_femto': {'depths': [2, 2, 6, 2], 'dims': [48, 96, 192, 384]},
            'Convnextv2_pico': {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512]},
            'Convnextv2_nano': {'depths': [2, 2, 8, 2], 'dims': [80, 160, 320, 640]},
            'Convnextv2_tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
            'Convnextv2_base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
            'Convnextv2_large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
        }

        config = backbone_configs.get(backbone_type, backbone_configs['Convnextv2_base'])
        FLAGS['Shared_Encoder']['dims'] = config['dims']
        FLAGS['Shared_Encoder']['depths'] = config['depths']

        weights_root = FLAGS.get('weights_root', '')
        pre_train_path = weights_root + f'Convnextv2/convnextv2_{backbone_type.split("_")[1]}_22k_384_ema.pt'

        self.encoder = ConvNeXtV2(
            MODEL_FLAGS=FLAGS,
            depths=config['depths'],
            dims=config['dims'],
            pre_train_path=pre_train_path if pretrain else None,
            pre_train=pretrain
        )

    def forward(self, rgb, f_maps, data_flag):
        if self.ptflops:
            self.FLAGS['ptflops_tensors']['ODCANet']['Shared_Encoder'] = {}
            self.FLAGS['ptflops_tensors']['ODCANet']['Shared_Encoder']['rgb'] = rgb
            self.FLAGS['ptflops_tensors']['ODCANet']['Shared_Encoder']['f_maps'] = {}
            self.FLAGS['ptflops_tensors']['ODCANet']['Shared_Encoder']['data_flag'] = data_flag

        _, en_st1, en_st2, en_st3, en_st4, f_maps = self.encoder(rgb, f_maps)
        st = [en_st1, en_st2, en_st3, en_st4]

        if self.f_maps:
            return st, f_maps
        else:
            return st
