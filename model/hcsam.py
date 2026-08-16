import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def pad(x, y):
    """Pad tensor y to match dimensions of x"""
    _, _, H_x, W_x = x.size()
    _, _, H_y, W_y = y.size()
    pad_h = (H_x - H_y) // 2
    pad_w = (W_x - W_y) // 2
    pad_h2 = pad_h + (H_x - H_y) % 2
    pad_w2 = pad_w + (W_x - W_y) % 2
    return F.pad(y, (pad_w, pad_w2, pad_h, pad_h2), "constant", 0)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='pl', groups=4, dyscope=True):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear', align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


def dy_yp(type, in_dim, scale):
    if type == 'DySample':
        return DySample(in_dim, scale, 'lp', groups=4, dyscope=False)
    if type == 'DySample+':
        return DySample(in_dim, scale, 'lp', groups=4, dyscope=True)
    if type == 'DySample-S':
        return DySample(in_dim, scale, 'pl', groups=4, dyscope=False)
    if type == 'DySample-S+':
        return DySample(in_dim, scale, 'pl', groups=4, dyscope=True)
    return None


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MultiScaleDynamicRefiner(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, alpha=4):
        super().__init__()
        self.mid_channels = max(1, in_channels // alpha)
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, self.mid_channels, 1, bias=False), nn.BatchNorm2d(self.mid_channels), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=1, bias=False), nn.BatchNorm2d(self.mid_channels), nn.ReLU(inplace=True))
        self.branch5 = nn.Sequential(nn.Conv2d(self.mid_channels, self.mid_channels, 5, padding=2, bias=False), nn.BatchNorm2d(self.mid_channels), nn.ReLU(inplace=True))
        self.branch7 = nn.Sequential(nn.Conv2d(self.mid_channels, self.mid_channels, 7, padding=3, bias=False), nn.BatchNorm2d(self.mid_channels), nn.ReLU(inplace=True))
        self.fuse_matrix = nn.Conv2d(self.mid_channels * 3, self.mid_channels * 3, 1)
        self.softmax = nn.Softmax(dim=1)
        self.restore = nn.Sequential(nn.Conv2d(self.mid_channels, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels))
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x_r = self.reduce(x)
        b3 = self.branch3(x_r)
        b5 = self.branch5(x_r)
        b7 = self.branch7(x_r)
        concat_feats = torch.cat([b3, b5, b7], dim=1)
        weights = self.softmax(self.fuse_matrix(concat_feats))
        w3, w5, w7 = torch.chunk(weights, 3, dim=1)
        fused = b3 * w3 + b5 * w5 + b7 * w7
        return self.final_relu(self.restore(fused) + residual)


class HCSAM(nn.Module):
    """Hierarchical Cross-Scale Aggregation Module."""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.up = dy_yp('DySample-S+', in_channels_list[1], 2)
        self.fuse_conv = nn.Sequential(nn.Conv2d(sum(in_channels_list), out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.multi_scale_refiner = MultiScaleDynamicRefiner(out_channels)

    def forward(self, target_feature, *other_features):
        target_h, target_w = target_feature.shape[2], target_feature.shape[3]
        fused_features = [target_feature]
        for feat in other_features:
            h, w = feat.shape[2], feat.shape[3]
            if h != target_h or w != target_w:
                if h > target_h:
                    feat_resized = F.adaptive_avg_pool2d(feat, (target_h, target_w))
                else:
                    feat_resized = self.up(feat)
                    if feat_resized.shape[2:] != target_feature.shape[2:]:
                        feat_resized = pad(target_feature, feat_resized)
                fused_features.append(feat_resized)
            else:
                fused_features.append(feat)
        x = torch.cat(fused_features, dim=1)
        x = self.fuse_conv(x)
        return self.multi_scale_refiner(x)


class PUB(nn.Module):
    """Progressive Upsampling Block."""
    def __init__(self, MODEL_FLAGS, de_up_style, embed_dim, scale, i):
        super().__init__()
        self.de_up_style = de_up_style
        if self.de_up_style == 'dysample':
            self.up = dy_yp('DySample-S+', embed_dim, scale)
        self.refine = nn.Sequential(BasicConv(embed_dim * 2, embed_dim, 1), BasicConv(embed_dim, embed_dim, 3, padding=1))

    def forward(self, lr, hr, f_maps):
        if self.de_up_style == 'dysample':
            lr = self.up(lr)
            if lr.shape[2:] != hr.shape[2:]:
                lr = pad(hr, lr)
        else:
            lr = F.interpolate(lr, size=hr.shape[2:], mode='bilinear', align_corners=False)
        f = torch.cat([lr, hr], dim=1)
        f = self.refine(f)
        return f, f_maps


class DYSegConv(nn.Module):
    def __init__(self, in_dim, embed_dim, up_scale):
        super().__init__()
        self.seg_conv = nn.Sequential(nn.Conv2d(in_dim, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True), dy_yp('DySample-S+', embed_dim, up_scale))

    def forward(self, x):
        return self.seg_conv(x)


class SegHead(nn.Module):
    def __init__(self, MODEL_FLAGS, in_dim, embed_dim, seg_head_up_style, use_drop_out, up_scale):
        super().__init__()
        self.MODEL_FLAGS = MODEL_FLAGS
        self.seg_head_up_style = seg_head_up_style
        self.use_drop_out = use_drop_out
        if self.use_drop_out:
            self.dropout = nn.Dropout2d(0.1)
        self.seg_convs = nn.ModuleDict()
        self.seg_outs = nn.ModuleDict()
        for data_type in MODEL_FLAGS['dataset_list']:
            if seg_head_up_style == 'linear':
                self.seg_convs[data_type] = nn.Identity()
            elif seg_head_up_style == 'dysample':
                self.seg_convs[data_type] = DYSegConv(in_dim, embed_dim, up_scale)
            self.seg_outs[data_type] = nn.Conv2d(embed_dim, MODEL_FLAGS[data_type]['num_classes'], kernel_size=1)
        self.apply(_init_weights)
        for data_type in MODEL_FLAGS['dataset_list']:
            self.seg_outs[data_type].weight.data.mul_(1.)
            self.seg_outs[data_type].bias.data.mul_(1.)

    def forward(self, f, a1, f_maps, data_flag):
        f = self.seg_convs[data_flag](f)
        if self.use_drop_out:
            f = self.dropout(f)
        out = self.seg_outs[data_flag](f)
        if self.seg_head_up_style == 'linear':
            out = F.interpolate(out, size=a1.size()[-2:], mode='bilinear', align_corners=True)
        return out, f_maps


class HCSAMDecoderBody(nn.Module):
    def __init__(self, MODEL_FLAGS, embed_dim, de_up_style, seg_head_up_style, use_drop_out):
        super().__init__()
        self.MODEL_FLAGS = MODEL_FLAGS
        self.hcsam_blocks = nn.ModuleList([
            HCSAM([embed_dim, embed_dim], embed_dim),
            HCSAM([embed_dim, embed_dim, embed_dim], embed_dim),
            HCSAM([embed_dim, embed_dim, embed_dim], embed_dim),
            HCSAM([embed_dim, embed_dim], embed_dim),
        ])
        self.pub_blocks = nn.ModuleList([
            PUB(MODEL_FLAGS, de_up_style, embed_dim, 2, i) for i in range(3)
        ])
        self.seg_head = SegHead(MODEL_FLAGS=MODEL_FLAGS, in_dim=embed_dim, embed_dim=embed_dim, seg_head_up_style=seg_head_up_style, use_drop_out=use_drop_out, up_scale=4)
        self.apply(_init_weights)

    def forward(self, x, st, f_maps, data_flag):
        en_st1, en_st2, en_st3, en_st4 = st
        st1_fused = self.hcsam_blocks[0](en_st1, en_st2)
        st2_fused = self.hcsam_blocks[1](en_st2, en_st1, en_st3)
        st3_fused = self.hcsam_blocks[2](en_st3, en_st2, en_st4)
        st4_fused = self.hcsam_blocks[3](en_st4, en_st3)
        en_st_cat_f_3, f_maps = self.pub_blocks[2](st4_fused, st3_fused, f_maps)
        en_st_cat_f_2, f_maps = self.pub_blocks[1](en_st_cat_f_3, st2_fused, f_maps)
        en_st_cat_f_1, f_maps = self.pub_blocks[0](en_st_cat_f_2, st1_fused, f_maps)
        out, f_maps = self.seg_head(en_st_cat_f_1, x, f_maps, data_flag)
        return out, f_maps


class HCSAMDecoder(nn.Module):
    def __init__(self, FLAGS, ptflops=False, f_maps=False):
        super().__init__()
        self.FLAGS = FLAGS
        self.ptflops = ptflops
        self.f_maps = f_maps
        self.decoder = HCSAMDecoderBody(MODEL_FLAGS=FLAGS, embed_dim=FLAGS['HCSAM_Decoder']['embed_dim'], de_up_style=FLAGS['HCSAM_Decoder']['de_up_style'],
                               seg_head_up_style=FLAGS['HCSAM_Decoder']['seg_head_up_style'], use_drop_out=FLAGS['HCSAM_Decoder']['use_drop_out'])

    def forward(self, x, st, f_maps, data_flag):
        if self.ptflops:
            self.FLAGS['ptflops_tensors']['ODCANet']['HCSAM_Decoder'] = {}
            self.FLAGS['ptflops_tensors']['ODCANet']['HCSAM_Decoder']['x'] = x
            self.FLAGS['ptflops_tensors']['ODCANet']['HCSAM_Decoder']['st'] = st
            self.FLAGS['ptflops_tensors']['ODCANet']['HCSAM_Decoder']['f_maps'] = f_maps
            self.FLAGS['ptflops_tensors']['ODCANet']['HCSAM_Decoder']['data_flag'] = data_flag
        out, f_maps = self.decoder(x, st, f_maps, data_flag)
        if self.f_maps:
            return out, f_maps
        return out
