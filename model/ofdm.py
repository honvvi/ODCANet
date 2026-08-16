import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class OPB(nn.Module):
    """
    Orthogonal Projector Block (OPB).
    Estimates common and unique components from target-reference features.
    """
    def __init__(self, channels):
        super().__init__()
        self.local_map = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.global_map = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1, bias=False)
        )
        self.align_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, target, reference):
        concat = torch.cat([target, reference], dim=1)
        local_score = self.local_map(concat)
        global_score = self.global_map(concat)
        corr = self.act(local_score + global_score)
        common_part = reference * corr
        unique_part = target - common_part
        return unique_part, common_part


class OFDMStage(nn.Module):
    """
    Orthogonal Feature Decoupling Module (OFDM).
    """
    def __init__(self, in_channels, out_channels, stage=0, is_distill=False):
        super().__init__()
        self.stage = stage
        self.is_distill = is_distill

        self.rgb_conv = BasicConv(in_channels, out_channels, 1, padding=0)
        self.t_conv = BasicConv(in_channels, out_channels, 1, padding=0)

        self.rgb_opb = OPB(out_channels)
        self.t_opb = OPB(out_channels)

        self.fuse = nn.Sequential(
            BasicConv(out_channels * 3, out_channels, 1, padding=0),
            BasicConv(out_channels, out_channels, 3, padding=1)
        )

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, rgb, t, f_maps):
        r_feat = self.rgb_conv(rgb)
        t_feat = self.t_conv(t)

        r_unique, r_common = self.rgb_opb(target=r_feat, reference=t_feat)
        t_unique, t_common = self.t_opb(target=t_feat, reference=r_feat)

        shared_feat = (r_common + t_common) / 2.0
        combined = torch.cat([r_unique, t_unique, shared_feat], dim=1)
        out = self.fuse(combined)
        out = self.final_relu(out + r_feat + t_feat)

        f_maps[f"stage{self.stage}_1rgb"] = rgb.detach()
        f_maps[f"stage{self.stage}_2t"] = t.detach()
        f_maps[f"stage{self.stage}_7out"] = out.detach()

        return out, r_unique, r_common, t_unique, t_common, f_maps


class OFDM(nn.Module):
    def __init__(self, FLAGS, ptflops=False, f_maps=False):
        super().__init__()
        self.FLAGS = FLAGS
        self.ptflops = ptflops
        self.f_maps = f_maps

        in_channels = FLAGS['Shared_Encoder']['dims']
        out_channel = FLAGS['HCSAM_Decoder']['embed_dim']
        self.ofdm_stages = nn.ModuleList([
            OFDMStage(in_channels[i], out_channel, stage=i + 1)
            for i in range(4)
        ])

    def forward(self, st_rgb, st_x, f_maps, data_flag):
        if self.ptflops:
            self.FLAGS['ptflops_tensors']['ODCANet']['OFDM'] = {}
            self.FLAGS['ptflops_tensors']['ODCANet']['OFDM']['st_rgb'] = st_rgb
            self.FLAGS['ptflops_tensors']['ODCANet']['OFDM']['st_x'] = st_x
            self.FLAGS['ptflops_tensors']['ODCANet']['OFDM']['f_maps'] = f_maps
            self.FLAGS['ptflops_tensors']['ODCANet']['OFDM']['data_flag'] = data_flag

        en_st1_rgb, en_st2_rgb, en_st3_rgb, en_st4_rgb = st_rgb
        en_st1_x, en_st2_x, en_st3_x, en_st4_x = st_x

        st1, r_unique1, r_common1, t_unique1, t_common1, f_maps = self.ofdm_stages[0](en_st1_rgb, en_st1_x, f_maps)
        st2, r_unique2, r_common2, t_unique2, t_common2, f_maps = self.ofdm_stages[1](en_st2_rgb, en_st2_x, f_maps)
        st3, r_unique3, r_common3, t_unique3, t_common3, f_maps = self.ofdm_stages[2](en_st3_rgb, en_st3_x, f_maps)
        st4, r_unique4, r_common4, t_unique4, t_common4, f_maps = self.ofdm_stages[3](en_st4_rgb, en_st4_x, f_maps)

        st = [st1, st2, st3, st4]
        r_unique = [r_unique1, r_unique2, r_unique3, r_unique4]
        r_common = [r_common1, r_common2, r_common3, r_common4]
        t_unique = [t_unique1, t_unique2, t_unique3, t_unique4]
        t_common = [t_common1, t_common2, t_common3, t_common4]

        if self.f_maps:
            return st, r_unique, r_common, t_unique, t_common, f_maps
        else:
            return st, r_unique, r_common, t_unique, t_common
