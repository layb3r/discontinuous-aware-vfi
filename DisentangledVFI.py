import torch
import torch.nn as nn
import torch.nn.functional as F

from softsplat import softsplat
from flow_estimator import FlowEstimator
from disentangled_flow import DisentangledFlowRefiner, clear_global_overlap
from helper import ResidualBlock, DownSampleBlock
from modules.backbone import Unet, LiteUnet


class Synthesizer(nn.Module):
    def __init__(self, synth_in_channels=10):
        super(Synthesizer, self).__init__()
        self.ehead_warp = nn.Sequential(
            nn.Conv2d(6, 128, 3, 1, 1),
            ResidualBlock(128, 128),
            DownSampleBlock(128),
        )
        self.synth_proj = nn.Sequential(
            nn.Conv2d(synth_in_channels, 128, 3, 1, 1),
            ResidualBlock(128, 128),
        )
        self.lite_unet = LiteUnet()
        self.unet = Unet()

    def forward(self, warped_img0, warped_img1, synth_input):
        warped_feature = self.ehead_warp(torch.cat([warped_img0, warped_img1], dim=1))
        residual = self.lite_unet(self.synth_proj(synth_input))
        it_feat = warped_feature + residual
        base_output = self.unet(it_feat)
        return base_output, warped_feature, residual, it_feat

class DisentangledVFI(nn.Module):
    def __init__(self, completion_fn=None):
        super(DisentangledVFI, self).__init__()
        self.flow_estimator = FlowEstimator()
        self.refiner = DisentangledFlowRefiner(completion_fn=completion_fn)
        # require grad of refiner is false
        for param in self.refiner.parameters():
            param.requires_grad = False
        self.synthesis_net = Synthesizer(synth_in_channels=10)

    def _as_time_tensor(self, time_period, like_tensor):
        b = like_tensor.shape[0]
        if not torch.is_tensor(time_period):
            t = torch.tensor(time_period, device=like_tensor.device, dtype=like_tensor.dtype)
        else:
            t = time_period.to(device=like_tensor.device, dtype=like_tensor.dtype)

        if t.dim() == 0:
            t = t.view(1, 1, 1, 1).expand(b, 1, 1, 1)
        elif t.dim() == 1:
            if t.shape[0] == 1:
                t = t.view(1, 1, 1, 1).expand(b, 1, 1, 1)
            elif t.shape[0] == b:
                t = t.view(b, 1, 1, 1)
            else:
                raise ValueError("time_period 1D tensor must have shape [1] or [B].")
        elif t.dim() == 4:
            if t.shape[0] == 1 and b > 1:
                t = t.expand(b, -1, -1, -1)
        else:
            raise ValueError("time_period must be scalar, [B], or [B,1,1,1].")

        return t

    def _prepare_disentangle_masks(self, m0, m1):
        m0_clean, m1_clean = clear_global_overlap(m0, m1)
        m0_bin = (F.avg_pool2d(m0_clean.float(), 9, 1, 4) != 0).float()
        m1_bin = (F.avg_pool2d(m1_clean.float(), 9, 1, 4) != 0).float()
        m_xor = (m0_bin != m1_bin).float()
        return m0_clean, m1_clean, m_xor

    def forward(self, img0, img1, m0, m1, time_period, flownet_deep=None, skip_num=0):
        if m0 is None or m1 is None:
            raise ValueError("m0 and m1 are required for disentangled flow refinement.")

        m0_clean, m1_clean, m_xor = self._prepare_disentangle_masks(m0, m1)

        # Use XOR occlusion as flow-estimation guidance mask (coarse-to-fine fade in FlowEstimator).
        _, flow_full = self.flow_estimator.get_flow(
            img0,
            img1,
            time_period,
            flownet_deep=flownet_deep,
            skip_num=skip_num,
            M=m_xor,
        )

        flow_01 = flow_full[:, :2].contiguous()
        flow_10 = flow_full[:, 2:4].contiguous()

        flow_01_ref, flow_10_ref = self.refiner(img0, img1, flow_01, flow_10, m0_clean, m1_clean)

        t = self._as_time_tensor(time_period, img0)
        flow_0t = flow_01_ref * t
        flow_1t = flow_10_ref * (1.0 - t)

        warped_img0 = softsplat.FunctionSoftsplat(
            tenInput=img0,
            tenFlow=flow_0t,
            tenMetric=None,
            strType="average",
        )
        warped_img1 = softsplat.FunctionSoftsplat(
            tenInput=img1,
            tenFlow=flow_1t,
            tenMetric=None,
            strType="average",
        )

        synth_input = torch.cat([warped_img0, warped_img1, flow_0t, flow_1t], dim=1)
        It, warped_feature, synth_residual, it_feat = self.synthesis_net(
            warped_img0,
            warped_img1,
            synth_input,
        )

        return It, warped_img0, warped_img1

    @torch.no_grad()
    def inference(self, img0, img1, m0, m1, time_period, flownet_deep=None, skip_num=0):
        return self.forward(img0, img1, m0, m1, time_period, flownet_deep=flownet_deep, skip_num=skip_num)
