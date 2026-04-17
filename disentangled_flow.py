import torch
import torch.nn as nn
from torch.nn import functional as F
from flow_completion import FCNet

def clear_global_overlap(mask1, mask2):
    vals1 = torch.unique(mask1)
    vals2 = torch.unique(mask2)

    # Intersection using isin
    common_vals = vals1[torch.isin(vals1, vals2)]
    common_vals = common_vals[common_vals != 0]

    if common_vals.numel() == 0:
        return mask1, mask2

    result1 = mask1.clone()
    result2 = mask2.clone()

    result1[torch.isin(result1, common_vals)] = 0
    result2[torch.isin(result2, common_vals)] = 0

    return result1, result2

def keep_only_common_labels(mask1, mask2):
    vals1 = torch.unique(mask1)
    vals2 = torch.unique(mask2)

    # Find common labels
    common_vals = vals1[torch.isin(vals1, vals2)]
    common_vals = common_vals[common_vals != 0]

    # If no overlap → return empty masks
    if common_vals.numel() == 0:
        return torch.zeros_like(mask1), torch.zeros_like(mask2)

    result1 = mask1.clone()
    result2 = mask2.clone()

    # Keep only common labels
    result1[~torch.isin(result1, common_vals)] = 0
    result2[~torch.isin(result2, common_vals)] = 0

    return result1, result2

class DisentangledFlowRefiner(nn.Module):
    def __init__(self, completion_fn=None):
        super().__init__()
        self.completion_fn = FCNet('weights/FCNet.pth') if completion_fn is None else completion_fn
        self.completion_fn.cuda()

    @torch.no_grad()
    def forward(self, I0, I1, F01, F10, M0, M1):
        """
        I0, I1: [B, 3, H, W] input images
        F01, F10: [B, 2, H, W] initial bidirectional flows
        M0, M1: [B, 1, H, W] binary masks (0 = bg, 1 = fg)
        """
        M0_bin = M0
        M1_bin = M1
        # remove pixel values existing in both M0_bin and M1_bin
        M0_bin, M1_bin = clear_global_overlap(M0_bin, M1_bin)

        M0_bin = F.avg_pool2d(M0_bin.float(), 9, 1, 4)  # dilate for better FCNet completion
        M0_bin = (M0_bin != 0)
        M1_bin = F.avg_pool2d(M1_bin.float(), 9, 1, 4)  # dilate for better FCNet completion
        M1_bin = (M1_bin != 0)

        # Disentanglement masks.
        M_xor_1c = (M0_bin != M1_bin).float()  # appearing OR disappearing
        # M_xor_1c = F.avg_pool2d(M_xor_1c, 9, 1, 4)  # dilate for better FCNet completion
        # M_xor_1c = (M_xor_1c != 0).float()
        A_1c = ((~M0_bin) & M1_bin).float()  # appearing in I1
        # A_1c = F.avg_pool2d(A_1c, 9, 1, 4)
        # A_1c = (A_1c != 0).float()
        D_1c = (M0_bin & (~M1_bin)).float()  # disappearing in I0
        # D_1c = F.avg_pool2d(D_1c, 9, 1, 4)
        # D_1c = (D_1c != 0).float()

        # M_pair = (M0_bin & M1_bin).float()  # persistent overlays
        # M__pair = F.avg_pool2d(M_pair, 9, 1, 4)
        # M_pair = (M_pair != 0).float()

        # Targeted background flow completion
        fcnet_masks = torch.stack([M_xor_1c, M_xor_1c], dim=1)  # [B, 2, 1, H, W]
        flow_forward = (1 - fcnet_masks[:, :-1]) * (F01.unsqueeze(1)) # shape [B, 1, 2, H, W]
        flow_backward = (1 - fcnet_masks[:, 1:]) * (F10.unsqueeze(1)) # shape [B, 1, 2, H, W]

        fcnet_flows = [flow_forward.cuda(), flow_backward.cuda()]

        # print("Flow shapes: ", flow_forward.shape, flow_backward.shape, fcnet_masks.shape)

        fcnet_inp_flows = self.completion_fn.forward_bidirect_flow(fcnet_flows, fcnet_masks)
        fcnet_inp_flows = self.completion_fn.combine_flow(fcnet_flows, fcnet_inp_flows, fcnet_masks)

        completed_flow_forward, completed_flow_backward = fcnet_inp_flows
        # [B, 1, 2, H, W] -> [B, 2, H, W]
        # completed flow is used for background alignment
        F01_bg = completed_flow_forward.squeeze(1)
        F10_bg = completed_flow_backward.squeeze(1)

        # Corrected / disentangled flows.
        F01_corr = F01_bg * (1 - D_1c) # appearing fix already in F01_bg; zero out disappearing
        F10_corr = F10_bg * (1 - A_1c) # disappearing fix already in F10_bg; zero out appearing

        # optionally masking persistent region
        # F01_corr = F01_corr * (1 - M_pair)
        # F10_corr = F10_corr * (1 - M_pair)

        return F01_corr, F10_corr