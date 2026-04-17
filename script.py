import torch
from flow_estimator import FlowEstimator
import cv2
import torch.nn.functional as F
from utils import flow_viz
import numpy as np
from softsplat import softsplat
from calc_metric import Metrics
from DisentangledVFI import DisentangledVFI

from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
# consume_prefix_in_state_dict_if_present(state_dict, "module.")

# add module. only for parallel training, otherwise keep as usual

def flow_load_test():
    model = FlowEstimator()
    model.load_state_dict(torch.load('flow_estimator.pth')["state_dict"])

    # 2. Total parameters (including non-trainable)
    total_params = sum(p.numel() for p in model.parameters())

    # 3. Trainable parameters only
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")

def load_pkl(path):
    data = torch.load(path, map_location=torch.device('cpu'))
    print(data["state_dict"].keys())

def get_deep_skip(img):
    n, c, h, w = img.shape
    lenss = max(h, w)/1024
    if lenss >= 3:
        flow_deep = 5
        skip_num = 1
    elif lenss >= 1.5:
        flow_deep = 4
        skip_num = 0
    else:
        flow_deep = 3
        skip_num = 0
    return flow_deep, skip_num

# flow inferene test
def flow(img0_path, img1_path, model_path='./weights/flow_estimator.pth'):
    model = FlowEstimator()
    model.to('cuda')
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.to('cuda')

    img0 = cv2.imread(img0_path)[:, :, ::-1]
    img1 = cv2.imread(img1_path)[:, :, ::-1]

    img0 = (torch.from_numpy(img0.copy()).permute(2, 0, 1)/255.).unsqueeze(0).to('cuda')
    img1 = (torch.from_numpy(img1.copy()).permute(2, 0, 1)/255.).unsqueeze(0).to('cuda')

    n, c, h, w = img1.shape
    flow_deep, skip_num = get_deep_skip(img1)
    divisor = 2**(flow_deep+2)

    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)

    with torch.no_grad():
        flow, original_flow = model(img0, img1, 0.5, flow_deep, skip_num)
        flow0t = flow[:, :2].contiguous()
        flow1t = flow[:, 2:4].contiguous()

        # warp img0:
        warped_img0 = softsplat.FunctionSoftsplat(
                tenInput=img0, tenFlow=flow0t,
                tenMetric=None, strType='average') * (255.)
        
        warped_img1 = softsplat.FunctionSoftsplat(
                tenInput=img1, tenFlow=flow1t,
                tenMetric=None, strType='average') * (255.)
        
    
        
    # flow0t = flow0t[0].permute(1, 2, 0).cpu().numpy()
    # flow1t = flow1t[0].permute(1, 2, 0).cpu().numpy()
    # flow0t_img = flow_viz.flow_to_image(flow0t)
    # flow1t_img = flow_viz.flow_to_image(flow1t)
    # cv2.imwrite('flow0t.png', flow0t_img[:, :, ::-1])
    # cv2.imwrite('flow1t.png', flow1t_img[:, :, ::-1])

    # cv2.imwrite('warped_img0.png', warped_img0[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1])
    # cv2.imwrite('warped_img1.png', warped_img1[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1])

def fast_inpaint_normalized(image, mask, num_iters=50):
    img = image.clone()
    mask = mask.float()

    kernel = torch.ones((1, 1, 3, 3), device=img.device)
    kernel[0, 0, 1, 1] = 0  # exclude center

    for _ in range(num_iters):
        # sum of neighbors
        neighbor_sum = F.conv2d(img, kernel.repeat(img.shape[1],1,1,1),
                                padding=1, groups=img.shape[1])

        # count valid neighbors
        valid = 1 - mask
        neighbor_count = F.conv2d(valid, kernel, padding=1)

        avg = neighbor_sum / (neighbor_count + 1e-6)

        img = img * (1 - mask) + avg * mask

    return img

def flow_completion(sample_path, model_path='./weights/flow_estimator.pth'):
    from flow_completion import FCNet
    flow_completion = FCNet('weights/FCNet.pth')
    flow_completion.eval()
    flow_completion.to('cuda')

    model = FlowEstimator()
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.to('cuda')

    img0_path = sample_path + '/I0.png'
    img1_path = sample_path + '/I1.png'
    M0_path = sample_path + '/aggregate_masks' + '/M0.png'
    M1_path = sample_path + '/aggregate_masks' + '/M1.png'

    img0 = cv2.imread(img0_path)[:, :, ::-1]
    img1 = cv2.imread(img1_path)[:, :, ::-1]
    img0 = (torch.from_numpy(img0.copy()).permute(2, 0, 1)/255.).unsqueeze(0).to('cuda') # shape [1, 3, H, W]
    img1 = (torch.from_numpy(img1.copy()).permute(2, 0, 1)/255.).unsqueeze(0).to('cuda') # shape [1, 3, H, W]
    M0 = cv2.imread(M0_path, cv2.IMREAD_GRAYSCALE)
    M1 = cv2.imread(M1_path, cv2.IMREAD_GRAYSCALE)
    M0 = (torch.from_numpy(M0.copy()).unsqueeze(0).unsqueeze(0)/255.).to('cuda') # shape [1, 1, H, W]
    M1 = (torch.from_numpy(M1.copy()).unsqueeze(0).unsqueeze(0)/255.).to('cuda') # shape [1, 1, H, W]
    # M0 OR M1 -> aggregate mask
    M0 = (M0 != 0)
    M1 = (M1 != 0)
    aggregate_mask = (M0 | M1).float() # shape [1, 1, H, W], values 0 or 1
    aggregate_mask = F.avg_pool2d(aggregate_mask, 9, 1, 4)
    aggregate_mask = (aggregate_mask != 0).float() # shape [1, 1, H, W], values 0 or 1

    # save mask visualization
    aggregate_mask_img = (aggregate_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite('aggregate_mask.png', aggregate_mask_img)

    n, c, h, w = img1.shape
    flow_deep, skip_num = get_deep_skip(img1)
    divisor = 2**(flow_deep+2)

    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)

    # mask images:
    img0_mask = img0 * (1 - aggregate_mask) # shape [1, 3, H, W]
    img1_mask = img1 * (1 - aggregate_mask) # shape [1, 3, H, W]

    with torch.no_grad():
        flow, original_flow = model(img0, img1, 0.5, flow_deep, skip_num)
        flow_forward = original_flow[:, :2].contiguous().unsqueeze(1) # shape [1, 1, 2, H, W]
        flow_backward = original_flow[:, 2:4].contiguous().unsqueeze(1)

        flow0t = flow[:, :2].contiguous()
        flow1t = flow[:, 2:4].contiguous()
        # warp img0:
        # warped_img0 = softsplat.FunctionSoftsplat(
        #         tenInput=img0, tenFlow=flow0t,
        #         tenMetric=None, strType='average') * (255.)
        
        # warped_img1 = softsplat.FunctionSoftsplat(
        #         tenInput=img1, tenFlow=flow1t,
        #         tenMetric=None, strType='average') * (255.)

        fcnet_masks = torch.stack([aggregate_mask, aggregate_mask], dim=1).cuda()
        flow_forward = (1 - fcnet_masks[:, :-1]) * flow_forward
        flow_backward = (1 - fcnet_masks[:, 1:]) * flow_backward

        # visualize flow foward and backward before completion
        flow_forward_vis = flow_viz.flow_to_image(flow_forward[0, 0].permute(1, 2, 0).cpu().numpy())
        flow_backward_vis = flow_viz.flow_to_image(flow_backward[0, 0].permute(1, 2, 0).cpu().numpy())
        cv2.imwrite('masked_flow_forward.png', flow_forward_vis[:, :, ::-1])
        cv2.imwrite('masked_flow_backward.png', flow_backward_vis[:, :, ::-1])

        fcnet_flows = [flow_forward.cuda(), flow_backward.cuda()]

        print("Flow shapes: ", flow_forward.shape, flow_backward.shape, fcnet_masks.shape)

        fcnet_inp_flows = flow_completion.forward_bidirect_flow(fcnet_flows, fcnet_masks)
        fcnet_inp_flows = flow_completion.combine_flow(fcnet_flows, fcnet_inp_flows, fcnet_masks)

        completed_flow_forward, completed_flow_backward = fcnet_inp_flows
        # [B, 1, 2, H, W] -> [B, 2, H, W]
        # completed flow is used for background alignment
        completed_flow_forward = completed_flow_forward.squeeze(1) * 0.5
        completed_flow_backward = completed_flow_backward.squeeze(1) * 0.5
        
    # visualize completed flows
    completed_flow_forward_vis = completed_flow_forward[0].permute(1, 2, 0).cpu().numpy()
    completed_flow_backward_vis = completed_flow_backward[0].permute(1, 2, 0).cpu().numpy()
    completed_flow_forward_img = flow_viz.flow_to_image(completed_flow_forward_vis)
    completed_flow_backward_img = flow_viz.flow_to_image(completed_flow_backward_vis)
    cv2.imwrite('completed_flow_forward.png', completed_flow_forward_img[:, :, ::-1])
    cv2.imwrite('completed_flow_backward.png', completed_flow_backward_img[:, :, ::-1])

    # warp img0 and img1 using completed flows
    warped_img0 = softsplat.FunctionSoftsplat(
            tenInput=img0, tenFlow=completed_flow_forward,
            tenMetric=None, strType='average') * (255.)
    warped_img1 = softsplat.FunctionSoftsplat(
            tenInput=img1, tenFlow=completed_flow_backward,
            tenMetric=None, strType='average') * (255.)
    
    cv2.imwrite('warped_img0_completed.png', warped_img0[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1])
    cv2.imwrite('warped_img1_completed.png', warped_img1[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1])

def calc_metrics(pred_path, gt_path):
    metrics = ["psnr", "ssim"]
    print("Building SCORE models...", metrics)
    metric = Metrics(metrics, skip_ref_frames=None, batch_size=1)
    print("Done")
    meta = dict(disImgs=pred_path, refImgs=gt_path)
    avg_score = metric.eval(meta)

    print("AVG Score of %s".center(41, "=") % "VTinker")
    for k, v in avg_score.items():
        print("{:<10} {:<10.3f}".format(k, v))

def test_disentangledvfi():
    # load with strict=False to ignore missing keys related to the synthesis net
    model = DisentangledVFI()
    model.load_state_dict(torch.load('weights/DisentangledVFI_init.pth')["state_dict"], strict=False)
    print("DisentangledVFI loaded successfully with strict=False.")

    # count number of parameters in the model
    total_params = sum(v.numel() for v in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters in DisentangledVFI: {total_params:,}")
    print(f"Trainable parameters in DisentangledVFI: {trainable_params:,}")

    # create a temporal checkpoint:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "note": "This is a temporal checkpoint for testing. It is not intended for actual use.",
            },
         },
         'weights/DisentangledVFI_test_checkpoint.pth',
    )

    # print model.state_dict() keys:
    print("Model state_dict keys:")
    for k in model.state_dict().keys():
        print(k)

def load_disentangledvfi_checkpoint():
    model = DisentangledVFI()
    checkpoint = torch.load('weights/DisentangledVFI_test_checkpoint.pth')
    print(checkpoint["state_dict"].keys())

    tmp_param = model.state_dict()
    for kk,v in checkpoint["state_dict"].items():
        k = kk[7:]
        if k in tmp_param:
            tmp_param[k] = v
        else:
            print(k, "  Not load!!!")
    model.load_state_dict(checkpoint["state_dict"])
    print("Checkpoint loaded successfully.")

if __name__ == "__main__":
    # load_pkl('flow_estimator.pth')
    # load_pkl('UPR_ReDesgin.pkl')
    # flow('dataset/0001/I0.png', 'dataset/0001/I1.png')
    # flow_completion('dataset/0001')
    # calc_metrics('./assets/pred', './assets/gt')
    # test_disentangledvfi()
    load_disentangledvfi_checkpoint()