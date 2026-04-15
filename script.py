import torch
from flow_estimator import FlowEstimator
import cv2
import torch.nn.functional as F
from utils import flow_viz
import numpy as np

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
        img = model(img0, img1, 0.5, flow_deep, skip_num)
        img = torch.clamp(img.float(), 0, 1)
    
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flow_img = flow_viz.flow_to_image(img)
    cv2.imwrite('flow.png', flow_img[:, :, ::-1])

if __name__ == "__main__":
    # load_pkl('flow_estimator.pth')
    # load_pkl('UPR_ReDesgin.pkl')
    flow('dataset/0001/I0.png', 'dataset/0001/I1.png')