import os
import cv2
import imageio
import torch
import torch.nn.functional as F
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def change4save(img):
    img = img.detach().cpu().numpy()
    img = img.squeeze()
    img = img.transpose([1,2,0])
    img = (img*255.0).astype('uint8')
    return img

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

def main(args):

    inference_time = args.time_period # Denotes "t" in our paper
    sample_path = args.sample

    i0_path = os.path.join(sample_path, "I0.png")
    i1_path = os.path.join(sample_path, "I1.png")
    img0 = cv2.imread(i0_path)[:, :, ::-1]
    img1 = cv2.imread(i1_path)[:, :, ::-1]

    img0 = (torch.from_numpy(img0.copy()).permute(2, 0, 1)/255.).unsqueeze(0).to(DEVICE)
    img1 = (torch.from_numpy(img1.copy()).permute(2, 0, 1)/255.).unsqueeze(0).to(DEVICE)

    M0_path = os.path.join(sample_path, "aggregate_masks", "M0.png")
    M1_path = os.path.join(sample_path, "aggregate_masks", "M1.png")

    M0 = cv2.imread(M0_path, cv2.IMREAD_GRAYSCALE)
    M1 = cv2.imread(M1_path, cv2.IMREAD_GRAYSCALE)
    M0 = (torch.from_numpy(M0.copy()).unsqueeze(0).unsqueeze(0)/255.).to(DEVICE) # shape [1, 1, H, W]
    M1 = (torch.from_numpy(M1.copy()).unsqueeze(0).unsqueeze(0)/255.).to(DEVICE) # shape [1, 1, H, W]

    n, c, h, w = img1.shape

    from DisentangledVFI import DisentangledVFI
    model = DisentangledVFI()

    param = torch.load(args.model_file)
    tmp_param = model.state_dict()
    # for kk,v in param.items():
    #     k = kk[7:]
    #     if k in tmp_param:
    #         tmp_param[k] = v
    #     else:
    #         print(k, "  Not load!!!")

    model.load_state_dict(tmp_param)
    # model = model.half()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)
    model.eval().to(DEVICE)

    flow_deep, skip_num = get_deep_skip(img1)
    divisor = 2**(flow_deep+2)

    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)

    with torch.no_grad():
        img, w0, w1 = model.inference(img0, img1, M0, M1, inference_time, flow_deep, skip_num)
        img = torch.clamp(img.float(), 0, 1)
        imageio.imsave(args.save_dir, change4save(img[:, :, :h, :w]))
        imageio.imsave(args.save_dir.replace(".jpg", "_warped0.jpg"), change4save(w0[:, :, :h, :w]))
        imageio.imsave(args.save_dir.replace(".jpg", "_warped1.jpg"), change4save(w1[:, :, :h, :w]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="interpolate for given pair of images")
    parser.add_argument("--sample", type=str, required=True,
            help="sample path")
    parser.add_argument("--time_period", type=float, default=0.5,
            help="time period for interpolated frame")
    
    parser.add_argument("--save_dir", type=str,
            default="./assets/results.jpg",
            help="dir to save interpolated frame")
    
    parser.add_argument('--model_file', type=str,
            default=r"./weights/DisentangledVFI_test_checkpoint.pth",
            help='weight of DisentangledVFI')
    
    args = parser.parse_args()

    main(args)
    # python DisentangledVFI_inference.py --sample ./dataset/0001 --save_dir ./assets/results.jpg --model_file ./weights/DisentangledVFI_test_checkpoint.pth