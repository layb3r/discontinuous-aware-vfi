class FeatPyramid(nn.Module):
    def __init__(self):
        super(FeatPyramid, self).__init__()
        self.conv_stage0 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_stage1 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                    stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_stage2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                    stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))

    def forward(self, img):
        C0 = self.conv_stage0(img)
        C1 = self.conv_stage1(C0)
        C2 = self.conv_stage2(C1)
        return [C0, C1, C2]

class MotionEstimator(nn.Module):
    def __init__(self):
        super(MotionEstimator, self).__init__()
        self.conv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=275, out_channels=160,
                    kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer2 = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=128,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=112,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels=112, out_channels=96,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer5 = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=64,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer6 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=2,
                    kernel_size=3, stride=1, padding=1))


    def forward(self, feat0, feat1, last_feat, last_flow):
        b, _, _,_ = last_feat.shape
        corr_fn=correlation.FunctionCorrelation
        feat0_warp = warp(feat1, last_flow[:b//2]*0.25)
        feat1_warp = warp(feat0, last_flow[b//2:]*0.25)

        volume0 = F.leaky_relu(
                input=corr_fn(tenFirst=feat0, tenSecond=feat0_warp),
                negative_slope=0.1, inplace=False)
        volume1 = F.leaky_relu(
                input=corr_fn(tenFirst=feat1, tenSecond=feat1_warp),
                negative_slope=0.1, inplace=False)
        
        input_feat0 = torch.cat([volume0, feat0, feat0_warp,last_feat[:b//2],last_flow[:b//2]], 1)
        input_feat1 = torch.cat([volume1, feat1, feat1_warp,last_feat[b//2:],last_flow[b//2:]], 1)
        input_feat = torch.cat([input_feat0,input_feat1],dim=0)
        feat = self.conv_layer1(input_feat)
        feat = self.conv_layer2(feat)
        feat = self.conv_layer3(feat)
        feat = self.conv_layer4(feat)
        feat = self.conv_layer5(feat)
        flow= self.conv_layer6(feat)
        return flow, feat
    
class FlowGen(nn.Module):
    def __init__(self):
        super(FlowGen, self).__init__()
        self.feat_pyramid = FeatPyramid()
        self.motion_estimator = MotionEstimator()

    def forward(self, x):
        return x

def get_flow(self, img0, img1, time_period, flownet_deep = None, skip_num = 0):
    if flownet_deep != None:
        self.flownet_deep = flownet_deep
    N, C, H, W = img0.shape
    down_rate = 2**(self.flownet_deep+2)
    if is_half:
        last_flow = torch.zeros((2*N, 2, H//down_rate, W //down_rate)).to(img0.device).half()
        last_feat = torch.zeros((2*N, 64, H//down_rate, W//down_rate)).to(img0.device).half()
    else:
        last_flow = torch.zeros((2*N, 2, H//down_rate, W //down_rate)).to(img0.device)
        last_feat = torch.zeros((2*N, 64, H//down_rate, W//down_rate)).to(img0.device)
    for rate in list(range(self.flownet_deep+1))[::-1]:
        if rate != 0:
            img0_ = resize(img0, 1./(2**rate))
            img1_ = resize(img1, 1./(2**rate))
        else:
            img0_ = img0
            img1_ = img1
        f0 = self.flowgen.feat_pyramid(img0_)
        f1 = self.flowgen.feat_pyramid(img1_)
        last_flow, last_feat = self.flowgen.motion_estimator(f0[-1], f1[-1], last_feat, last_flow)

        last_flow = resize(last_flow, 2.)*2.
        input_img = resize(torch.cat([img0_, img1_], dim=0), 1./2)
        last_flow = self.up_sample(last_flow, input_img)
        last_feat = (resize(last_feat, 2.))*2.
        if rate == skip_num:
            self.flow_base = last_flow.clone()
            last_flow = resize(last_flow, 2**(skip_num+1))*(2**(skip_num+1))
            input_img = torch.cat([img0, img1], dim=0)
            last_flow = self.up_sample(last_flow, input_img)
            self.flow_large = last_flow.clone()
            break

    flow_0t = last_flow[:N] * time_period
    flow_1t = last_flow[N:] * (1 - time_period)
    return torch.cat([flow_0t, flow_1t], dim=1)/4.