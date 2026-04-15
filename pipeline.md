FlowEstimator(img0, img1) -> flow, flow_bg (inpaint flow from masking out union of masks of objects in 2 frames to get flow_bg)

M0: mask of objects in img0
M1: mask of objects in img1
M = M0 | M1

warp(img0, flow_01.tt) -> img0_warped
warp(img0, flow_bg.t.(M0 XOR M)) -> img0_warped_bg
warp(img1, flow_10.(1-t)) -> img1_warped

warped_feature = Ehead_warp(concat(img0_warped, img1_warped))

I_copied = I0 M + warped_feature (1 - M)
M_aligned = M0 XOR M (align background in img0)
I_aligned = I_copied (1 - M_aligned) + img0_warped_bg (M_aligned)