# Inputs
I0, I1 # [B, 3, H, W]
F01, F10 # [B, 2, H, W] optical flows (from pretrained flow estimator)
M0, M1 # [B, 1, H, W] temporal instance masks (integers, 0 = bg)
t # scalar in [0, 1]

# Occlusion masks
M_xor = ((M0 > 0) ^ (M1 > 0)).float() # appearing OR disappearing
A     = ((M0 == 0) & (M1 > 0)).float() # appearing in I1
D     = ((M0 > 0) & (M1 == 0)).float() # disappearing in I0

# Targeted background flow completion
F01_masked = F01 * (1 - M_xor)
F10_masked = F10 * (1 - M_xor)
F01_bg = flow_completion(F01_masked, mask=M_xor)
F10_bg = flow_completion(F10_masked, mask=M_xor)

# Corrected flows
F01_corr = F01_bg * (1 - D) # appearing fix already in F01_bg; zero out disappearing
F10_corr = F10_bg * (1 - A) # disappearing fix already in F10_bg; zero out appearing

[can package step 1 -> step 3 as a disentangled flow refinement module used in another VFI pipeline.]

# Step 4: Forward softmax splatting to time t
I0_t = soft_splat(I0, t * F01_corr)          # [B, 3, H, W]
I1_t = soft_splat(I1, (1 - t) * F10_corr)

from I0_t, I1_t -> generate I_base

# Step 5: Persistent overlay conditioning (texture prior)
# Warp original I0 texture only on persistent regions (using original flow)
M_pair = ((M0 > 0) & (M1 > 0)).float()         # persistent overlays
I0_paired = soft_splat(I0, t * F01)

return It