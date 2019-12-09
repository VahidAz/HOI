import time
import torch
import pdb


# INFO: scripting doesn't work in Ver 0.4


#@torch.jit.script
def cdist_v1(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    #res = res.clamp_min_(1e-30).sqrt_()
    res = res.clamp(min=1e-30).sqrt_()

    return res


#@torch.jit.script
def cdist_v2(x1, x2):
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
  
    # Compute squared distance matrix using quadratic expansion
    # But be clever and do it with a single matmul call
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))
  
    # Zero out negative values
    res = res.clamp(min=1e-30).sqrt_()

    normalized_res = (res-torch.min(res))/(torch.max(res)-torch.min(res))
    reverse_norm_res = 1 - normalized_res
    
    # return res
    return normalized_res, reverse_norm_res
