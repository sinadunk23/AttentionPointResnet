import torch
import torch.nn.functional as F

def cycle_consistency(rotation_ab, translation_ab, rotation_ba, translation_ba):
    batch_size = rotation_ab.size(0)
    identity = torch.eye(3, device=rotation_ab.device).unsqueeze(0).repeat(batch_size, 1, 1)
    return F.mse_loss(torch.matmul(rotation_ab, rotation_ba), identity) + F.mse_loss(translation_ab, -translation_ba)