import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_euler_angles

def mat2euler(mat):
    # mat: Bx3x3
    B = mat.size(0)
    x = torch.atan2(mat[:, 2, 1], mat[:, 2, 2])
    y = torch.atan2(-mat[:, 2, 0], torch.sqrt(mat[:, 2, 1] ** 2 + mat[:, 2, 2] ** 2))
    z = torch.atan2(mat[:, 1, 0], mat[:, 0, 0])
    return torch.stack([x, y, z], dim=1)

def batch_inverse(T):
    """
    Invert a batch of 4x4 transformation matrices.
    Args:
        T: A torch tensor of shape (B, 4, 4), where B is the batch size.
    Returns:
        inv_T: A torch tensor of shape (B, 4, 4), where each 4x4 matrix
               is the inverse of the corresponding input matrix.
    """
    B = T.shape[0]
    inv_T = torch.zeros_like(T)
    for i in range(B):
        inv_T[i] = torch.inverse(T[i])
    return inv_T

def loss_function(pred_mat, pred_trans, target_rot, target_trans):
    pred_euler = matrix_to_euler_angles(pred_mat,'XYZ')
    target_euler = matrix_to_euler_angles(target_rot, 'XYZ')
    rotation_mse = F.mse_loss(pred_euler, target_euler)
    rotation_mae = F.l1_loss(pred_euler, target_euler)
    rotation_rmse = torch.sqrt(rotation_mse)

    pred_trans = pred_trans
    translation_mae =  F.l1_loss(pred_trans, target_trans)
    translation_mse = F.mse_loss(pred_trans, target_trans)
    translation_rmse = torch.sqrt(translation_mse)

    result = {
        'rotation_mae': rotation_mae,
        'rotation_mse': rotation_mse,
        'rotation_rmse': rotation_rmse,
        'translation_mae': translation_mae,
        'translation_mse': translation_mse,
        'translation_rmse': translation_rmse
    }
    return result