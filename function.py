import torch
import torch.nn.functional as F


def calc_mean_std(tensor, eps=1e-5):
    assert (isinstance(tensor, torch.Tensor))
    size = tensor.size()
    assert (len(tensor.size()) == 4)
    b, c = size[:2]
    tensor_var = tensor.view(b, c, -1).var(dim=2) + eps
    tensor_std = tensor_var.sqrt().view(b, c, 1, 1)
    tensor_mean = tensor.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return tensor_mean, tensor_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean) / content_std

    return normalized_feat * style_std + style_mean


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)  # 控制输出张量在0-1之间
    return res
