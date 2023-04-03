import numpy as np

import torch
from torch.nn import functional as F

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)  # np.arrange(256)
    omega /= embed_dim / 2.  # np.arrange(256) / 256 normalize
    omega = 1. / 10000**omega  # (D/2,)  # 1 / 10000**(np.arrange(256) / 256)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output


# copy from reverse distillation
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([fs_list[0].shape[0], out_size, out_size])
    else:
        anomaly_map = np.zeros([fs_list[0].shape[0], out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]

        a_map = 1 - F.cosine_similarity(fs, ft)  # cosine similarity alongside the dimension 1
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map.squeeze(1).cpu().detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def each_patch_loss_function(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_tem = a[item].permute(0, 2, 3, 1)
        b_tem = b[item].permute(0, 2, 3, 1)

        loss += torch.mean(1 - cos_loss(a_tem.contiguous().view(-1, a_tem.shape[-1]),
                                        b_tem.contiguous().view(-1, b_tem.shape[-1])))
    return loss


def mmr_adjust_learning_rate(optimizer, epoch, cfg):
    """cosine lr"""
    if epoch < cfg.TRAIN_SETUPS.warmup_epochs:
        lr = cfg.TRAIN_SETUPS.learning_rate * epoch / cfg.TRAIN_SETUPS.warmup_epochs
    else:
        if epoch + 1 == 120:
            cfg.TRAIN_SETUPS.learning_rate /= 10
        elif epoch + 1 == 160:
            cfg.TRAIN_SETUPS.learning_rate /= 10
        lr = cfg.TRAIN_SETUPS.learning_rate
    # lr *= 0.5 * (1. + math.cos(math.pi * epoch / cfg.TRAIN_SETUPS.epochs))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
