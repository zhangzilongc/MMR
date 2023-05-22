from functools import partial

import torch
import torch.nn as nn
import math
from torch.nn import functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from .utils import get_2d_sincos_pos_embed


# copy from detectron2
class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# copy from detectron2
class Conv_LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MMR_(nn.Module):
    """
    Build based MAE and Simplr FPN
    MAE: Masked Autoencoders Are Scalable Vision Learners https://github.com/facebookresearch/mae
    Simple FPN: Exploring Plain Vision Transformer Backbones for Object Detection https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 cfg=None, scale_factors=(4.0, 2.0, 1.0), FPN_output_dim=(256, 512, 1024)):
        super().__init__()
        img_size = cfg.DATASET.imagesize
        pretrain_image_size = 224
        self.pretrain_num_patches = (pretrain_image_size // patch_size) * (pretrain_image_size // patch_size)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.pretrain_num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------

        decoder_embed_dim = embed_dim
        self.decoder_FPN_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_FPN_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                                  requires_grad=False)  # fixed sin-cos embedding

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE Simple FPN specifics
        # for scale = 4, 2, 1
        strides = [int(patch_size / scale) for scale in scale_factors]  # [4, 8, 16]

        self.stages = []
        use_bias = False
        for idx, scale in enumerate(scale_factors):
            out_dim = decoder_embed_dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim // 2, kernel_size=2, stride=2),
                    Conv_LayerNorm(decoder_embed_dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(decoder_embed_dim // 2, decoder_embed_dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = decoder_embed_dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim // 2, kernel_size=2, stride=2)]
                out_dim = decoder_embed_dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        FPN_output_dim[idx],
                        kernel_size=1,
                        bias=use_bias,
                        norm=Conv_LayerNorm(FPN_output_dim[idx]),
                    ),
                    Conv2d(
                        FPN_output_dim[idx],
                        FPN_output_dim[idx],
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=Conv_LayerNorm(FPN_output_dim[idx]),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)
        # --------------------------------------------------------------------------
        self.cfg = cfg

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pretrain_num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_FPN_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_FPN_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.decoder_FPN_mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio, ids_shuffle=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if ids_shuffle is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, ids_shuffle=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        if self.patch_embed.num_patches != self.pretrain_num_patches:
            hw = (int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])))
            x = x + get_abs_pos(self.pos_embed[:, 1:, :], hw)
        else:
            x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, ids_shuffle=ids_shuffle)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder_FPN(self, x, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.decoder_FPN_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_ = x_ + self.decoder_FPN_pos_embed[:, 1:, :]

        # FPN stage
        h = w = int(x_.shape[1] ** 0.5)
        decoder_dim = x_.shape[2]

        x = x_.permute(0, 2, 1).view(-1, decoder_dim, h, w)  # (B, channel, h, w)
        results = []

        for idx, stage in enumerate(self.stages):
            stage_feature_map = stage(x)
            results.append(stage_feature_map)

        return {layer: feature for layer, feature in zip(self.cfg.TRAIN.MMR.layers_to_extract_from, results)}

    def forward(self, imgs, mask_ratio=0.75, ids_shuffle=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, ids_shuffle=ids_shuffle)
        reverse_features = self.forward_decoder_FPN(latent, ids_restore)  # [N, L, p*p*3]
        return reverse_features


def MMR_base(**kwargs):
    model = MMR_(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def get_abs_pos(abs_pos, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw

    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    new_abs_pos = F.interpolate(
        abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )

    return new_abs_pos.permute(0, 2, 3, 1).reshape(1, int(h * w), -1)

