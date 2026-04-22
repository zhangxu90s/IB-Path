import torch
import torch.nn as nn
import torch.nn.functional as F

from segm.model.utils import padding, unpadding
from timm.models.layers import trunc_normal_
from einops import rearrange
from segm.model.decoder import DecoderLinear

from torchvision.models import resnet50
import math
import copy


class ConvUnit(nn.Module):
    """
    ConvUnit: Apply spatial convolution on token features.

    Args:
        embed_dim (int): Channel dimension.
    """
    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim

        self.cnn = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False
        )

        self.fuse = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            bias=False
        )


    def forward(self, x: torch.Tensor, H: int, W: int, patch_size: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
            H, W: original image size
            patch_size: patch size

        Returns:
            (B, N, C)
        """
        res = x
        B, N, C = x.shape
        assert C == self.embed_dim, "Channel mismatch"

        H_p, W_p = H // patch_size, W // patch_size
        assert N == H_p * W_p, "Token number mismatch"

        # (B, N, C) -> (B, C, H_p, W_p)
        x = x.transpose(1, 2).reshape(B, C, H_p, W_p).contiguous()

        # Conv block
        x = self.cnn(x)
        x = self.fuse(x)

        # (B, C, H_p, W_p) -> (B, N, C)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x

class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        embed_dim=384,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder_fri = copy.deepcopy(encoder)
        self.encoder = encoder
        self.decoder = decoder

        self.conv = ConvUnit(embed_dim)

    def forward(self, im):
        
        B, H, W = im.size(0), im.size(2), im.size(3)

        # ===== Encoder =====
        x_fri = self.encoder_fri(im, return_features=True)  # [B,197,384]

        num_extra_tokens = 1 + self.encoder_fri.distilled_num
        patch_token_fri = x_fri[:, num_extra_tokens:].contiguous()       # [B,196,384]

        
        # ===== First Decode =====
        fir_patch_masks, _, _ = self.decoder(
            patch_token_fri, (H, W), distilled=False
        )   # [B,n_cls,H_p,W_p]

       
        x = self.encoder(im, return_features=True)  # [B,197,384]
        patch_token = x[:, num_extra_tokens:]       # [B,196,384]

        
        mask_weight = torch.sigmoid(
            fir_patch_masks.mean(1, keepdim=True)
        ).flatten(2).transpose(1,2)

        
        patch_token = patch_token * mask_weight


        
        patch_token = self.conv(patch_token, H, W, self.patch_size)

        
        fin_patch_masks, _, _ = self.decoder(
            patch_token, (H, W), distilled=False
        )
        
        
        # ===== Patch → Pixel =====
        fin_pix_masks = F.interpolate(
            fin_patch_masks, size=(H, W), mode="bilinear", align_corners=False
        )
        fin_pix_masks = unpadding(fin_pix_masks, (H, W))


        return {
            "fin_patch_masks": fin_patch_masks,
            "fin_pix_masks": fin_pix_masks,
            "fir_patch_masks": fir_patch_masks
        }        

