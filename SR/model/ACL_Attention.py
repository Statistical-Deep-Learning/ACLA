import torch
import torchvision
import torchvision.ops
from typing import Optional, Tuple
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    out_channels = weight.shape[0]

    use_mask = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, _, _ = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            f"Got offset.shape[1]={offset.shape[1]}, while 2 * weight.size[2] * weight.size[3]={2 * weights_h * weights_w}"
        )

    return torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
        )


class ACL_Attention(nn.Module):
    def __init__(self, channel_num, key_num, dp_conv=False):
        super(ACL_Attention, self).__init__()
        self.key_num = key_num
        self.dp_conv = dp_conv

        self.mask_unit = nn.Conv2d(channel_num, key_num, 1)

        self.conv_offset = nn.Conv2d(channel_num, 2 * key_num, 1)
        self.conv_attn = nn.Conv2d(channel_num, key_num, 1)
        self.conv_refer = nn.Conv2d(channel_num, channel_num, 1)
        self.depthwise_conv = nn.Conv2d(channel_num,
                                        channel_num,
                                        3,
                                        1, (3 - 1) // 2,
                                        groups=channel_num,
                                        bias=False)
        self.softmax =  nn.Softmax(1)
        self.bn = nn.BatchNorm2d(channel_num)
            
    def forward(self, query_layer, key_layer, temp):
        b, c, h, w = query_layer.shape
        key_feature = self.conv_refer(key_layer)
        attn_weights = self.conv_attn(query_layer)
        soft_mask = self.mask_unit(query_layer)
        hard_mask = F.gumbel_softmax(soft_mask, tau=temp, hard=False)
        offset = self.conv_offset(query_layer)
        weight = torch.eye(c)
        weight = weight[:, :, None, None]
        sampled_keys = []
        for i in range(self.key_num):
            sampled_key = torchvision.ops.deform_conv2d(key_feature, offset[:, i : (i + 2), :, :], weight)
            sampled_keys.append(sampled_key.unsqueeze(1))
        sampled_keys = torch.cat((sampled_keys), dim=1)

        attn_weights = attn_weights * hard_mask
        attn_weights = self.softmax(attn_weights)
        attn_weights = attn_weights.unsqueeze(2)
        out = attn_weights * sampled_keys

        out = torch.sum(out, dim=1, keepdim=False)
        if self.dp_conv:
            out = self.depthwise_conv(out)
            out = self.bn(out)
            out = out + query_layer
        return out