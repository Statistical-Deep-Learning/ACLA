# Enhanced Deep Residual Networks for Single Image Super-Resolution

# https://arxiv.org/abs/1707.02921

# EDSR: single-scale SR network (EDSR).

# conv - > resblock -> ··· resblock -> upsample -> conv
url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}
from model import common
from model.deformable_attn import deformable_attention
import torch.nn as nn


# args.n_resblocks 16  32
# args.n_feats 64  256
# args.res_scale 1

# args.scale
# args.rgb_range 255
# args.n_colors 3

def make_model(args, parent=False):
    return EDSR_deform_each(args)


class EDSR_deform_each(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        """
        def default_conv(in_channels, out_channels, kernel_size, bias=True):
            return nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=(kernel_size//2), bias=bias)

        """
        super(EDSR_deform_each, self).__init__()
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        # m_body = [
        #     common.ResBlock(
        #         conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
        #     ) for _ in range(n_resblock)
        # ]
        m_body = []
        for _ in range(n_resblock):
            m_body.append(common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
            m_body.append(deformable_attention(n_feats, n_feats, 4))
            
        m_body.append(conv(n_feats, n_feats, kernel_size))
        
        # define tail module
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)  # n_resblock 个 ResBlock
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

