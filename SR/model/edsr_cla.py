
from model import common
from model.CL_Attention import CLA
import torch.nn as nn


# args.n_resblocks 16  32
# args.n_feats 64  256
# args.res_scale 1

# args.scale
# args.rgb_range 255
# args.n_colors 3

def make_model(args, parent=False):
    return EDSR_CLA(args)


class EDSR_CLA(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        """
        def default_conv(in_channels, out_channels, kernel_size, bias=True):
            return nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=(kernel_size//2), bias=bias)

        """
        super(EDSR_CLA, self).__init__()
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        key_num = 9
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body_1 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for i in range(0, n_resblock/4)
        ]
        m_body_2 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock/4, 2*n_resblock/4)
        ]
        m_body_3 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(2*n_resblock/4, 3*n_resblock/4)
        ]
        m_body_4 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(3*n_resblock/4, 4*n_resblock/4)
        ]
        m_body = [m_body_1, m_body_2, m_body_3, m_body_4]
        # define tail module
        m_tail = [
            conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]


        cla_modules = [
            CLA(n_feats, key_num, 1),
            CLA(n_feats, key_num, 2),
            CLA(n_feats, key_num, 3),
            CLA(n_feats, key_num, 4)
        ]
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(*m_body)
        self.cla = nn.ModuleList(*cla_modules)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        refs = [x]
        for i, stage in enumerate(self.body):
            x = stage(x)
            y = x + self.cla[i](x, refs)
            refs.append(x)
            x = y
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

