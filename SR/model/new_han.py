from model import common

import torch.nn as nn
import torch

# this code is from https://github.com/yulunzhang/RCAN
# A part of the code is changed.
# implemented in https://github.com/Jungjaewon/Single_Image_SuperResolution_via_Holistic_Attention_Network


def make_model(args, parent=False):
    return HAN(args)


class Channel_Spatial_Attention_Module(nn.Module):
    def __init__(self, initial_gamma, fix_gamma=False):
        super(Channel_Spatial_Attention_Module, self).__init__()
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.tensor([initial_gamma]).float(), requires_grad=not fix_gamma)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


class Layer_Attention_Module(nn.Module):
    def __init__(self, n_feats, n_resgroups=10, initial_gamma=0, fix_gamma=False):
        super(Layer_Attention_Module, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.tensor([initial_gamma]).float(), requires_grad=not fix_gamma)
        self.n = n_resgroups
        self.c = n_feats
        self.conv = nn.Conv2d(self.n * self.c, self.c, kernel_size=3, padding=1)

    def forward(self, feature_group):
        b, n, c, h, w = feature_group.size()
        feature_group_reshape = feature_group.view(b, n, c * h * w)

        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, c * h * w, n))
        attention_map = self.softmax(attention_map)  # N * N

        attention_feature = torch.bmm(attention_map, feature_group_reshape)  # N * CHW
        b, n, chw = attention_feature.size()
        attention_feature = attention_feature.view(b, n, c, h, w)

        attention_feature = self.gamma * attention_feature + feature_group
        b, n, c, h, w = attention_feature.size()

        return self.conv(attention_feature.view(b, n * c, h, w))



## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = list()
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        return res + x


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class HAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HAN, self).__init__()

        n_resgroups = args.n_resgroups
        self.n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.ModuleList(modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.CSA = Channel_Spatial_Attention_Module(args.initial_gamma_CSAM, args.fix_gamma_CSAM)
        self.LA = Layer_Attention_Module(n_feats, n_resgroups, args.initial_gamma_LAM, args.fix_gamma_LAM)

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.head(x)

        body_results = list()
        body_results.append(x)
        for RG in self.body:
            x = RG(x)
            body_results.append(x)

        feature_LA = self.LA(torch.stack(body_results[1:-1], dim=1))  # b, n * c, h, w
        feature_CSA = self.CSA(body_results[-1])  # # b, c, h, w

        x = self.tail(body_results[0] + feature_CSA + feature_LA)
        x = self.add_mean(x)

        del body_results

        return x

    def load_state_dict(self, state_dict, strict=False):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))