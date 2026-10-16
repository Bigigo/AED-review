from functools import partial
import torch
from torch import nn

class BatchNormAct(nn.BatchNorm2d):

    def forward(self, x):
        return super().forward(x).relu_()

class ConvNormAct(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size=1, stride=1, padding=0, norm_layer=None, act_layer=None, norm_kwargs=None, act_kwargs=None):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, bias=False)
        self.bn = BatchNormAct(out_chs, eps=(norm_kwargs or {}).get('eps', 0.001))

    def forward(self, x):
        return self.bn(self.conv(x))

class GlobalPool(nn.Module):

    def forward(self, x):
        return x.mean((2, 3))

def create_classifier(channels, classes, pool_type='avg', drop_rate=0):
    return (GlobalPool(), nn.Dropout(drop_rate), nn.Linear(channels, classes))

class Mixed_5b(nn.Module):

    def __init__(self, conv_block=None):
        super(Mixed_5b, self).__init__()
        conv_block = conv_block or ConvNormAct
        self.branch0 = conv_block(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(conv_block(192, 48, kernel_size=1, stride=1), conv_block(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(conv_block(192, 64, kernel_size=1, stride=1), conv_block(64, 96, kernel_size=3, stride=1, padding=1), conv_block(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), conv_block(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Block35(nn.Module):

    def __init__(self, scale=1.0, conv_block=None):
        super(Block35, self).__init__()
        self.scale = scale
        conv_block = conv_block or ConvNormAct
        self.branch0 = conv_block(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(conv_block(320, 32, kernel_size=1, stride=1), conv_block(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(conv_block(320, 32, kernel_size=1, stride=1), conv_block(32, 48, kernel_size=3, stride=1, padding=1), conv_block(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.act(out)
        return out

class Mixed_6a(nn.Module):

    def __init__(self, conv_block=None):
        super(Mixed_6a, self).__init__()
        conv_block = conv_block or ConvNormAct
        self.branch0 = conv_block(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(conv_block(320, 256, kernel_size=1, stride=1), conv_block(256, 256, kernel_size=3, stride=1, padding=1), conv_block(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Block17(nn.Module):

    def __init__(self, scale=1.0, conv_block=None):
        super(Block17, self).__init__()
        self.scale = scale
        conv_block = conv_block or ConvNormAct
        self.branch0 = conv_block(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(conv_block(1088, 128, kernel_size=1, stride=1), conv_block(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), conv_block(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.act(out)
        return out

class Mixed_7a(nn.Module):

    def __init__(self, conv_block=None):
        super(Mixed_7a, self).__init__()
        conv_block = conv_block or ConvNormAct
        self.branch0 = nn.Sequential(conv_block(1088, 256, kernel_size=1, stride=1), conv_block(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(conv_block(1088, 256, kernel_size=1, stride=1), conv_block(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(conv_block(1088, 256, kernel_size=1, stride=1), conv_block(256, 288, kernel_size=3, stride=1, padding=1), conv_block(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Block8(nn.Module):

    def __init__(self, scale=1.0, no_relu=False, conv_block=None):
        super(Block8, self).__init__()
        self.scale = scale
        conv_block = conv_block or ConvNormAct
        self.branch0 = conv_block(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(conv_block(2080, 192, kernel_size=1, stride=1), conv_block(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)), conv_block(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        self.relu = None if no_relu else nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if self.relu is not None:
            out = self.relu(out)
        return out

class InceptionResnetV2(nn.Module):

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0.0, output_stride=32, global_pool='avg', norm_layer='batchnorm2d', norm_eps=0.001, act_layer='relu'):
        super(InceptionResnetV2, self).__init__()
        self.num_classes = num_classes
        self.num_features = 1536
        assert output_stride == 32
        conv_block = partial(ConvNormAct, padding=0, norm_layer=norm_layer, act_layer=act_layer, norm_kwargs=dict(eps=norm_eps), act_kwargs=dict(inplace=True))
        self.conv2d_1a = conv_block(in_chans, 32, kernel_size=3, stride=2)
        self.conv2d_2a = conv_block(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = conv_block(32, 64, kernel_size=3, stride=1, padding=1)
        self.feature_info = [dict(num_chs=64, reduction=2, module='conv2d_2b')]
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = conv_block(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = conv_block(80, 192, kernel_size=3, stride=1)
        self.feature_info += [dict(num_chs=192, reduction=4, module='conv2d_4a')]
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b(conv_block=conv_block)
        self.repeat = nn.Sequential(*[Block35(scale=0.17, conv_block=conv_block) for _ in range(10)])
        self.feature_info += [dict(num_chs=320, reduction=8, module='repeat')]
        self.mixed_6a = Mixed_6a(conv_block=conv_block)
        self.repeat_1 = nn.Sequential(*[Block17(scale=0.1, conv_block=conv_block) for _ in range(20)])
        self.feature_info += [dict(num_chs=1088, reduction=16, module='repeat_1')]
        self.mixed_7a = Mixed_7a(conv_block=conv_block)
        self.repeat_2 = nn.Sequential(*[Block8(scale=0.2, conv_block=conv_block) for _ in range(9)])
        self.block8 = Block8(no_relu=True, conv_block=conv_block)
        self.conv2d_7b = conv_block(2080, self.num_features, kernel_size=1, stride=1)
        self.feature_info += [dict(num_chs=self.num_features, reduction=32, module='conv2d_7b')]
        self.global_pool, self.head_drop, self.classif = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, drop_rate=drop_rate)

    def forward_features(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.classif(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
