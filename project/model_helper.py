import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import ScriptModule, script_method, script

from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op

@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler(g, input, grid, interpolation_mode, padding_mode, align_corners=False):
    '''
    torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
    Need convert interpolation_mode, padding_mode ? NO for simpler at now !!!
    '''
    return g.op('onnxservice::grid_sampler', input, grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners)

register_op('grid_sampler', grid_sampler, '', 11)



arguments_strModel = 'sintel-final'
SpyNet_model_dir = './models'  # The directory of SpyNet's weights
import pdb

def normalize(input):
    R = (input[:, 0:1, :, :] - 0.485) / 0.229
    G = (input[:, 1:2, :, :] - 0.456) / 0.224
    B = (input[:, 2:3, :, :] - 0.406) / 0.225
    return torch.cat([R, G, B], 1)

def denormalize(input):
    R = (input[:, 0:1, :, :] * 0.229) + 0.485
    G = (input[:, 1:2, :, :] * 0.224) + 0.456
    B = (input[:, 2:3, :, :] * 0.225) + 0.406
    return torch.cat([R, G, B], 1)

@script
def warp(image, flow):
    # image -- torch.Size([1, 3, 32, 56])
    B = image.shape[0]
    H = image.shape[2]
    W = image.shape[3]

    grid_h = torch.arange(-1.0, 1.0, 2.0/W, device=flow.device).view(1, 1, 1, W).expand(B, -1, H, -1)
    grid_v = torch.arange(-1.0, 1.0, 2.0/H, device=flow.device).view(1, 1, H, 1).expand(B, -1, -1, W)
    warp_grid = torch.cat([grid_h, grid_v], 1)

    flow = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)

    g = (warp_grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(input=image, grid=g, mode='bilinear', padding_mode='zeros', align_corners=True)

# Optical Flow Estimation Using a Spatial Pyramid Network
class SpyNet(nn.Module):
    def __init__(self):
        super(SpyNet, self).__init__()

        class Basic(nn.Module):
            def __init__(self):
                super(Basic, self).__init__()
                self.moduleBasic = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            def forward(self, input):
                return self.moduleBasic(input)

        self.moduleBasic = nn.ModuleList()
        for i in range(4):
            self.moduleBasic.append(Basic())

        self.load_state_dict(torch.load("models/SpyNet.pth"), strict=False)
        # 'models/SpyNet.pth -- /models/network-sintel-final.pytorch'

    def forward(self, first, second):
        '''
        !!! FORCE:  first, second->height % 16 == 0 && width %16 == 0 !!! 

        # torch.Size([1, 3, 256, 448])
        # torch.Size([1, 3, 256, 448])
        '''

        first = [first]
        second = [second]
        for i in range(3):
            first.insert(0, F.avg_pool2d(first[0], kernel_size=2, stride=2))
            second.insert(0, F.avg_pool2d(second[0], kernel_size=2, stride=2))

        B = first[0].size(0)
        H = first[0].size(2)
        W = first[0].size(3)
        flow = torch.zeros(B, 2, H//2, W//2, device=first[0].device)
        for i in range(4):
            upflow = F.interpolate(flow, scale_factor=2.0, mode='bilinear', align_corners=True) * 2.0

            basic = torch.cat([first[i], warp(second[i], upflow), upflow], 1)
            flow = self.moduleBasic[i](basic) + upflow

        # flow.size() -- torch.Size([1, 2, 256, 448])
        return flow

class ResNet(nn.Module):
    """
    Three-layers ResNet/ResBlock
    reference: https://blog.csdn.net/chenyuping333/article/details/82344334
    """
    def __init__(self, task):
        super(ResNet, self).__init__()
        self.task = task
        self.conv_3x2_64_9x9 = nn.Conv2d(in_channels=3 * 2, out_channels=64, kernel_size=9, padding=8 // 2)
        self.conv_3x7_64_9x9 = nn.Conv2d(in_channels=3 * 7, out_channels=64, kernel_size=9, padding=8 // 2)
        self.conv_64_64_9x9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, padding=8 // 2)
        self.conv_64_64_1x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv_64_3_1x1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def ResBlock(self, x, aver):
        if self.task == 'slow':
            x = F.relu(self.conv_3x2_64_9x9(x))
            x = F.relu(self.conv_64_64_1x1(x))
        elif self.task in ['clean']:
            x = F.relu(self.conv_3x7_64_9x9(x))
            x = F.relu(self.conv_64_64_1x1(x))
        elif self.task in ['zoom']:
            x = F.relu(self.conv_3x7_64_9x9(x))
            x = F.relu(self.conv_64_64_9x9(x))
            x = F.relu(self.conv_64_64_1x1(x))
        else:
            raise NameError('Only support: [slow, clean, zoom]')
        x = self.conv_64_3_1x1(x) + aver
        return x

    def forward(self, frames):
        # frames.size() -- torch.Size([1, 7, 3, 256, 448])

        aver = frames.mean(dim=1)
        # x = frames[:, 0, :, :, :]
        # for i in range(1, frames.size(1)):
        #     x = torch.cat((x, frames[:, i, :, :, :]), dim=1)

        x =frames.view(frames.size(0), frames.size(1) * frames.size(2), frames.size(3), frames.size(4))

        # x.size() -- torch.Size([1, 21, 256, 448])
        result = self.ResBlock(x, aver)

        # result.size() -- torch.Size([1, 3, 256, 448])
        return result


class TOFlow(nn.Module):
    def __init__(self, task):
        super(TOFlow, self).__init__()
        self.task = task
        self.SpyNet = SpyNet()
        self.ResNet = ResNet(task=self.task)

    def forward(self, frames):
        """
        :param frames: [batch_size=1, img_num, n_channels=3, h, w]
        :return:
        """
        # frames.size()
        # torch.Size([7, 3, 256, 448])
        frames = frames.unsqueeze(0)

        for i in range(frames.size(1)):
            frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])

        flows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4), device=frames.device)
        warpframes = torch.zeros(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4), device=frames.device)

        if self.task == 'slow':
            process_index = [0, 1]
            flows[:, 1, :, :, :] = self.SpyNet(frames[:, 0, :, :, :], frames[:, 1, :, :, :]) / 2
            flows[:, 0, :, :, :] = self.SpyNet(frames[:, 1, :, :, :], frames[:, 0, :, :, :]) / 2
        elif self.task in ['clean', 'zoom']:
            process_index = [0, 1, 2, 4, 5, 6]
            for i in process_index:
                flows[:, i, :, :, :] = self.SpyNet(frames[:, 3, :, :, :], frames[:, i, :, :, :])
            warpframes[:, 3, :, :, :] = frames[:, 3, :, :, :]
        else:
            raise NameError('Only support: [slow, clean, zoom]')

        for i in process_index:
            warpframes[:, i, :, :, :] = warp(frames[:, i, :, :, :], flows[:, i, :, :, :])
        # warpframes: [batch_size=1, img_num=7, n_channels=3, height=256, width=448]
        # (Pdb) warpframes.size() -- torch.Size([1, 7, 3, 256, 448])

        image = self.ResNet(warpframes)
        image = denormalize(image)
        # image -- torch.Size([1, 3, 256, 448])

        return image.clamp(0.0, 1.0)


class CleanFlow(nn.Module):
    def __init__(self):
        super(CleanFlow, self).__init__()
        self.SpyNet = SpyNet()
        self.ResNet = ResNet(task="clean")

    def forward(self, frames):
        """
        :param frames: [batch_size=1, img_num, n_channels=3, h, w]
        :return:
        """
        # frames.size()
        # torch.Size([7, 3, 256, 448])
        # for i in range(frames.size(1)):
        frames = frames.unsqueeze(0)

        for i in range(7):
            frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])

        flows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4), device=frames.device)
        warpframes = torch.zeros(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4), device=frames.device)

        process_index = [0, 1, 2, 4, 5, 6]
        for i in process_index:
            flows[:, i, :, :, :] = self.SpyNet(frames[:, 3, :, :, :], frames[:, i, :, :, :])
        warpframes[:, 3, :, :, :] = frames[:, 3, :, :, :]

        for i in process_index:
            warpframes[:, i, :, :, :] = warp(frames[:, i, :, :, :], flows[:, i, :, :, :])
        # warpframes: [batch_size=1, img_num=7, n_channels=3, height=256, width=448]
        # (Pdb) warpframes.size() -- torch.Size([1, 7, 3, 256, 448])

        image = self.ResNet(warpframes)
        image = denormalize(image)
        # image -- torch.Size([1, 3, 256, 448])

        return image.clamp(0.0, 1.0)

class SlowFlow(nn.Module):
    def __init__(self):
        super(SlowFlow, self).__init__()
        self.SpyNet = SpyNet()
        self.ResNet = ResNet(task="slow")

    def forward(self, frames):
        """
        :param frames: [batch_size=1, img_num, n_channels=3, h, w]
        :return:
        """
        # frames.size()
        # torch.Size([2, 3, 256, 448])
        # for i in range(frames.size(1)):
        frames = frames.unsqueeze(0)

        for i in range(2):
            frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])

        flows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4), device=frames.device)
        warpframes = torch.zeros(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4), device=frames.device)

        process_index = [0, 1]
        flows[:, 1, :, :, :] = self.SpyNet(frames[:, 0, :, :, :], frames[:, 1, :, :, :]) / 2
        flows[:, 0, :, :, :] = self.SpyNet(frames[:, 1, :, :, :], frames[:, 0, :, :, :]) / 2

        for i in process_index:
            warpframes[:, i, :, :, :] = warp(frames[:, i, :, :, :], flows[:, i, :, :, :])
        # warpframes: [batch_size=1, img_num=7, n_channels=3, height=256, width=448]
        # (Pdb) warpframes.size() -- torch.Size([1, 7, 3, 256, 448])

        image = self.ResNet(warpframes)
        image = denormalize(image)
        # image -- torch.Size([1, 3, 256, 448])

        return image.clamp(0.0, 1.0)


class ZoomFlow(nn.Module):
    def __init__(self):
        super(ZoomFlow, self).__init__()
        self.SpyNet = SpyNet()
        self.ResNet = ResNet(task="zoom")

    def forward(self, frames):
        """
        :param frames: [batch_size=1, img_num, n_channels=3, h, w]
        :return:
        """
        # frames.size()
        # torch.Size([7, 3, 256, 448])
        # for i in range(frames.size(1)):
        frames = frames.unsqueeze(0)

        for i in range(7):
            frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])

        flows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4), device=frames.device)
        warpframes = torch.zeros(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4), device=frames.device)

        process_index = [0, 1, 2, 4, 5, 6]
        for i in process_index:
            flows[:, i, :, :, :] = self.SpyNet(frames[:, 3, :, :, :], frames[:, i, :, :, :])
        warpframes[:, 3, :, :, :] = frames[:, 3, :, :, :]

        for i in process_index:
            warpframes[:, i, :, :, :] = warp(frames[:, i, :, :, :], flows[:, i, :, :, :])
        # warpframes: [batch_size=1, img_num=7, n_channels=3, height=256, width=448]
        # (Pdb) warpframes.size() -- torch.Size([1, 7, 3, 256, 448])

        image = self.ResNet(warpframes)
        image = denormalize(image)
        # image -- torch.Size([1, 3, 256, 448])

        return image.clamp(0.0, 1.0)
