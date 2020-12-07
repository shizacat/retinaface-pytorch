"""Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from lib.config import ConfigNet


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list: list, out_channels: int, ull=True):
        """
        :params in_channels_list - List of int, size in channel
        :params ull - Use last layer, add to feature out from last layer
        """
        super().__init__()
        self.ull = ull
        self.outputs = []
        self.merges = []
        self.last_out = None
        leaky = 0.1 if out_channels <= 64 else 0
        
        for in_channels in in_channels_list:
            self.outputs.append(
                self.create_conv_bn1x1(
                    in_channels, out_channels, stride = 1, leaky = leaky)
            )
        for _ in range(len(self.outputs) - 1):
            self.merges.append(
                self.create_conv_bn3x3(
                    out_channels, out_channels, leaky = leaky)
            )
        
        if self.ull:
            self.last_out = self.create_conv_bn3x3(
                in_channels_list[-1], out_channels, stride=2, leaky = leaky
            )
    
    def forward(self, input):
        results_c = []
        result_uul = []

        last_key = ""
        for idx, key in enumerate(input):
            last_key = key
            results_c.append(
                self.outputs[idx](input[key])
            )
        
        if self.ull:
            result_uul.append(
                self.last_out(input[last_key])
            )
        
        for idx in range(len(results_c - 1, 0, -1)):
            # idx - номер слоя для объединения
            up = F.interpolate(
                results_c[idx+1],
                size=[results_c[idx].size(2),
                results_c[idx].size(3)],
                mode="nearest"
            )
            results_c[idx] = self.merges[idx](results_c[idx] + up)

        return results_c + result_uul
    
    def get_cnt_feature(self) -> int:
        cnt = len(self.outputs)
        if self.ull:
            cnt += 1
        return cnt
    
    def create_conv_bn1x1(self, in_channels, out_channels, stride, leaky=0):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 1, stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )
    
    def create_conv_bn3x3(self, in_channels, out_channels, stride=1, leaky=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channels):
        super().__init__()

        assert out_channels % 4 == 0
        leaky = 0.1 if out_channels <= 64 else 0

        self.conv3X3 = self.conv_bn_no_relu(
            in_channel, out_channels//2, kernel=3, stride=1)

        self.conv5X5_1 = self.conv_bn(
            in_channel, out_channels//4, kernel=3, stride=1, leaky = leaky)
        self.conv5X5_2 = self.conv_bn_no_relu(
            out_channels//4, out_channels//4, kernel=3, stride=1)

        self.conv7X7_2 = self.conv_bn(
            out_channels//4, out_channels//4, kernel=3, stride=1, leaky = leaky)
        self.conv7x7_3 = self.conv_bn_no_relu(
            out_channels//4, out_channels//4, kernel=3, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out
    
    def conv_bn_no_relu(self, in_channel, out_channel, kernel, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
    
    def conv_bn(self, in_channel, out_channel, kernel, stride = 1, leaky = 0):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors*2, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors*10, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):

    def __init__(self, config: ConfigNet, path: str = None):
        """
        :param cfg:  Network related settings.
        :param path: Path to model, if he is None will load pretrained model
        """
        super().__init__()
        self.config = config
        self.ssh_list = nn.ModuleList()
        backbone = None

        self._create_backbone(path)
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            self.config.in_channels_list,
            self.config.out_channels,
            ull=self.config.use_last_layer
        )
        
        # Context Module
        for _ in range(self.fpn.get_cnt_feature()):
            self.ssh_list.append(
                SSH(self.config.out_channels, self.config.out_channels))
        
        # Out
        self.BboxHead = self._create_bbox_head(
            fpn_cnt=self.fpn.get_cnt_feature(),
            inchannels=self.config.out_channels
        )
        self.ClassHead = self._create_class_head(
            fpn_cnt=self.fpn.get_cnt_feature(),
            inchannels=self.config.out_channels
        )
        self.LandmarkHead = self._create_landmark_head(
            fpn_cnt=self.fpn.get_cnt_feature(),
            inchannels=self.config.out_channels
        )
    
    def _create_backbone(self, path):
        # Config backbone
        if self.config.backbone == "resnet50":
            backbone = models.resnet50(pretrained=path is None)
        
        if backbone is None:
            raise ValueError("Backbone don't found")
        
        self.backbone = models._utils.IntermediateLayerGetter(
            backbone, self.config.return_layers)
    
    def _create_bbox_head(self, fpn_cnt, inchannels, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_cnt):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead
    
    def _create_class_head(self, fpn_cnt, inchannels, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_cnt):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead
    
    def _create_landmark_head(self, fpn_cnt, inchannels, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_cnt):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead
    
    def forward(self, inputs):
        out = self.backbone(inputs)
        features = self.fpn(out)  # C {2,..,6}

        for idx in range(self.fpn.get_cnt_feature()):
            features[idx] = self.ssh_list[idx](features[idx])
        
        bbox_regressions = torch.cat(
            [self.BboxHead[i](f) for i, f in enumerate(features)], dim=1)
        classifications = torch.cat(
            [self.ClassHead[i](f) for i, f in enumerate(features)], dim=1)
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](f) for i, f in enumerate(features)], dim=1)
        
        output = (
            bbox_regressions,
            F.softmax(classifications, dim=-1),
            ldm_regressions
        )
        return output
