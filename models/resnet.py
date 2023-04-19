import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict

class ResnetCustomModel(torch.nn.Module):

    def __make_stage(self, _in, out_block, start_stride, index, n_block):
        stage = OrderedDict([])

        downsample = False
        if start_stride != 1:
            downsample = True

        stage[f'stage-{index}-Block(1/{n_block})'] = self.ResidualBlock(_in, is_plane=(start_stride==1), downsample=downsample, stride=start_stride)
   
        for i in range(1, n_block-1):
            stage[f'stage-{index}-Block({i+1}/{n_block})'] = self.ResidualBlock(_in)
        
        stage[f'stage-{index}-Block({n_block}/{n_block})'] = self.ResidualBlock(_in, out_block=out_block)

        return nn.Sequential(stage)

    def __init__(self, n_classes=None, is_classification=False, pretrain:nn.Sequential=None,  is_freeze_pretrain=False):
        super(ResnetCustomModel, self).__init__()

        self.is_classification = is_classification

        self.pretrain_part = nn.Sequential(OrderedDict(
                [
                    ('conv-00', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
                    ('bash-00', nn.BatchNorm2d(64)),
                    ('conv-01', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
                    ('bash-01', nn.BatchNorm2d(64)),
                    ('relu-00', nn.ReLU(inplace=True))
                ]
            )
        )
        # init pretrain_part 
        if pretrain is not None:
            self.pretrain_part[0].weight.data = pretrain[0].weight.data
            self.pretrain_part[1].weight.data = pretrain[1].weight.data
            self.pretrain_part[2].weight.data = pretrain[2].weight.data
            self.pretrain_part[3].weight.data = pretrain[3].weight.data
            if is_freeze_pretrain:
                self.pretrain_part[0].weight.requires_grad_ = False
                self.pretrain_part[1].weight.requires_grad_ = False
                self.pretrain_part[2].weight.requires_grad_ = False
                self.pretrain_part[3].weight.requires_grad_ = False
        
        self.stage1 = nn.Sequential(OrderedDict(
                [
                    ('conv-1', nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=4)),
                    ('batch_norm-1', nn.BatchNorm2d(64)),
                    ('relu-1', nn.ReLU(inplace=True)),
                    ('maxpooling', nn.MaxPool2d(kernel_size=3, stride=2))
                ]
            )
        )

        self.stage2 = self.__make_stage(64, 128, start_stride=1, index=2, n_block=3)
        self.stage3 = self.__make_stage(128, 256, start_stride=2, index=3, n_block=4)
        self.stage4 = self.__make_stage(256, 512, start_stride=2, index=4, n_block=6)
        self.stage5 = self.__make_stage(512, 512, start_stride=2, index=5, n_block=3)

        self.avgpooling = nn.AvgPool2d(kernel_size=7)
        out_network = 1
        if is_classification and n_classes is not None:
            out_network = n_classes
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, out_network)

    def forward(self, x):
        x = self.pretrain_part(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpooling(x)

        x = self.flatten(x)
        x = self.fc(x)

        if not self.is_classification:
            x_after_dot = x.detach().clone()
            x_after_dot = x_after_dot % 1

            x = x - x_after_dot

        return x

    
    class ResidualBlock(torch.nn.Module):
        def __init__(self, _in, stride=1, downsample=False, is_plane=False, out_block=None):
            super(ResnetCustomModel.ResidualBlock, self).__init__()
            self.is_downsample = False

            if stride == 2 or is_plane:
                self.is_downsample = True
                self.shortcut = nn.Conv2d(_in, _in , kernel_size=1,  stride=stride)
                self.conv1 = nn.Conv2d(_in, _in, kernel_size=3, padding=1, stride=stride)
            else: 
                self.conv1 = nn.Conv2d(_in, _in, kernel_size=3, padding=1)

            self.batchnorm1 = nn.BatchNorm2d(_in) 

            if out_block is not None:
                self.is_downsample = True
                self.shortcut = nn.Conv2d(_in, out_block, kernel_size=1)
                self.conv2 = nn.Conv2d(_in, out_block, kernel_size=3, padding=1)
                self.batchnorm2 = nn.BatchNorm2d(out_block) 
            else:
                self.conv2 = nn.Conv2d(_in, _in, kernel_size=3, padding=1)
                self.batchnorm2 = nn.BatchNorm2d(_in) 
        
        def forward(self, x):
            residual = x

            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = self.batchnorm2(x)
            if self.is_downsample:
                # y = F(x, {Wi}) + Wsx
                residual = self.shortcut(residual)
            x += residual
            return F.relu(x)

class ResnetPytorchModel:
    def __init__(self, version='resnet18', pretrained=True, classification=False):
        pass
