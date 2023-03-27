import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict

class ResnetCustomModel(torch.nn.Module):

    def __make_stage(self, _in, _out, out_block, start_stride, index, n_block):
        stage = OrderedDict([])
        stage[f'stage-{index}-Block(1/{n_block})'] = self.ResidualBlock(_in, _out, is_plane=(start_stride==1), stride=start_stride)
        for i in range(1, n_block-1):
            stage[f'stage-{index}-Block({i+1}/{n_block})'] = self.ResidualBlock(_in, _out)
        stage[f'stage-{index}-Block({n_block}/{n_block})'] = self.ResidualBlock(_in, _out, out_block=out_block)

        return nn.Sequential(stage)

    def __init__(self, n_classes=None, is_classification=False, pretrain:nn.Sequential=None):
        super(ResnetCustomModel, self).__init__()

        self.pretrain_part = nn.Sequential(OrderedDict(
                [
                    ('conv-00', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
                    ('relu-00', nn.ReLU()),
                    ('conv-01', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
                    ('relu-01', nn.ReLU()),
                    ('dropout', nn.Dropout2d(p=0.2))
                ]
            )
        )
        # init pretrain_part 
        if pretrain is not None:
            with torch.no_grad():
                self.pretrain_part[0].weight.data = pretrain[0].weight.data
                self.pretrain_part[2].weight.data = pretrain[2].weight.data
        
        self.stage1 = nn.Sequential(OrderedDict(
                [
                    ('conv-1', nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3)),
                    ('batch_norm-1', nn.BatchNorm2d(64)),
                    ('relu-1', nn.ReLU()),
                    ('maxpooling', nn.MaxPool2d(kernel_size=2, stride=2))
                ]
            )
        )

        self.stage2 = self.__make_stage(64, 256, 128, start_stride=1, index=2, n_block=3)
        self.stage3 = self.__make_stage(128, 512, 256, start_stride=2, index=3, n_block=8)
        self.stage4 = self.__make_stage(256, 1024, 512, start_stride=2, index=4, n_block=36)
        self.stage5 = self.__make_stage(512, 2048, 512, start_stride=2, index=5, n_block=3)

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
        
        return self.fc(x)

    
    class ResidualBlock(torch.nn.Module):
        def __init__(self, _in, _out, stride=1, is_plane=False, out_block=None):
            super(ResnetCustomModel.ResidualBlock, self).__init__()

            if stride == 2 or is_plane:
                self.conv1 = nn.Conv2d(_in, _in, kernel_size=1, stride=stride)
            else: 
                self.conv1 = nn.Conv2d(_out, _in, kernel_size=1)

            self.conv2 = nn.Conv2d(_in, _in, kernel_size=3, padding=1)

            if out_block is not None:
                self.conv3 = nn.Conv2d(_in, out_block, kernel_size=1)
            else:
                self.conv3 = nn.Conv2d(_in, _out, kernel_size=1)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return F.relu(x)

class ResnetPytorchModel:
    def __init__(self, version='resnet18', pretrained=True, classification=False):
        pass
