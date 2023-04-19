import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from collections import OrderedDict


class CustomVGGPytorch(nn.Module):
    def __init__(self):
        super(CustomVGGPytorch, self).__init__()
        self.conv1_a = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1_a = nn.BatchNorm2d(16)
        self.conv1_b = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1_b = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)



        self.conv2_a = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2_a = nn.BatchNorm2d(32)
        self.conv2_b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2_b = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv3_a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_a = nn.BatchNorm2d(64)
        self.conv3_b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_b = nn.BatchNorm2d(64)
        self.conv3_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_c = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)


        self.conv4_a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4_a = nn.BatchNorm2d(128)
        self.conv4_b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4_b = nn.BatchNorm2d(128)
        self.conv4_c = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4_c = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)



        self.conv5_a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5_a = nn.BatchNorm2d(256)
        self.conv5_b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5_b = nn.BatchNorm2d(256)
        self.conv5_c = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5_c = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(196, 196), mode='bilinear', align_corners=True)
        x = self.conv1_a(x)
        x = self.bn1_a(x)
        x = self.conv1_b(x)
        x = self.bn1_b(x)
        x = self.relu1(x)
        x = self.maxpool1(x)


        x = self.conv2_a(x)
        x = self.bn2_a(x)
        x = self.conv2_b(x)
        x = self.bn2_b(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3_a(x)
        x = self.bn3_a(x)
        x = self.conv3_b(x)
        x = self.bn3_b(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4_a(x)
        x = self.bn4_a(x)
        x = self.conv4_b(x)
        x = self.bn4_b(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5_a(x)
        x = self.bn5_a(x)
        x = self.conv5_b(x)
        x = self.bn5_b(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return torch.sigmoid(x)
class VGGPytorch:
    def __init__(self, version=''):
        pass
