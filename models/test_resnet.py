import unittest
import torch
import torch.nn as nn
from resnet import ResnetCustomModel
from collections import OrderedDict

class TestClassFaceDataset(unittest.TestCase):
    def test_creation_of_model(self):
        model = ResnetCustomModel()
        x = torch.rand(1, 3, 196, 196)
        out = model.forward(x)
        self.assertTrue(out.size()[0] == 1), f'expect a tensor with size 1; we {out.size()}'
        print(f'The false prediction {out}')
    
    def test_initialisation_of_model(self):
        pretrain = nn.Sequential(OrderedDict(
                [
                    ('conv-00', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
                    ('relu-00', nn.Relu()),
                    ('conv-01', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
                    ('relu-01', nn.Relu())
                ]
            )
        )
        with torch.no_grad():
            pretrain[0].weight.data = torch.rand(64, 3, 3, 3)
            pretrain[2].weight.data = torch.rand(64, 64, 3, 3)

        model = ResnetCustomModel(pretrain=pretrain)
        x = torch.rand(1, 3, 196, 196)
        out = model.forward(x, )
        self.assertTrue(out.size()[0] == 1), f'expect a tensor with size 1; we {out.size()}'
        print(f'The false prediction {out}')
    
    def test_classification_model(self):
        model = ResnetCustomModel(is_classification=True, n_classes=30)
        x = torch.rand(1, 3, 196, 196)
        out = model.forward(x)
        self.assertTrue(out.size()[1] == 30), f'expect a tensor with size 30; we {out}'
        print(f'The false prediction {out}')

if __name__=='__main__':
    unittest.main()