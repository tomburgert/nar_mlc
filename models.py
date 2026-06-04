import torch.nn as nn
from torchvision import models


def get_network(name, pretrained, num_channels, num_classes):
    if name == 'resnet18':
        resnet = models.resnet18
        weights = models.ResNet18_Weights
        fc_size = 512
        return ResNet(resnet, weights, fc_size, pretrained=pretrained, num_classes=num_classes, num_channels=num_channels)
    elif name == 'resnet50':
        resnet = models.resnet50
        weights = models.ResNet
        return ResNet(resnet, weights, fc_size, pretrained=pretrained, num_classes=num_classes, num_channels=num_channels)
    elif name == 'vit_b':
        vit = vit = models.vit_b_16(
            weights=None,
            image_size=256
        )
        return ViT(vit, num_classes=num_classes, num_channels=num_channels)
    elif name == 'efficientNet':
        net = models.efficientnet_b0
        weights = None
        fc_size = None
        return MyEfficientNet(net, pretrained=pretrained, num_classes=num_classes, num_channels=num_channels)
        
    raise RuntimeError(f'Model {name} not implemented ')


class ResNet(nn.Module):
    def __init__(self, resnet, weights, fc_size, pretrained=False, num_classes=None, num_channels=None):
        super().__init__()
        weights = weights.DEFAULT if pretrained else None
        self.fc_size = fc_size
        self.resnet = resnet(weights=weights)
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        return self.resnet.forward(x)

    def register_hooks(self, get_activation):
        '''Put hook avg_pool layer for feature extraction.'''
        return self.resnet.avgpool.register_forward_hook(get_activation('avg_pool'))


class ViT(nn.Module):
    def __init__(self, vit, num_classes, num_channels=10):
        super().__init__()
        self.vit = vit
        hidden_dim = vit.hidden_dim
        in_features = vit.heads.head.in_features
        
        self.vit.conv_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=(16, 16), stride=(16, 16))
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


class MyEfficientNet(nn.Module):
    def __init__(self, net, pretrained=False, num_classes=None, num_channels=None):
        super().__init__()
        self.efficientnet = net()
        self.fc_size = self.efficientnet.classifier[1].in_features
        self.efficientnet.features[0][0]= nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.efficientnet.classifier[1] = nn.Linear(self.fc_size, num_classes)

    def forward(self, x):
        return self.efficientnet.forward(x)

    def register_hooks(self, get_activation):
        '''Put hook avg_pool layer for feature extraction.'''
        return self.efficientnet.avgpool.register_forward_hook(get_activation('avg_pool'))
