import torch.nn as nn
import torch
from collections import OrderedDict as od
# from torchvision.models.utils import load_state_dict_from_url

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward_to_layer(self, x, layer):
        for subnet in self.features:
            x = subnet(x)
        return x

    def forward_conv5(self, x):
        for subnet in self.features:
            x = subnet(x)
        return x
    
    def forward_convs(self, x):
        feats = []
        feat_idxs = [1,4,7,9,11]
        feat_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        feat_dic = od()
        for idx, subnet in enumerate(self.features):
            x = subnet(x)
            if idx in feat_idxs:
                feats.append(x)
        for k, v in zip(feat_names, feats):
            feat_dic.update({k:v})
        return feat_dic

    def forward(self, x):
        for subnet in self.features:
            x = subnet(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def alexnet(pretrained, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained != '' and pretrained != 'none':
        model.load_state_dict(torch.load(pretrained))
    return model
