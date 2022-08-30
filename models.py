from torch import nn
from torchvision import models
from archs.vit import ViT
from archs.demo import DemoClassifier


def resnet50(out_features: int, device: str):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=2048, out_features=out_features)

    model.to(device)

    return model


def vit(out_features: int, channels: int, duration: int, device: str):
    model = ViT(input_shape=(channels, 80, duration), device=device,
                out_dim=out_features)

    model.to(device)
    model.cuda()
    return model


def demo(channels: int, device: str):
    model = DemoClassifier(channels)

    model.to(device)
    model.cuda()
    return model
