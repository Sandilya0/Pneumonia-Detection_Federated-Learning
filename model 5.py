import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def replace_bn_with_gn(module):
    """Recursively replace all BatchNorm2d with GroupNorm(1 group)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(1, child.num_features))
        else:
            replace_bn_with_gn(child)

def get_resnet():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Replace BN with GN
    replace_bn_with_gn(model)

    # Modify for grayscale input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last two blocks + fc
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # Binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model
