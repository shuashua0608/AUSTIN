import torch
import torch.nn as nn


def freeze_bn(m):
    # freezing gamma and beta, running mean and running var
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False



if __name__ == "__main__":
    from model import slowfastnet
    model = slowfastnet.resnet50(class_num=2)
    model.train()
    model.apply(freeze_bn)
    for m in model.modules():
        if not m.training:
            print(m)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name)
