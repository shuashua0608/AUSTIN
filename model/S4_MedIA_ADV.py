import torch
import torch.nn as nn
from .s4 import S4
import torchvision.models as models
import math 


def LoadParameter(_structure, _parameterDir):
    checkpoint = torch.load(_parameterDir)
    model_state_dict = _structure.state_dict()
    for key in checkpoint:
        if ((key == 'fc.weight') | (key == 'fc.bias')):
            pass
        else:
            model_state_dict[key.replace('module.', '')] = checkpoint[key]
    _structure.load_state_dict(model_state_dict)
    model = _structure.cuda()
    return model


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Builder(nn.Module):
    def __init__(self, block, layers, num_classes=2, freeze_bn=True):
        self.inplanes = 64
        self.freeze_bn = freeze_bn
        super(ResNet_Builder, self).__init__()
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 1))
        self.softmax = nn.Softmax(dim=1)
        self.pred_fc2 = nn.Linear(512 * 6, num_classes)
        self.layer_name = ["layer1",
                           "layer2",
                           "layer3",
                           "layer4",
                           "fc"]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x='', phrase='train', return_fea="layer2"):
        f1 = self.conv0(x)
        f1 = self.conv1(f1)
        f1 = self.bn1(f1)
        f1 = self.relu(f1)
        f1 = self.maxpool(f1)  # [-1, 64, 64, 64]
        f1 = self.layer1(f1)   # [-1, 64, 64, 64]
        f1 = self.layer2(f1)   # [-1, 128, 32, 32]
        f1 = self.layer3(f1)   # [-1, 256, 16, 16]
        f1 = self.layer4(f1)   # [-1, 512, 8, 8]
        f1 = self.avgpool(f1)  # [-1, 512, 6, 1]
        f1 = f1.squeeze(3)
        f1 = f1.view(-1, 512 * 6)
        
        return f1


class DummyLayer(nn.Module):
    def forward(self, x):
        return x


def ResNet34(**kwargs):
    model = ResNet_Builder(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


class Discriminator(nn.Module):
    def __init__(self, ndf=256, n_downsampling=2):
        super(Discriminator, self).__init__()
        model = []
        self.n_layers = 2 + n_downsampling
        model += [[nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                   nn.LeakyReLU(0.2, inplace=True)]]
        for i in range(n_downsampling):
            m = 2**(i+1)
            model += [[nn.Conv2d(ndf * m, ndf * (m*2), 4, 2, 1, bias=False),
                       nn.InstanceNorm2d(ndf * (m*2)),
                       nn.LeakyReLU(0.2, inplace=True)]]
        model += [[nn.Conv2d(ndf * (2**(n_downsampling+1)), 1, 2, 1, 0, bias=False)]]

        for n in range(len(model)):
            setattr(self, 'model' + str(n), nn.Sequential(*model[n]))

    def forward(self, input):
        res = input
        for n in range(self.n_layers):
            model = getattr(self, 'model' + str(n))
            res = model(res)
        return res.view(-1)


class LateralFusion(nn.Module):
    def __init__(self, num_classes, video_model_pth):
        super(LateralFusion, self).__init__()
        self.video_model = ResNet34(num_classes=num_classes)
        self.video_model = LoadParameter(self.video_model, video_model_pth)
        self.audio_model = models.resnet18(pretrained=True)
        # self.audio_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, input_video, input_spec, phrase='train'):
        # video conv1
        f1_v = self.video_model.conv1(input_video)  # 2, 64, 128, 128
        f1_v = self.video_model.bn1(f1_v)
        f1_v = self.video_model.relu(f1_v)
        f1_v = self.video_model.maxpool(f1_v)  # 2, 64, 64, 64
        # audio conv1
        f1_a = self.audio_model.conv1(input_spec)
        f1_a = self.audio_model.bn1(f1_a)
        f1_a = self.audio_model.relu(f1_a)
        f1_a = self.audio_model.maxpool(f1_a)
        # video layer1
        f1_v = self.video_model.layer1(f1_v+f1_a*0.5)  # 2, 64, 64, 64
        # audio layer1
        f1_a = self.audio_model.layer1(f1_a)
        # video layer2
        f1_v = self.video_model.layer2(f1_v+f1_a*0.5)  # 2, 128, 32, 32
        # audio layer2
        f1_a = self.audio_model.layer2(f1_a)
        # video layer3
        f1_v = self.video_model.layer3(f1_v+f1_a*0.5)  # 2, 256, 16, 16
        # audio layer3
        f1_a = self.audio_model.layer3(f1_a)
        # video layer4
        f1_v = self.video_model.layer4(f1_v+f1_a*0.5)  # 2, 512, 8, 8
        # audio layer4
        f1_a = self.audio_model.layer4(f1_a)
        # final
        f1 = self.video_model.avgpool(f1_v+f1_a*0.5)  # 2, 512, 6, 1
        
        f1 = f1.squeeze(3)
        f1 = f1.view(-1, 512 * 6)  # 2, 3072
        
        return f1


class S4_MedIA(nn.Module):

    def __init__(
            self,
            d_input,
            l_max,
            d_output,
            d_model,
            n_layers,
            video_model_pth,
            dropout=0.2,
            prenorm=True,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.d_model = d_model
        self.d_input = d_input

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = LateralFusion(num_classes=2,video_model_pth=video_model_pth)
        self.encoder.eval()
        # self.ast = ASTModel(label_dim=2, input_tdim=600, imagenet_pretrain=True, audioset_pretrain=True)
        # self.ast.eval()
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.gelus = nn.ModuleList()
        for idx in range(n_layers):
            self.s4_layers.append(
                S4(H=d_model, l_max=l_max, dropout=dropout, transposed=True)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))
            self.pools.append(nn.AvgPool1d(2))
            self.linears.append(nn.Linear(d_model, int(d_model/2)))
            self.gelus.append(nn.GELU())
            d_model = int(d_model/2)
            l_max = int(l_max/2)

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_s, phrase='train'):
        """
        Input x is shape (B, L, d_input)
        """
        x = x.to(torch.float32)
        B, L = x.shape[0], x.shape[1]
        x = torch.reshape(x, (B * L, *x.shape[2:]))
        x_s = x_s.unsqueeze(1).repeat(1, L, 1, 1, 1)
        x_s = torch.reshape(x_s, (B * L, *x_s.shape[2:]))
        
        x = self.encoder(x, x_s)  # (B, L, d_input) -> (B, L, d_model)
        x = torch.reshape(x, (B, L, -1))
        
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for idx, (layer, norm, dropout, pool,linear, gelu) in \
                enumerate(zip(self.s4_layers, self.norms, self.dropouts, self.pools, self.linears, self.gelus)):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

            #pooling layer
            x = pool(x)

            # MLP
            x = x.transpose(-1, -2)
            x = linear(x)
            x = gelu(x)
            x = x.transpose(-1, -2)

        x = x.transpose(-1, -2)
        res = x.view(-1, 12, 16, 16)
        
        if phrase=='adv':
            return res
        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        #x = x.max(dim=1)
        # x_a = self.ast.forward_features(x_a)
        # x = torch.cat([x,x_a],axis=-1)
        # Decode the outputs
        pred_logit = self.decoder(x)  # (B, d_model) -> (B, d_output)
        pred_score = self.softmax(pred_logit)
        if phrase=='eval':
            return pred_score
        return pred_score, pred_logit, res
    
    
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    x1 = torch.rand((2, 64, 3, 224, 224)).cuda()
    xs = torch.rand((2, 3, 224, 224)).cuda()
    x2 = torch.rand([2, 600, 128]).cuda()
    ResNet_parameterDir = '../../Stroke_Net/model/fair_7.pt'
    model = S4_MedIA(n_layers=3, d_model=3072,d_input=1024,d_output=2,dropout=0.2,l_max=64, video_model_pth=ResNet_parameterDir).cuda()
    model.train()
    _,_,res = model(x1,xs, phrase='train')
    print(res.shape)
    dis = Discriminator(n_downsampling=2, ndf=24).cuda()
    dis_input = torch.cat([res, res], dim=1)
    dis_pred = dis(dis_input.detach())
    print(dis_pred.shape)