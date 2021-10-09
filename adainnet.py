import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import function as function
from function import adaptive_instance_normalization as adain

vggpath = r"./pretrainmodels\vgg19-dcbb9e9d.pth"
vgg19 = models.vgg19()
pre = torch.load(vggpath)
vgg19.load_state_dict(pre)


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg = vgg19.features
        self.v1 = vgg[:2]   # input -> relu_1
        self.v2 = vgg[2:7]  # relu_1 -> relu_2
        self.v3 = vgg[7:14]  # relu_2 -> relu_3
        self.v4 = vgg[14:21]  # relu_3 -> relu_4
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, last_features=True):
        h1 = self.v1(x)
        h2 = self.v2(h1)
        h3 = self.v3(h2)
        h4 = self.v4(h3)
        if last_features:
            return h4
        else:
            return h1, h2, h3, h4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(inplace=True),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1)),
        )

    def forward(self, x):
        output = self.decode(x)
        return output


class AdainNet(nn.Module):
    def __init__(self):
        super(AdainNet, self).__init__()
        self.vggencoder = VGGEncoder()
        self.decoder = Decoder()

    def calc_content_loss(self, de_feature, srt_feature):
        return F.mse_loss(de_feature, srt_feature)

    def calc_style_loss(self, _input, target):
        loss = 0
        for i, t in zip(_input, target):
            input_mean, input_std = function.calc_mean_std(i)
            target_mean, target_std = function.calc_mean_std(t)
            loss += F.mse_loss(input_mean, target_mean) + F.mse_loss(input_std, target_std)
        return loss

    def generator(self, content_feature, style_feature, alpha=1.0):
        g_t = self.vggencoder(content_feature)
        s = self.vggencoder(style_feature)
        adain_out = adain(g_t, s)
        t = alpha * adain_out + (1-alpha)+g_t
        return self.decoder(t)

    def forward(self, content_feature, style_feature, alpha=1.0, lam=5.0):
        assert 0 <= alpha <= 1.0
        en_c = self.vggencoder(content_feature)
        en_s = self.vggencoder(style_feature)
        adain_out = adain(en_c, en_s)
        t = alpha * adain_out + (1 - alpha) + en_c
        out = self.decoder(t)

        # vgg get content feature
        g_t = self.vggencoder(out, last_features=True)
        # vgg get style features (h1, h2, h3, h4)
        g_t_s = self.vggencoder(out, last_features=False)
        s = self.vggencoder(style_feature, last_features=False)

        loss_c = self.calc_content_loss(g_t, t)
        loss_s = self.calc_style_loss(g_t_s, s)
        loss = loss_c + lam * loss_s
        return loss

