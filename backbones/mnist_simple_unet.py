import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mnist_trajectory_loader import MNISTTrajLoader
import sys
#sys.path.append('../pneumothorax-segmentation/unet_pipeline/')

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            # x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1,
                 retain_dim=True, out_sz=(28, 28)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out





class RewardNet(nn.Module):
    def __init__(self, enc_chs=(1, 64, 128, 256)):
        super().__init__()


        self.layers = []
        for i in range(len(enc_chs)-1):
            self.layers.append(nn.Conv2d(enc_chs[i], enc_chs[i+1], 4, 2, 1, bias=True).to('cuda'))
            self.layers.append(nn.BatchNorm2d(enc_chs[i+1]).to('cuda'))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Flatten())
        self.layers.append(nn.LazyLinear(1).to('cuda'))

        # self.conv1 = nn.Conv2d(enc_chs[0], enc_chs[1],
        #     4, 2, 1, bias=False)
        #
        # self.conv2 = nn.Conv2d(enc_chs[1], enc_chs[2],
        #     4, 2, 1, bias=False)
        # self.bn2 = nn.BatchNorm2d(enc_chs[2])
        #
        # self.conv3 = nn.Conv2d(enc_chs[2], enc_chs[3],
        #     4, 2, 1, bias=False)
        # self.bn3 = nn.BatchNorm2d(enc_chs[3])
        #
        # self.conv4 = nn.Conv2d(enc_chs[3], enc_chs[4],
        #     4, 2, 1, bias=False)
        # self.bn4 = nn.BatchNorm2d(enc_chs[4])
        #
        # self.flatten = nn.Flatten()
        # self.readout = nn.Linear(enc_chs[4], 1)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        # x = F.leaky_relu(self.conv1(x), 0.2, True)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        # x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        # x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        # x = self.flatten(x)
        # x = self.readout(x)

        return x


class DebugNet(nn.Module):
    def __init__(self, in_chs=(1, 2, 2), num_class=2, torso=None):
        super().__init__()

        self.n_total = int(torch.prod(torch.tensor(in_chs)))
        self.n_pixel = int(torch.prod(torch.tensor(in_chs[1:])))
        self.num_class = num_class
        self.in_chs = in_chs
        n_hidden = 16

        if torso is None:
            self.torso = torch.nn.Sequential()
            self.torso.append(nn.Flatten())
            self.torso.append(nn.Linear(self.n_total, n_hidden).to('cuda'))
            self.torso.append(nn.ReLU())
            self.torso.append(nn.Linear(n_hidden, n_hidden).to('cuda'))
            self.torso.append(nn.ReLU())
        else:
            self.torso = torso

        self.prereadout = nn.Linear(n_hidden, n_hidden).to('cuda')
        self.activation = nn.ReLU()
        self.readout = nn.Linear(n_hidden, self.num_class*self.n_pixel).to('cuda')
    def forward(self, x):

        for layer in self.torso:
            x = layer(x)

        x = self.prereadout(x)
        x = self.activation(x)
        x = self.readout(x)

        return x.reshape(-1, self.num_class, self.in_chs[1], self.in_chs[2])


if __name__ == '__main__':
    custom_mnist_dataset = MNISTTrajLoader(img_dir='../data/MNIST/0/', eps_noise_background=0.1, noise_strategy='gaussian', beta=0.15)
    train_dataloader = DataLoader(custom_mnist_dataset, batch_size=64, shuffle=True)

    traj, _ = next(iter(train_dataloader))
    x = traj[:, 0:1].to('cuda')

    unet = UNet(enc_chs=(1, 64, 128), dec_chs=(128, 64), num_class=2).to('cuda')
    print(unet(x).shape)

    reward_net = RewardNet().to('cuda')
    print(reward_net(x).shape)

    x = torch.zeros(64,1,3,3).to('cuda')
    debug_net = DebugNet(in_chs=(1,3,3))
    print(debug_net(x).shape)
    # print(debug_net(x))