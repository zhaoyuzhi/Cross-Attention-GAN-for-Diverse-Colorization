import torch
import torch.nn as nn
import torch.nn.functional as F
from spectralnorm import SpectralNorm
from Self_Attn import Self_Attn_FM, Self_Attn_C

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                  Blocks
# ----------------------------------------
# Parameters:
# normalize: True for there is BN layer in this block
# dropout: Non-zero dropout value means the dropout probability
class ResAttnBlock(nn.Module):
    def __init__(self, in_dim, latent_dim = 8):
        super(ResAttnBlock, self).__init__()
        # Attention blocks
        self.attn_fm = Self_Attn_FM(in_dim, latent_dim)
        self.attn_c = Self_Attn_C(in_dim, latent_dim)
        # ResBlock
        self.res_gamma = nn.Parameter(torch.zeros(1))
        self.res_conv = SpectralNorm(nn.Conv2d(in_dim, in_dim, 3, 1, 2, 2, bias = False))

    def forward(self, x):
        attn_fm, attn_fm_map = self.attn_fm(x)
        attn_c, attn_c_map = self.attn_c(x)
        res_conv = self.res_gamma * self.res_conv(x)
        out = x + attn_fm + attn_c + res_conv
        return out

class GateAttnBlock(nn.Module):
    def __init__(self, in_dim):
        super(GateAttnBlock, self).__init__()
        # 2 ConvBlocks
        self.gate_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias = False)),
            nn.Sigmoid()
        )
        self.res_conv = SpectralNorm(nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias = False))

    def forward(self, x):
        gate_conv = self.gate_conv(x)
        res_conv = self.res_conv(x)
        res_conv = gate_conv * res_conv
        out = x + res_conv
        return out

class DownBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(DownBlock, self).__init__()
        self.down_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_size, out_size, 4, 2, 1, bias = False)),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(out_size, out_size, 3, 1, 1, bias = False)),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2, inplace = True)
        )
        
    def forward(self, x):
        x = self.down_conv(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpBlock, self).__init__()
        self.up_conv = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias = False)),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(out_size, out_size, 3, 1, 1, bias = False)),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2, inplace = True)
        )
        
    def forward(self, x):
        x = self.up_conv(x)
        return x

class FinalBlock(nn.Module):
    def __init__(self, in_size, out_size = 2):
        super(FinalBlock, self).__init__()
        self.final_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_size, in_size // 2, 3, 1, 1, bias = False)),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_size // 2, out_size, 3, 1, 1, bias = False)),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.final_conv(x)
        return x

# ----------------------------------------
#               Sub network
# ----------------------------------------
# Fully convolutional layers in feature extraction part (Encoder)
# This is for generator only, so BN cannot be attached to the input and output layers of feature extraction part
# Each output of block (conv*) is "convolutional layer + LeakyReLU" that avoids feature sparse
# We replace the adaptive average pooling layer with a convolutional layer with stride = 2, to ensure the size of feature maps fit classifier
class Encoder(nn.Module):
    def __init__(self, in_dim = 1, noise_dim = 1, out_dim = 512):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = in_dim + noise_dim, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.conv2 = DownBlock(64 + noise_dim, 128)
        self.conv3 = DownBlock(128 + noise_dim, 256)
        self.conv4 = DownBlock(256 + noise_dim, 512)
        self.conv5 = DownBlock(512 + noise_dim, 512)
    
    def noise_upsampler(self, noise):
        noise64 = F.interpolate(input = noise, scale_factor = 2, mode = 'nearest')
        noise128 = F.interpolate(input = noise, scale_factor = 4, mode = 'nearest')
        noise256 = F.interpolate(input = noise, scale_factor = 8, mode = 'nearest')
        return noise64, noise128, noise256
    
    def forward(self, x, noise):
        noise64, noise128, noise256 = self.noise_upsampler(noise)
        conv1 = self.conv1(torch.cat((x, noise256), 1))         # batch * 64 * 256 * 256
        conv2 = self.conv2(torch.cat((conv1, noise256), 1))     # batch * 128 * 128 * 128
        conv3 = self.conv3(torch.cat((conv2, noise128), 1))     # batch * 256 * 64 * 64
        conv4 = self.conv4(torch.cat((conv3, noise64), 1))      # batch * 512 * 32 * 32
        conv5 = self.conv5(torch.cat((conv4, noise), 1))        # batch * 512 * 16 * 16
        return conv1, conv2, conv3, conv4, conv5

class MidAttnEncoder(nn.Module):
    def __init__(self):
        super(MidAttnEncoder, self).__init__()
        self.res1 = ResAttnBlock(512)
        self.res2 = ResAttnBlock(512)
        self.res3 = ResAttnBlock(512)
        self.res4 = ResAttnBlock(512)
        self.res5 = ResAttnBlock(512)
        self.res6 = ResAttnBlock(512)
        
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        return x

class ColDecoder(nn.Module):
    def __init__(self, in_size = 512, out_size = 2):
        super(ColDecoder, self).__init__()
        # colup1 is for the input of conv4
        self.colup1 = UpBlock(in_size, 512)
        self.cross1_1 = ResAttnBlock(512)
        self.cross1_2 = ResAttnBlock(512)
        self.cross1_3 = ResAttnBlock(512)
        # colup2 is for the input of conv3
        self.colup2 = UpBlock(1024, 256)
        self.cross2 = ResAttnBlock(256)
        # colup3 is for the input of conv2
        self.colup3 = UpBlock(512, 128)
        # colup4 is for the input of conv1
        self.colup4 = UpBlock(256, 64)
        # salup2 is for the input of up5: batch * (512 = 256 + 256) * 64 * 64
        self.finalconv = FinalBlock(128, out_size)

    def forward(self, x, conv4, conv3, conv2, conv1):
        # 1st attention module
        x = self.colup1(x)                                      # batch * 512 * 32 * 32
        conv4 = self.cross1_1(conv4)
        conv4 = self.cross1_2(conv4)
        conv4 = self.cross1_3(conv4)
        y1 = torch.cat((x, conv4), 1)                           # batch * (1024 = 512 + 512) * 32 * 32
        # 2nd attention module
        y1_ = self.colup2(y1)                                   # batch * 256 * 64 * 64
        conv3 = self.cross2(conv3)
        y2 = torch.cat((y1_, conv3), 1)                         # batch * (512 = 256 + 256) * 64 * 64
        # 3rd attention module
        y2_ = self.colup3(y2)                                   # batch * 128 * 128 * 128
        y3 = torch.cat((y2_, conv2), 1)                         # batch * (256 = 128 + 128) * 128 * 128
        # 4th attention module
        y3_ = self.colup4(y3)                                   # batch * 64 * 256 * 256
        y4 = torch.cat((y3_, conv1), 1)                         # batch * (128 = 64 + 64) * 256 * 256
        # final module
        y4 = self.finalconv(y4)                                 # batch * 3 * 256 * 256
        return y1, y2, y3, y4

class SalDecoder(nn.Module):
    def __init__(self, mid_size = 128, out_size = 1):
        super(SalDecoder, self).__init__()
        # salup1 is for the input of y2: batch * (512 = 256 + 256) * 64 * 64
        self.salup1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(512, mid_size, 4, 2, 1, bias = False)),
            nn.BatchNorm2d(mid_size),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # salup2 is for the input of y3: batch * (256 = 128 + 128) * 128 * 128
        self.salup2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(256, mid_size, 3, 1, 1, bias = False)),
            nn.BatchNorm2d(mid_size),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # salfinal is for the final prediction
        self.salfinal = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(mid_size * 2, mid_size, 4, 2, 1, bias = False)),
            nn.BatchNorm2d(mid_size),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.ConvTranspose2d(mid_size, out_size, 3, 1, 1, bias = False)),
            nn.Sigmoid()
        )

    def forward(self, y2, y3):
        y2 = self.salup1(y2)                                    # batch * mid_size * 128 * 128
        y3 = self.salup2(y3)                                    # batch * mid_size * 128 * 128
        y = torch.cat((y2, y3), 1)                              # batch * (mid_size * 2) * 128 * 128
        y = self.salfinal(y)                                    # batch * 1 * 256 * 256
        return y

# ----------------------------------------
#                Generator
# ----------------------------------------
# Generator contains 2 Auto-Encoders
class GeneratorAttnUNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 2):
        super(GeneratorAttnUNet, self).__init__()
        # The generator is U shaped
        # It means: input -> downsample -> upsample -> output
        # Encoder
        self.Encoder = Encoder()
        # Encoder (middle)
        self.MidAttnEncoder = MidAttnEncoder()
        # Decoder (colorization)
        self.ColDecoder = ColDecoder()
        # Decoder (saliency map)
        self.SalDecoder = SalDecoder()

    def forward(self, x, noise):
        # U-Net generator with skip connections from encoder to decoder
        # Encoder
        # conv1: batch * 64 * 256 * 256; conv2: batch * 128 * 128 * 128; conv3: batch * 256 * 64 * 64
        # conv4: batch * 512 * 32 * 32; conv5: batch * 512 * 16 * 16
        conv1, conv2, conv3, conv4, conv5 = self.Encoder(x, noise)
        # Encoder (middle)
        conv5 = self.MidAttnEncoder(conv5)
        # Decoder (colorization)
        y1, y2, y3, col = self.ColDecoder(conv5, conv4, conv3, conv2, conv1)
        # Decoder (saliency map)
        sal = self.SalDecoder(y2, y3)

        return col, sal

# ----------------------------------------
#    Perceptual Network / Discriminator
# ----------------------------------------
# Discriminator follows the architecture of LabVGG16
# You may define the hierarchical output layers according to the pre-trained network structure
# In this work, we define conv4_3 layer as the output
class LabDiscriminator(nn.Module):
    def __init__(self, in_dim = 3, num_classes = 1000, z_dim = 8):
        super(LabDiscriminator, self).__init__()
        # feature extraction part
        self.num_classes = num_classes
        self.z_dim = z_dim
        # conv1 output size 224 * 224
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = in_dim, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # conv2 output size 112 * 112
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # conv3 output size 56 * 56
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # conv4 output size 28 * 28
        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # category classifier
        self.conv5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.conv6 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.category_classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # GAN regressor
        self.gan_regressor = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 1, padding = 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 256, out_channels = 1, kernel_size = 4, stride = 1, padding = 1))
        )
        # noise regressor
        self.noise_regressor = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 256, out_channels = 1, kernel_size = 3, stride = 1, padding = 1))
        )

    def forward(self, l, ab):
        # Share-weights part
        lab = torch.cat((l, ab), 1)                             # batch * 3 * 256 * 256
        conv1 = self.conv1(lab)                                 # batch * 64 * 256 * 256
        conv2 = self.conv2(conv1)                               # batch * 128 * 128 * 128
        conv3 = self.conv3(conv2)                               # batch * 256 * 64 * 64
        conv4 = self.conv4(conv3)                               # batch * 512 * 32 * 32
        # Classifier part
        conv5 = self.conv5(conv4)                               # batch * 512 * 16 * 16
        conv6 = self.conv6(conv5)                               # batch * 512 * 8 * 8
        conv6 = conv6.view(l.size(0), -1)                       # batch * (512 * 8 * 8)
        category_feature = self.category_classifier(conv6)      # batch * 1000
        # GAN regressor part
        gan_feature = self.gan_regressor(conv4)                 # batch * 1 * 30 * 30
        # Noise regressor part
        recon_z = self.noise_regressor(conv4)                   # batch * 1 * 32 * 32
        return conv1, conv2, conv3, conv4, gan_feature, category_feature, recon_z

'''
a = torch.randn(4, 1, 256, 256).cuda()
net = GeneratorAttnUNet().cuda()
b, c = net(a)
print(b.shape)
print(c.shape)
'''
'''
a = torch.randn(1, 1, 256, 256).cuda()
b = torch.randn(1, 2, 256, 256).cuda()
net = LabDiscriminator().cuda()
conv1, conv2, conv3, conv4, gan_feature, category_feature = net(a, b)
print(conv1.shape)
print(conv2.shape)
print(conv3.shape)
print(conv4.shape)
print(gan_feature.shape)
print(category_feature.shape)
'''