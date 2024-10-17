import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        """ UNet EncoderBlock"""
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        """ UNet DecoderBlock"""
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False):
        """ UNet"""
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(64, 128, bn=bn)
        self.enc3 = _EncoderBlock(128, 256, bn=bn)
        self.enc4 = _EncoderBlock(256, 512, bn=bn)
        self.polling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlock(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlock(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlock(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], mode='bilinear',
                                                  align_corners=True), enc4], 1))
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        final = self.final(dec1)
        return final


class _EncoderBlockV2(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        """ UNet EncoderBlockV2"""
        super(_EncoderBlockV2, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        """ UNet DecoderBlockV2"""
        super(_DecoderBlockV2, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decode(x)


class UNetV2(nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False):
        """ UNetV2"""
        super(UNetV2, self).__init__()
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

# Fine-tuning Sperate Encoder and Decoder
class UNetV2_Encoder(nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False):
        super(UNetV2_Encoder, self).__init__()
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return enc1, enc2, enc3, enc4, center, final

class UNetV2_Decoder(nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False):
        super(UNetV2_Decoder, self).__init__()
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, enc1, enc2, enc3, enc4, center):
    
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final
#MLP
class MLPModel(nn.Module):
    
  def __init__(self):
    super(MLPModel, self).__init__()
    self.flatten = nn.Flatten()  # 将二维图像展开为一维
    self.linear1 = nn.Linear(512*4*4, 512*4*4)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(512*4*4, 512*4*4)
    self.linear3 = nn.Linear(512*4*4, 512*4*4)
    
  def forward(self, x):
    out = self.flatten(x)
    out = self.linear1(out)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    return out


import torch.optim
from collections import OrderedDict
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, seq_net, name='MLP', activation=torch.tanh):
        super().__init__()
        self.features = OrderedDict()
        for i in range(len(seq_net) - 1):
            self.features['{}_{}'.format(name, i)] = nn.Linear(seq_net[i], seq_net[i + 1], bias=True)
        self.features = nn.ModuleDict(self.features)
        self.active = activation

        # initial_bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = x.view(-1, 2)
        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            x = layer(x)
            if i == length - 1: break
            i += 1
            x = self.active(x)
        return x

class Net_Encoder(nn.Module):
    def __init__(self, seq_net, name='MLP', activation=torch.tanh):
        super().__init__()
        self.features = OrderedDict()
        for i in range(len(seq_net) - 1):
            self.features['{}_{}'.format(name, i)] = nn.Linear(seq_net[i], seq_net[i + 1], bias=True)
        self.features = nn.ModuleDict(self.features)
        self.active = activation

        # initial_bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = x.view(-1, 2)
        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            x = layer(x)
            if i == 2: break
            # if i == length - 1: break
            i += 1
            x = self.active(x)
        return x

class Net_Decoder(nn.Module):
    def __init__(self, seq_net, name='MLP', activation=torch.tanh):
        super().__init__()
        self.features = OrderedDict()
        for i in range(len(seq_net) - 1):
            self.features['{}_{}'.format(name, i)] = nn.Linear(seq_net[i], seq_net[i + 1], bias=True)
        self.features = nn.ModuleDict(self.features)
        self.active = activation

        # initial_bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = x.view(-1, 2)
        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            if i == 0 or i == 1 or i == 2: 
                x = x
            else:
                x = layer(x)
                x = self.active(x)
            if i == length - 1: break
            i += 1
            
        return x
#VAE
class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, keep_prob=0):
        super(Encoder, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 16)

        self._enc_mu = torch.nn.Linear(16, D_out)
        self._enc_log_sigma = torch.nn.Linear(16, D_out)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.linear1(x))
        x = self.linear2(x)
        return self._enc_mu(x), self._enc_log_sigma(x)

class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, keep_prob=0):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.linear1(x))
        return self.linear2(x)
    

class Encoder2d(torch.nn.Module):
    def __init__(self, in_channels,
                 latent_dim,
                 hidden_dims):
        super(Encoder2d, self).__init__()
        
        modules = []
        img_length = 64
        for cur_channels in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            in_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(in_channels * img_length * img_length,
                                     latent_dim)
        self.fc_var = nn.Linear(in_channels * img_length * img_length,
                                    latent_dim)
        self.latent_dim = latent_dim

      

    def forward(self, x):
        # x = self.encoder(x)
        # print(x.shape)
        # x=torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        # print(x.shape)
        x=torch.flatten(x, 1)
        # print(x.shape)
        
        return self.fc_mu(x), self.fc_var(x)

class Decoder2d(torch.nn.Module):
    def __init__(self, in_channels,
                 latent_dim,
                 hidden_dims):
        super(Decoder2d, self).__init__()

        modules = []
        img_length = 1
        in_channels=512
        self.decoder_projection = nn.Linear(
            latent_dim, in_channels * img_length * img_length)
        self.decoder_input_chw = (in_channels, img_length, img_length)
        for i in range(len(hidden_dims) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i - 1]), nn.ReLU()))
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[0],
                                   hidden_dims[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[0]), nn.ReLU(),
                nn.Conv2d(hidden_dims[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU()))
        self.decoder = nn.Sequential(*modules)


    def forward(self, x):
        # x = self.decoder_input(x)
        # # print(x.shape)
        # x = x.view(-1, 512, 2, 2)
        # x = self.decoder(x)
        # x = self.final_layer(x)
        x = self.decoder_projection(x)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        x = self.decoder(x)
        return x

class VanillaVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples,
               current_device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]