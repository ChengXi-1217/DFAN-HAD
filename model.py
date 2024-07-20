import torch
from torch import nn
import numpy as np
from torch import nn, optim

import transforms
from transforms import GramSchmidtTransform
from torch import Tensor


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, FWT: GramSchmidtTransform, input: Tensor):
        #happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input.to(self.device))
        b = input.size(0)
        return input.view(b, -1)


def RX(hsi_2D):
    m, n = hsi_2D.shape
    X_in = hsi_2D
    X_in_mean = np.mean(X_in, 0)
    X_in = X_in - np.tile(X_in_mean, (m, 1))
    X_in_T = X_in.T
    D = np.cov(X_in_T)
    invD = np.linalg.inv(D)
    out = np.zeros(len(X_in))
    for i in range(len(X_in)):
        x = X_in[i]
        out[i] = np.sqrt(np.dot(np.dot(x.T, invD), x))
    return out, invD


class OrthoAE_Unet(nn.Module):
    def __init__(self, input_dim, latent_layer_dim, args):
        super(OrthoAE_Unet, self).__init__()
        # Put your encoder network here, remember about the output D-dimension
        self.input_dim = input_dim
        width = args.width
        height = args.length
        self.act_f_1 = nn.ReLU()
        self.act_f_2 = nn.Sigmoid()
        self.encoder_layer1 = torch.nn.Linear(input_dim, input_dim // 2, bias=False)
        self.BatchNorm1d_enc_layer1 = nn.BatchNorm1d(input_dim // 2)

        self.encoder_layer2 = torch.nn.Linear(input_dim // 2, input_dim // 4, bias=False)
        self.BatchNorm1d_enc_layer2 = nn.BatchNorm1d(input_dim // 4)

        self.encoder_layer3 = torch.nn.Linear(input_dim // 4, latent_layer_dim, bias=False)
        self.BatchNorm1d_enc_layer3 = nn.BatchNorm1d(latent_layer_dim)

        self._excitation = nn.Sequential(
            nn.Linear(in_features=latent_layer_dim, out_features=round(latent_layer_dim // 2),  bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(latent_layer_dim // 2), out_features=latent_layer_dim,  bias=False),
            nn.Sigmoid(),
        )

        self.OrthoAttention = Attention()
        self.F_C_A = transforms.GramSchmidtTransform.build(latent_layer_dim, height)
        self.decoder_layer1 = torch.nn.Linear(latent_layer_dim, input_dim // 4, bias=False)
        self.BatchNorm1d_dec_layer1 = nn.BatchNorm1d(input_dim // 4)
        self.decoder_layer2 = torch.nn.Linear(input_dim // 4 * 2, input_dim // 2, bias=False)
        self.BatchNorm1d_dec_layer2 = nn.BatchNorm1d(input_dim // 2)
        self.decoder_layer3 = torch.nn.Linear(input_dim // 2 * 2, input_dim, bias=False)
        self.BatchNorm1d_dec_layer3 = nn.BatchNorm1d(input_dim)

    def forward(self, x, args):
        width = args.width
        height = args.length
        x = x.to(torch.float32)
        e1 = self.act_f_1(self.BatchNorm1d_enc_layer1(self.encoder_layer1(x)))
        e2 = self.act_f_1(self.BatchNorm1d_enc_layer2(self.encoder_layer2(e1)))
        enc = self.act_f_1(self.BatchNorm1d_enc_layer3(self.encoder_layer3(e2)))

        d3_enc_fea = enc.unsqueeze(0)
        d3_enc_fea = d3_enc_fea.reshape((height, width, enc.shape[1]))
        d3_enc_fea = d3_enc_fea.permute(2, 0, 1)
        d4_enc_fea = d3_enc_fea.unsqueeze(0)
        fea_1 = self.OrthoAttention(self.F_C_A, d4_enc_fea)
        b, c = fea_1.size(0), fea_1.size(1)
        attention_4d = self._excitation(fea_1).view(b, c, 1, 1)
        attention_2d = attention_4d.view(-1, c)
        attention_fea = enc * attention_2d

        d1 = self.act_f_1(self.BatchNorm1d_dec_layer1(self.decoder_layer1(attention_fea)))
        d11 = torch.cat([d1, e2], axis=1)
        d2 = self.act_f_1(self.BatchNorm1d_dec_layer2(self.decoder_layer2(d11)))
        d22 = torch.cat([d2, e1], axis=1)
        dec = self.act_f_2(self.BatchNorm1d_dec_layer3(self.decoder_layer3(d22)))

        return attention_fea, dec