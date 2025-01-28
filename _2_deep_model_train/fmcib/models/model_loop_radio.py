import torch
import torch.nn.functional as F
from torch import nn
from reproduction._3_feature_fusion.model.mcVAE import Normal, compute_kl, compute_ll, compute_logvar
import math

class deep_Radio(nn.Module):
    def __init__(self, radio_f_dim=1580, output_dim=128, head_hidden_1=64, head_hidden_2=32, head_hidden_3=32,dropout=0.4, head_id=0, actf='ReLU'):
        super(deep_Radio, self).__init__()

        # self.head_mut = nn.Sequential(nn.Linear(output_dim*2, head_hidden_1), nn.Tanh(), nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_1),
        #                               nn.Linear(head_hidden_1, head_hidden_2), nn.Tanh(), nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_2),
        #                               nn.Linear(head_hidden_2, head_hidden_3), nn.Tanh(), nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_3),
        #                               nn.Linear(head_hidden_3, 1))
        self.act = None
        if actf == 'ReLU':
            self.act = nn.ReLU()
        elif actf == 'Tanh':
            self.act = nn.Tanh()
        elif actf == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        elif actf == 'GELU':
            self.act = nn.GELU()
        elif actf == 'SiLU':
            self.act = nn.SiLU()

        self.head_mut = None
        if head_id == 0:
            self.head_mut = nn.Sequential(nn.Linear(output_dim, head_hidden_1), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_1, 1))
        elif head_id == 1:
            self.head_mut = nn.Sequential(nn.Linear(output_dim, head_hidden_1), self.act, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_1),
                                          nn.Linear(head_hidden_1, 1))
        elif head_id == 2:
            self.head_mut = nn.Sequential(nn.Linear(output_dim, head_hidden_1), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_1, head_hidden_2), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_2, 1))
        elif head_id == 3:
            self.head_mut = nn.Sequential(nn.Linear(output_dim, head_hidden_1), self.act, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_1),
                                          nn.Linear(head_hidden_1, head_hidden_2), self.act, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_2),
                                          nn.Linear(head_hidden_2, 1))
        elif head_id == 4:
            self.head_mut = nn.Sequential(nn.Linear(output_dim, head_hidden_1), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_1, head_hidden_2), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_2, head_hidden_3), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_3, 1))
        elif head_id == 5:
            self.head_mut = nn.Sequential(nn.Linear(output_dim, head_hidden_1), self.act, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_1),
                                          nn.Linear(head_hidden_1, head_hidden_2), self.act, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_2),
                                          nn.Linear(head_hidden_2, head_hidden_3), self.act, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_3),
                                          nn.Linear(head_hidden_3, 1))

        self.encoder_radio = nn.Sequential(nn.Linear(radio_f_dim, 4*output_dim), nn.LeakyReLU(), nn.Linear(4*output_dim, 2*output_dim), nn.LeakyReLU(), nn.Linear(2*output_dim, output_dim))

        self.log_alpha_radio = torch.nn.Parameter(torch.Tensor(1, output_dim))

        tmp_noise_par_radio = torch.FloatTensor(1, radio_f_dim).fill_(-3)

        self.W_out_logvar_radio = torch.nn.Parameter(data=tmp_noise_par_radio, requires_grad=True)

        torch.nn.init.normal_(self.log_alpha_radio, 0.0, 0.01)

        self.sparse = True
        self.n_channels = 1
        self.beta = 1.0
        self.enc_channels = list(range(self.n_channels))
        self.dec_channels = list(range(self.n_channels))
        for layer in self.head_mut:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)
        for layer in self.encoder_radio:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)




    def encode_radio(self, x):
        mu = self.encoder_radio(x)
        logvar = compute_logvar(mu, self.log_alpha_radio)
        return Normal(loc=mu, scale=logvar.exp().pow(0.5))


    def forward(self, radio_feats):

        q_radio = self.encode_radio(radio_feats)
        x = torch.concat([q_radio.loc], dim=1)
        y = self.head_mut(x)
        return y.squeeze(1)
