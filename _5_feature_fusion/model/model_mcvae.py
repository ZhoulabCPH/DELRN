import torch
from torch import nn
from _5_feature_fusion.model.mcVAE import Normal, compute_kl, compute_ll, compute_logvar

class mcVAE_BRCA_mut(nn.Module):
    def __init__(self, radio_f_dim=1580, deep_f_dim=256, output_dim=128, head_hidden_1=128, head_hidden_2=64, head_hidden_3=32,dropout=0.4, head_id=0, actf='ReLU'):
        super(mcVAE_BRCA_mut, self).__init__()

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
            self.head_mut = nn.Sequential(nn.Linear(output_dim*2, head_hidden_1), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_1, 1))
        elif head_id == 1:
            self.head_mut = nn.Sequential(nn.Linear(output_dim*2, head_hidden_1), self.act, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_1),
                                          nn.Linear(head_hidden_1, 1))
        elif head_id == 2:
            self.head_mut = nn.Sequential(nn.Linear(output_dim*2, head_hidden_1), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_1, head_hidden_2), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_2, 1))
        elif head_id == 3:
            self.head_mut = nn.Sequential(nn.Linear(output_dim*2, head_hidden_1), self.act1, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_1),
                                          nn.Linear(head_hidden_1, head_hidden_2), self.act2, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_2),
                                          nn.Linear(head_hidden_2, 1))
        elif head_id == 4:
            self.head_mut = nn.Sequential(nn.Linear(output_dim*2, head_hidden_1), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_1, head_hidden_2), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_2, head_hidden_3), self.act, nn.Dropout(dropout),
                                          nn.Linear(head_hidden_3, 1))
        elif head_id == 5:
            self.head_mut = nn.Sequential(nn.Linear(output_dim*2, head_hidden_1), self.act, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_1),
                                          nn.Linear(head_hidden_1, head_hidden_2), self.act, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_2),
                                          nn.Linear(head_hidden_2, head_hidden_3), self.act, nn.Dropout(dropout), nn.BatchNorm1d(head_hidden_3),
                                          nn.Linear(head_hidden_3, 1))

        self.encoder_patho = nn.Sequential(nn.Linear(radio_f_dim, 4*output_dim), nn.LeakyReLU(), nn.Linear(4*output_dim, 2*output_dim), nn.LeakyReLU(), nn.Linear(2*output_dim, output_dim))
        self.encoder_pheno = nn.Sequential(nn.Linear(deep_f_dim, 4*output_dim), nn.LeakyReLU(), nn.Linear(4*output_dim, 2*output_dim), nn.LeakyReLU(), nn.Linear(2*output_dim, output_dim))
        self.decoder_patho = nn.Sequential(nn.Linear(output_dim, 2*output_dim), nn.LeakyReLU(), nn.Linear(2*output_dim, 4*output_dim), nn.LeakyReLU(), nn.Linear(4*output_dim, radio_f_dim))
        self.decoder_pheno = nn.Sequential(nn.Linear(output_dim, 2*output_dim), nn.LeakyReLU(), nn.Linear(2*output_dim, 4*output_dim), nn.LeakyReLU(), nn.Linear(4*output_dim, deep_f_dim))
        self.log_alpha_patho = torch.nn.Parameter(torch.Tensor(1, output_dim))
        self.log_alpha_pheno = torch.nn.Parameter(torch.Tensor(1, output_dim))
        tmp_noise_par_patho = torch.FloatTensor(1, radio_f_dim).fill_(-3)
        tmp_noise_par_pheno = torch.FloatTensor(1, deep_f_dim).fill_(-3)
        self.W_out_logvar_patho = torch.nn.Parameter(data=tmp_noise_par_patho, requires_grad=True)
        self.W_out_logvar_pheno = torch.nn.Parameter(data=tmp_noise_par_pheno, requires_grad=True)
        torch.nn.init.normal_(self.log_alpha_patho, 0.0, 0.01)
        torch.nn.init.normal_(self.log_alpha_pheno, 0.0, 0.01)
        self.sparse = True
        self.n_channels = 2
        self.beta = 1.0
        self.enc_channels = list(range(self.n_channels))
        self.dec_channels = list(range(self.n_channels))
        for layer in self.head_mut:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)
        for layer in self.encoder_patho:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)
        for layer in self.encoder_pheno:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)
        for layer in self.decoder_patho:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)
        for layer in self.decoder_pheno:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)

    def encode_patho(self, x):
        mu = self.encoder_patho(x)
        logvar = compute_logvar(mu, self.log_alpha_patho)
        return Normal(loc=mu, scale=logvar.exp().pow(0.5))

    def decode_patho(self, z):
        pi = Normal(
            loc=self.decoder_patho(z),
            scale=self.W_out_logvar_patho.exp().pow(0.5)
        )
        return pi

    def encode_pheno(self, x):
        mu = self.encoder_pheno(x)
        logvar = compute_logvar(mu, self.log_alpha_pheno)
        return Normal(loc=mu, scale=logvar.exp().pow(0.5))

    def compute_kl(self, q):
        kl = 0
        for i, qi in enumerate(q):
            if i in self.enc_channels:
                # "compute_kl" ignores p2 if sparse=True.
                kl += compute_kl(p1=qi, p2=Normal(0, 1), sparse=self.sparse)

        return kl

    def compute_ll(self, p, x):
        # p[x][z]: p(x|z)
        ll = 0
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                # xi = reconstructed; zj = encoding
                if i in self.dec_channels and j in self.enc_channels:
                    ll += compute_ll(p=p[i][j], x=x[i])

        return ll
    def decode_pheno(self, z):
        pi = Normal(
            loc=self.decoder_pheno(z),
            scale=self.W_out_logvar_pheno.exp().pow(0.5)
        )
        return pi

    def loss_function(self, fwd_ret):
        x = fwd_ret['x']
        q = fwd_ret['q']
        p = fwd_ret['p']

        kl = self.compute_kl(q)
        kl *= self.beta
        ll = self.compute_ll(p=p, x=x)

        total = kl - ll
        return total

    def forward(self, patho_feats,pheno_feats):

        q_patho = self.encode_patho(patho_feats)
        q_pheno = self.encode_pheno(pheno_feats)
        x = torch.concat([q_patho.loc, q_pheno.loc], dim=1)
        y = self.head_mut(x)
        return y.squeeze(1)
