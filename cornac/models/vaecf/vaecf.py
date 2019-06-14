# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

from ...utils.common import scale
from ...utils.init_utils import normal
from ...utils import get_rng
from ...utils.data_utils import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math


class VAE(nn.Module):
    def __init__(self, dim_data, dim_z, h_dim, n_users, dim_f):
        super(VAE, self).__init__()

        self.n_users = n_users
        self.fc1_0 = nn.Linear(dim_data, h_dim)
        #self.fc1_01 = nn.Linear(2 * h_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, dim_z)
        self.fc22 = nn.Linear(h_dim, dim_z)
        self.fc3 = nn.Linear(dim_z, h_dim)
        self.fc3_1 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, dim_data)
        self.eps = 1e-10
        self.mu = torch.zeros([n_users, dim_z], dtype=torch.double, requires_grad=True)
        self.log_sig = torch.zeros([n_users, dim_z], dtype=torch.double, requires_grad=True)
        
        #
        self.p_fc1 = nn.Linear(dim_f, h_dim)
        self.p_fc21 = nn.Linear(h_dim, dim_z)
        
        #Parameters of the mixture
        self.mu1 = torch.rand([5, dim_z], dtype=torch.double, requires_grad=True)
        #self.mu1 = self.mu1.unsqueeze(0)
        #nn.init.kaiming_uniform_(self.mu1, a=math.sqrt(5))
        #self.mu1 = self.mu1.squeeze(0)
        #self.mu1 = torch.tensor(self.mu1, dtype=torch.double, requires_grad=True).requires_grad_()
        
        #self.mu2 = torch.rand([dim_z], dtype=torch.double, requires_grad=True)
        #self.mu3 = torch.rand([dim_z], dtype=torch.double, requires_grad=True)
        #self.mu4 = torch.rand([dim_z], dtype=torch.double, requires_grad=True)
        #self.mu5 = torch.rand([dim_z], dtype=torch.double, requires_grad=True)
        #self.mu2 = self.mu2.unsqueeze(0)
        #nn.init.kaiming_uniform_(self.mu2, a=math.sqrt(5))
        #self.mu2 = self.mu2.squeeze(0)
        
        #self.mu2 = torch.tensor(self.mu2, dtype=torch.double, requires_grad=True).requires_grad_()
        #print(self.mu1)

    def encode(self, x):
        h0 = F.relu(self.fc1_0(x))
        #h01 = F.relu(self.fc1_01(h0))
        h1 = h0
        #h1 = F.tanh(self.fc1(h0))
        return self.fc21(h1), self.fc22(h1)
    
    def prior_encoder(self, f):
        h0 = F.relu(self.p_fc1(f))
        #h01 = F.relu(self.fc1_01(h0))
        h1 = h0
        #h1 = F.tanh(self.fc1(h0))
        return self.p_fc21(h1)
    
    def mog_prior(self, z):
        """
        v1 = -0.5*(z-self.mu1)**2
        v1 = torch.sum(v1, dim = 1)
        v1 = v1.unsqueeze(1)
        
        v2 = -0.5*(z-self.mu2)**2 # B x K
        v2  = torch.sum(v2, dim = 1) # B x 1
        v2 = v2.unsqueeze(1)
        
        v3 = -0.5*(z-self.mu3)**2 # B x K
        v3  = torch.sum(v3, dim = 1) # B x 1
        v3 = v3.unsqueeze(1)
        
        v4 = -0.5*(z-self.mu4)**2 # B x K
        v4  = torch.sum(v4, dim = 1) # B x 1
        v4 = v4.unsqueeze(1)
        
        v5 = -0.5*(z-self.mu5)**2 # B x K
        v5  = torch.sum(v5, dim = 1) # B x 1
        v5 = v5.unsqueeze(1)
        
        v = torch.cat([v1,v2, v3, v4, v5], dim = 1) # B x C
        #v = torch.cat([v1,v2], dim = 1)
        """
        v = -0.5*(z.unsqueeze(1) - self.mu1.unsqueeze(0))**2
        #print(v.shape)
        v = torch.sum(v, dim = 2)
        
        
        a_max, _ = torch.max(v, 1)  # B
        #a_max = a_max.unsqueeze(1)  # B x 1
        
        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(v - a_max.unsqueeze(1)), 1))  # B x 1
        
        return log_prior
    
    
    def aux_prior(self, z, batch_f):
        
        
        mu = self.prior_encoder(batch_f)
        
        v = -0.5*(z.unsqueeze(1) - mu.unsqueeze(0))**2
        v = torch.sum(v, dim = 2)
        
        a_max, _ = torch.max(v, 1)  # B
        #a_max = a_max.unsqueeze(1)  # B x 1
        
        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(v - a_max.unsqueeze(1)), 1))  # B x 1
        
        return log_prior
        

    def reparameterize(self, mu, logvar):
        #std = 0.01  # torch.exp(0.5*logvar)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        #h31 = F.tanh(self.fc3_1(h3))
        return F.sigmoid(self.fc4(h3))
        #return self.fc4(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar, self.fc4, self.fc1, self.fc21, self.fc3

    def loss(self, recon_x, bin_, f, x, mu, logvar, mu2, logvar2):
        # - Poisson likelihood
        #pss = -x * torch.log(recon_x + self.eps) +  (bin_+ 0.5)*recon_x
        pss = -x * torch.log(recon_x + self.eps) -  0.5*(1-x) * torch.log(1 - recon_x + self.eps)
        #pss = (bin_ + 0.1)*(x - recon_x)**2
        pss = torch.sum(pss, dim=1)
        
        #reg = torch.sum(self.fc4.weight**2)
        #pss = F.binary_cross_entropy(recon_x, x, reduce=True, reduction='mean')
        std = torch.exp(0.5*logvar)
        std2 = torch.exp(0.5*logvar2)
        kld = -0.5 * (1 + 2. * torch.log(std) - 2. * torch.log(std2) - (mu -mu2).pow(2)/std2.pow(2) - std.pow(2)/std2.pow(2))
        kld2 = -0.5 * (1 + 2. * torch.log(std) - mu.pow(2) - std.pow(2))
        z_t = self.reparameterize(mu, logvar)
        #kld3 = -0.5 * (1 + 2. * torch.log(std) - z_t.pow(2))
        kld3 = -0.5 * (1 + 2. * torch.log(std))
        
        
        kld = torch.sum(kld, dim=1)
        kld2 = torch.sum(kld2, dim=1)
        kld3 = torch.sum(kld3, dim=1)
        #mm = self.mog_prior(z_t)
        mm = self.aux_prior(z_t, f)
        
        return torch.mean(pss + kld3 - mm), kld3



def learn(train_set, f, k, h_dim, n_epochs, batch_size, learn_rate, gamma, init_params, use_gpu, verbose, seed):

    torch.manual_seed(123)
    print(f.shape)
    print(train_set.matrix.shape)
    # parameter initialization
    users = Dataset(train_set.matrix)
    binmat = users.data.copy()
    binmat.data = np.ones(len( binmat.data )) #- 0.5
    
    users_bin = Dataset(binmat)
    features = Dataset(f)
    
    dim_data = users.data.shape[1]
    n_users = users.data.shape[0]
    dim_z = k
    learn_r = learn_rate
    kl_vals = torch.zeros([users.data.shape[0]], dtype=torch.double)
    f_vals = torch.zeros([users.data.shape[0]], dtype=torch.double)

    vae = VAE(dim_data, dim_z, h_dim, n_users, n_users)
    
    m_params = [{'params': vae.fc3.parameters()}, \
               {'params': vae.fc4.parameters()}, \
               {'params': vae.mu}, \
               {'params': vae.log_sig}\
               ]
    
    v_params = [{'params': vae.fc1_0.parameters()}, \
                {'params': vae.fc21.parameters()}, \
                {'params': vae.fc22.parameters()} \
             ]
    
    
    params = list(vae.parameters())
    params.append(vae.mu1)
    #params.append(vae.mu2)
    #params.append(vae.mu3)
    #params.append(vae.mu4)
    #params.append(vae.mu5)
    
    #optimizer = torch.optim.RMSprop(params=params, lr=learn_r, alpha=gamma)
    optimizer = torch.optim.Adam(params=params, lr=learn_r)
    optimizer_m = torch.optim.RMSprop(params=m_params, lr=learn_r, alpha=gamma)
    optimizer_v = torch.optim.RMSprop(params=v_params, lr=learn_r, alpha=gamma)

    for epoch in range(1, n_epochs + 1):
        sum_loss = 0.
        count = 0
        num_steps = int(users.data.shape[0] / batch_size)
        progress_bar = tqdm(total=num_steps,
                            desc='Epoch {}/{}'.format(epoch, n_epochs),
                            disable=False)

        for i in range(1, num_steps + 1):
            u_batch, u_ids = users.next_batch(batch_size)
            u_batch.data =  np.ones(len(u_batch.data))
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=torch.double)
            
            bin_batch, u_ids = users_bin.next_batch(batch_size)
            f_batch, _ = features.next_batch(batch_size*5)
            #f_batch = torch.tensor(f[u_ids], dtype=torch.double)
            f_batch = torch.tensor(f_batch.A, dtype = torch.double)
            # update variational prameters
            z_batch, recon_batch, mu, logvar, beta, fc1, fc21, fc3 = vae(u_batch)
            #bin_ = torch.tensor(binmat[u_ids].A, dtype=torch.double)
            bin_ = torch.tensor(bin_batch.A, dtype=torch.double)
            loss, kl_vals[u_ids] = vae.loss(recon_batch, bin_, f_batch, u_batch, mu, logvar, vae.mu[u_ids], vae.log_sig[u_ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(u_batch)
            if count % (batch_size * 10) == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))
            progress_bar.update(1)
        progress_bar.close()
        print(sum_loss)


    disc = Discriminator(dim_z,10)
    #optimizer_d = torch.optim.RMSprop(params=list(disc.parameters()), lr=learn_r*0.1, alpha=gamma)
    optimizer_d = torch.optim.Adam(disc.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()


    """# train the discriminator here to estimate mutual information
    batch_size = int(batch_size/2)
    for epoch in range(1, n_epochs + 1):
        sum_loss_d = 0.
        count = 0
        num_steps = int(users.data.shape[0] / batch_size)
        progress_bar_d = tqdm(total=num_steps,
                            desc='Epoch {}/{}'.format(epoch, n_epochs),
                            disable=False)

        for i in range(1, num_steps + 1):
            u_batch, u_ids = users.next_batch(batch_size)
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=torch.double)

            z_real, recon_batch, mu, logvar, beta, fc1, fc21, fc3 = vae(u_batch)
            real_label = torch.ones([batch_size,1], dtype=torch.double, requires_grad=False)

            z_fake = torch.randn_like(z_real)
            fake_label = torch.zeros([batch_size,1], dtype=torch.double, requires_grad=False)

            z_batch = torch.cat([z_real, z_fake], dim=0)
            labels = torch.cat([real_label, fake_label])

            x_, ff = disc(z_batch)
            #loss_d = disc.loss(x_, labels)
            loss_d = criterion(x_,labels)

            f_vals[u_ids] = ff[0:batch_size, 0]

            optimizer_d.zero_grad()
            
            loss_d.backward()
            optimizer_d.step()

            sum_loss_d += loss_d.data.item()
            count += len(u_batch)
            if count % (batch_size * 10) == 0:
                progress_bar_d.set_postfix(loss=(sum_loss_d / count))
            progress_bar_d.update(1)
        progress_bar_d.close()
        print(sum_loss_d)
        """

    return vae, disc, kl_vals.data.cpu().numpy(), f_vals.data.cpu().numpy()






class Discriminator(nn.Module):
    def __init__(self, z_dim, h_dim):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(z_dim, 2*h_dim)
        self.fc2 = nn.Linear(2*h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        return F.sigmoid(h3), h3

    def forward(self, x):
        x_, f = self.encode(x)
        return x_, f

    def loss(self, x_, x):
        return F.binary_cross_entropy(x_, x, reduce=True, reduction='mean')



"""
for i in range(1, num_steps + 1):
    u_batch, u_ids = users.next_batch(batch_size)
    u_batch.data =  np.ones(len(u_batch.data))
    u_batch = u_batch.A
    u_batch = torch.tensor(u_batch, dtype=torch.double)
    # update model params
    z_batch, recon_batch, mu, logvar, beta, fc1, fc21, fc3 = vae(u_batch)
    bin_ = torch.tensor(binmat[u_ids].A, dtype=torch.double)
    loss, kl_vals[u_ids] = vae.loss(recon_batch, bin_, u_batch, mu, logvar, vae.mu[u_ids], vae.log_sig[u_ids])
    optimizer_m.zero_grad()
    loss.backward()
    optimizer_m.step()
    
    # update variational prameters
    z_batch, recon_batch, mu, logvar, beta, fc1, fc21, fc3 = vae(u_batch)
    bin_ = torch.tensor(binmat[u_ids].A, dtype=torch.double)
    loss, kl_vals[u_ids] = vae.loss(recon_batch, bin_, u_batch, mu, logvar, vae.mu[u_ids], vae.log_sig[u_ids])
    optimizer_v.zero_grad()
    loss.backward()
    optimizer_v.step()
"""








torch.set_default_dtype(torch.double)


def _load_or_randn(size, init_values, seed, device):
    if init_values is None:
        rng = get_rng(seed)
        tensor = normal(size, mean=0.0, std=0.001, random_state=rng, dtype=np.double)
        tensor = torch.tensor(tensor, requires_grad=True, device=device)
    else:
        tensor = torch.tensor(init_values, requires_grad=True, device=device)
    return tensor


def _l2_loss(*tensors):
    l2_loss = 0
    for tensor in tensors:
        l2_loss += torch.sum(tensor ** 2) / 2
    return l2_loss