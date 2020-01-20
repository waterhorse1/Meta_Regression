import torch
import torch.nn as nn
import torch.nn.functional as F
from data.task_multi import multi
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import utils

class encoder(nn.Module):   
    def __init__(self, hidden_size = 10, batch_size = 32, latent_size = 10):
        super(encoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.l1 = nn.Linear(2, 64)
        self.l = nn.Linear(2, 10)
        self.mean = nn.Linear(self.hidden_size, self.latent_size)
        self.log_std = nn.Linear(self.hidden_size, self.latent_size)
        self.gru = nn.GRU(64, self.hidden_size, batch_first=True)
                           
    def forward(self, input): # input = batch * 10 * 2
        x = self.l1(input)# batch * 10 * 20
        hidden = self.hidden()
        output, hidden = self.gru(x, hidden)
        out = self.l(input)
        mean = self.mean(hidden)[0]# 1 * batch * latent
        log_std = self.log_std(hidden)[0]
        return mean, log_std, out
    
    def hidden(self):
        return torch.zeros((1, self.batch_size, self.hidden_size))
    
class decoder(nn.Module):   
    def __init__(self, reverse = True, hidden_size = 10, batch_size = 32):
        super(decoder, self).__init__()
        self.reverse = reverse
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(self.hidden_size,10)
        self.gru = nn.GRU(64, self.hidden_size, batch_first=True)
        self.l2 = nn.Linear(10,64)
                           
    def forward(self, hidden, step):# input = batch * 10 * 2
        out = []
        output = torch.zeros((self.batch_size, 1, 64))
        output, hidden = self.gru(output, hidden)#b*1*hidden_size
        output = self.l1(output)
        out.append(output)
        for step in range(step-1):
            input = self.l2(output)
            output, hidden = self.gru(input, hidden)
            output = self.l1(output)
            out.append(output)
        if self.reverse:
            out = out[::-1]
        
        out = torch.cat(out,dim=1)
        return out
    
    def hidden(self):
        return torch.zeros((1, self.batch_size, self.hidden_size))
    
class classifer(nn.Module):
    def __init__(self, input_size = 10, num=4):
        super(classifer, self).__init__()
        self.num = num
        self.l1 = nn.Linear(input_size, 50)
        self.l2 = nn.Linear(50, num)
    def forward(self, x):
        x1 = F.relu(self.l1(x))
        x2 = F.softmax(self.l2(x1), dim = -1)
        return x2
    
class sample(nn.Module):
    def __init__(self):
        super(sample, self).__init__()
    def forward(self, mean, log_std):
        epsilon = torch.randn((mean.shape[0], mean.shape[1]))
        return mean + torch.exp(log_std) * epsilon

class vae(nn.Module):
    def __init__(self, batch_size=32, hidden_size=32, latent_size=16, class_num = 4):
        super(vae, self).__init__()
        self.latent_size = latent_size
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.encoder = encoder(batch_size=batch_size, hidden_size=hidden_size, latent_size=latent_size)
        self.decoder = decoder(hidden_size = latent_size, batch_size = batch_size)
        self.classifer = classifer(input_size = latent_size, num=class_num)
        self.sample = sample()
        self.mean = nn.Parameter(torch.zeros([class_num, latent_size]))
        
    def forward(self, x, return_kl=False):# batch * point * 2
        mean, log_std, train_targets = self.encoder(x)# batch * ?
        z = self.sample(mean, log_std)# batch * latent_size
        hidden_state = torch.unsqueeze(z, dim = 0)# 1 * b * latent
        recon = self.decoder(hidden_state, step = x.shape[1])#b * point * 2(10)
        cate = self.classifer(z)# batch * class
        if return_kl:
            kl_loss, cat_loss = self.kl_loss(z, recon, cate, log_std)
            return z,recon,cate, train_targets, kl_loss, cat_loss
        else:
            return z,recon,cate, train_targets
    
    def kl_loss(self, z, recon, cate, z_log_std):
        expand_z = z.unsqueeze(dim = 1).expand([-1, self.class_num, -1])#batch * class * latent
        expand_mean = self.mean.unsqueeze(dim = 0).expand([self.batch_size, -1, -1])
        distance = expand_z - expand_mean
        kl_loss = - torch.sum(2 * z_log_std.unsqueeze(dim=1) - distance ** 2, dim = -1)#batch * class
        kl_loss = torch.mean(torch.sum(kl_loss * cate, dim=1))
        cat_loss = torch.mean(torch.sum(cate * torch.log(cate * self.class_num + 1e-10), dim = 1))
        return kl_loss, cat_loss
    
    def mse_loss(self, target, recon):
        return torch.sum(torch.mean((target-recon) ** 2,dim=0))