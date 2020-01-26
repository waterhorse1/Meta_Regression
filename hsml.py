import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from task_embedding_GRU import RNN_AE

class layer(nn.Module):
    def __init__(self, num, sigma = 1,hidden_dim=20, input_dim = 20):
        super(layer, self).__init__()
        self.num = num 
        self.sigma = sigma
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.center = nn.ParameterList([nn.Parameter(torch.zeros([input_dim])) for _ in range(num)])
        self.transform = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num)])
    def forward(self, h):
        #h -> b*p_num * hidden
        #batch * p_num * hidden -> b * p_num * num
        dis = []
        for c in self.center:
            distance = torch.sum((h - c.view(1,1,-1))**2, dim=-1)/(2.0 * self.sigma)#b*p_num
            dis.append(distance)
        dis = torch.stack(dis, dim = -1) #b*p_num*num
        prob = F.softmax(-dis, dim = -1) #b * p * num
        # b * p * num & b * p
        all_hidden = []
        for i in range(h.shape[1]):
            hidden = h[:,i,:]
            post = []
            for linear in self.transform:
                post_linear = linear(hidden)#b*after_linear_hidden
                post.append(post_linear)
            post = torch.stack(post, dim=1)#b * num * after_hidden
            all_hidden.append(post)
        all_hidden = torch.stack(all_hidden, dim=1)#b * p_num * num * after_hidden
        all_hidden = torch.sum(prob.unsqueeze(dim=-1) * all_hidden, dim = 1)# b * num * after_hidden
        return all_hidden
    
class clustering(nn.Module):
    def __init__(self, layer_unit = [4,2,1]):
        super(clustering, self).__init__()
        self.layer_all = [layer(num) for num in layer_unit]
    def forward(self, x):
        for l in self.layer_all:
            x = l(x)
        return x

class HSML(nn.Module):
    def __init__(self):
        super(HSML, self).__init__()
        self.cluster = clustering()
        self.rnn = RNN_AE(hidden_size=20)
    def forward(self, x):
        recon, z, target,output = self.rnn(x)
        mseloss = nn.MSELoss()(recon, target)
        cluster_result = self.cluster(z[0].unsqueeze(dim=1))
        cluster_result = cluster_result.squeeze(dim = 1)#batch*20
        gate = torch.cat([z[0],cluster_result], dim = -1)
        return gate, mseloss