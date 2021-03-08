import torch
import torch.nn.functional as F
from torch import nn


class simple_MLP(nn.Module):
    def __init__(self):
        super(simple_MLP, self).__init__()
        self.linear = F.linear
    def forward(self, x, new_paras):
        l1 = F.linear(x, weight = new_paras[:50].view((50,1)), bias = new_paras[50:100])
        l2 = F.relu(F.linear(l1, weight = new_paras[100:2600].view((50,50)), bias = new_paras[2600:2650]))
        l3 = F.relu(F.linear(l2, weight = new_paras[2650:5150].view((50,50)), bias = new_paras[5150:5200]))
        l4 = F.linear(l3, weight = new_paras[5200:5250].view((1,50)), bias = new_paras[5250:5251])
        return l4
