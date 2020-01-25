import torch
import torch.nn.functional as F
from torch import nn

class place(nn.Module):
    def __init__(self):
        super(place, self).__init__()
        self.l1 = nn.Linear(10, 50)
        #self.l2 = nn.Linear(30, 40)
        self.l3 = nn.Linear(50, 30)
    def forward(self, x):
        x = F.relu(self.l1(x))
        #x = F.relu(self.l2(x))
        logits = self.l3(x)
        return logits
    
class pool_encoder(nn.Module):
    def __init__(self):
        super(pool_encoder, self).__init__()
        self.l1 = nn.Linear(2, 50)
        self.l2 = nn.Linear(50, 10)
        #self.l3 = nn.Linear(2, 10)
    def forward(self, x, return_embed = False):
        x1 = F.relu(self.l1(x))
        x2 = self.l2(x1)
        '''
        if return_embed:
            embed = self.l3(x)
            return x2, embed
        else:
            return x2
        '''
        return x2
    
class pool_decoder(nn.Module):
    def __init__(self):
        super(pool_decoder, self).__init__()
        self.l1 = nn.Linear(15, 50)
        self.l2 = nn.Linear(50,10)
    def forward(self, x):
        x1 = F.relu(self.l1(x))
        x2 = self.l2(x1)
        return x2    
    
class CaviaModel(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 n_in,
                 n_out,
                 num_context_params,
                 n_hidden,
                 device
                 ):
        super(CaviaModel, self).__init__()
        assert len(num_context_params) == len(n_hidden) + 1
        self.device = device

        # fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_in + num_context_params[0], n_hidden[0]))
        #self.fc_layers.append(nn.Linear(n_in, n_hidden[0]))
        for k in range(len(n_hidden) - 1):
            self.fc_layers.append(nn.Linear(n_hidden[k] + num_context_params[k+1], n_hidden[k + 1]))
        self.fc_layers.append(nn.Linear(n_hidden[-1] + num_context_params[-1], n_out))

        # context parameters (note that these are *not* registered parameters of the model!)
        self.num_context_params = num_context_params
        self.context_params = None
        self.reset_context_params()

    def reset_context_params(self):
        self.context_params = torch.zeros(sum(self.num_context_params)).to(self.device)
        self.context_params.requires_grad = True
        
    def set_context_params(self, x):
        self.context_params = x[0]
        #self.context_params.detach()
        
    def forward(self, x):

        # concatenate input with context parameters

        for k in range(len(self.fc_layers) - 1):
            if k == 0:
                x = torch.cat((x, self.context_params[0:self.num_context_params[0]]\
                               .expand(x.shape[0], -1)), dim=1)
            else:
                x = torch.cat((x, self.context_params[sum(self.num_context_params[:k]):\
                                                      sum(self.num_context_params[:k+1])]\
                               .expand(x.shape[0], -1)), dim=1)
            x = F.relu(self.fc_layers[k](x))                
                
        k += 1
        x = torch.cat((x, self.context_params[sum(self.num_context_params[:k]):]\
                        .expand(x.shape[0], -1)), dim=1)
        #print(self.context_params[-self.num_context_params[k]:].shape)
        y = self.fc_layers[-1](x)

        return y
