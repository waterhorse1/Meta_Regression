import torch
import torch.nn.functional as F
from torch import nn

class pool_encoder(nn.Module):
    def __init__(self):
        super(pool_encoder, self).__init__()
        self.l1 = nn.Linear(2, 50)
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

        self.device = device

        # fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_in + num_context_params, n_hidden[0]))
        #self.fc_layers.append(nn.Linear(n_in, n_hidden[0]))
        for k in range(len(n_hidden) - 1):
            self.fc_layers.append(nn.Linear(n_hidden[k]+ num_context_params, n_hidden[k + 1]))
        self.fc_layers.append(nn.Linear(n_hidden[-1]+ num_context_params, n_out))

        # context parameters (note that these are *not* registered parameters of the model!)
        self.num_context_params = num_context_params
        self.context_params = []
        self.reset_context_params()

    def reset_context_params(self):
        self.context_params = []
        for i in range(len(self.fc_layers)):
            params = torch.zeros(self.num_context_params)#self.fc_layers[i].weight.shape[0]).to(self.device)
            params.requires_grad = True
            self.context_params.append(params)
        
        '''
        self.param_1 = torch.ones(int(self.fc_layers[1].weight.shape[0]/2),requires_grad=True)
        length = self.fc_layers[1].weight.shape[0] - int(self.fc_layers[1].weight.shape[0]/2)
        self.param_2 = torch.ones(length, requires_grad = True)
        '''
        
    def set_context_params(self, x, m):
        self.context_params[m] = x.detach()
        self.context_params[m].requires_grad = True
        #self.context_params[m].detach()
        
    def forward(self, x):

        # concatenate input with context parameters
        #x = torch.cat((x, self.context_params[0].expand(x.shape[0], -1)), dim=1)

        for k in range(len(self.fc_layers) - 1):
            x = torch.cat((x, self.context_params[k].expand(x.shape[0], -1)), dim=1)
            #print(x.shape)
            x = F.relu(self.fc_layers[k](x))
            '''
            if k == 1:
                context = torch.cat([self.param_1,self.param_2])
                x = context * x 
            '''
        x = torch.cat((x, self.context_params[-1].expand(x.shape[0], -1)), dim=1)
        y = self.fc_layers[-1](x)

        return y
