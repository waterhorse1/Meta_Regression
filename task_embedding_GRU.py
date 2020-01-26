import torch
import torch.nn as nn
import torch.nn.functional as F
class encoder(nn.Module):   
    def __init__(self, hidden_size = 20):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(2, 2)
        self.gru = nn.GRU(2, self.hidden_size, batch_first=True)
                           
    def forward(self, input): # input = batch * 10 * 2
        batch_size = input.shape[0]
        x = self.l1(input)# batch * 10 * 20
        hidden = self.hidden(batch_size)
        output, hidden = self.gru(x, hidden)
        out = x
        return hidden, out, output
    
    def hidden(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size))
    
class decoder(nn.Module):   
    def __init__(self, reverse = True, hidden_size = 20):
        super(decoder, self).__init__()
        self.reverse = reverse
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(self.hidden_size,2)
        self.gru = nn.GRU(2, self.hidden_size, batch_first=True)
                           
    def forward(self, hidden, step):
        out = []
        output = torch.zeros((hidden.shape[1], 1, 2))
        output, hidden = self.gru(output, hidden)#b*1*hidden_size
       
        output = self.l1(output)
        out.append(output)
        for step in range(step-1):
            output, hidden = self.gru(output, hidden)
            output = self.l1(output)
            out.append(output)
        if self.reverse:
            out = out[::-1]
        
        out = torch.cat(out,dim=1)
        return out
    
    def hidden(self):
        return torch.zeros((1, self.batch_size, self.hidden_size))
    
class RNN_AE(nn.Module):
    def __init__(self,  hidden_size=32):
        super(RNN_AE, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = encoder(hidden_size=hidden_size)
        self.decoder = decoder(hidden_size = hidden_size)
    def forward(self, x):
        z, target, output = self.encoder(x)# batch * ?
        recon = self.decoder(z, step = x.shape[1])#b * point * 2(10)
        return recon, z, target,output
    


        