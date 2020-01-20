class encoder(nn.Module):   
    def __init__(self, hidden_size = 20, batch_size = 32):
        super(encoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(2, 2)
        self.gru = nn.GRU(2, self.hidden_size, batch_first=True)
                           
    def forward(self, input): # input = batch * 10 * 2
        x = self.l1(input)# batch * 10 * 20
        hidden = self.hidden()
        output, hidden = self.gru(x, hidden)
        out = x
        return hidden, out, output
    
    def hidden(self):
        return torch.zeros((1, self.batch_size, self.hidden_size))
    
class decoder(nn.Module):   
    def __init__(self, reverse = True, hidden_size = 20, batch_size = 32):
        super(decoder, self).__init__()
        self.reverse = reverse
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(self.hidden_size,2)
        self.gru = nn.GRU(2, self.hidden_size, batch_first=True)
        #self.l2 = nn.Linear(2,64)
                           
    def forward(self, hidden, step):# input = batch * 10 * 2
        out = []
        output = torch.zeros((self.batch_size, 1, 2))
        output, hidden = self.gru(output, hidden)#b*1*hidden_size
        output = self.l1(output)
        out.append(output)
        for step in range(step-1):
            #input = self.l2(output)
            output, hidden = self.gru(output, hidden)
            output = self.l1(output)
            out.append(output)
        if self.reverse:
            out = out[::-1]
        
        out = torch.cat(out,dim=1)
        return out
    
    def hidden(self):
        return torch.zeros((1, self.batch_size, self.hidden_size))
    
def RNN_AE(nn.Module):
    def __init__(self, batch_size=32, hidden_size=32):
        super(RNN_AE, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.encoder = encoder(batch_size=batch_size, hidden_size=hidden_size)
        self.decoder = decoder(hidden_size = hidden_size, batch_size = batch_size)
    def forward(self, x):
        z, target, output = self.encoder(x)# batch * ?
        recon = self.decoder(hidden_state, step = x.shape[1])#b * point * 2(10)
        return recon, z, target,output
    


        