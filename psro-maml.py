import numpy as np
from random import shuffle
from scipy.stats import entropy
# import pandas as pd
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import math
import time
import random
import os
from scipy import stats
import glob
import copy
np.set_printoptions(suppress=True)
#from tqdm import tqdm
#from scipy.special import softmax
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)
#np.random.seed(0)
from tensorboardX import SummaryWriter
import nashpy as nash
import argparse
import copy
import time
import json

from maml_model import MamlModel
from arguments import parse_args_psro

#Search over the pure strategies to find the BR to a strategy
def get_br_to_strat(strat, payoffs, verbose=False):
    row_weighted_payouts = strat@payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br
    

def get_exploitability_direct(metanash, payoffs, popn=True, pop=None):
    if popn == True:
      numpy_pop = pop.pop2numpy(K=pop.pop_size)
      meta_nash_numpy = metanash
      emp_game_matrix = numpy_pop @ payoffs @ numpy_pop.T 
      strat = meta_nash_numpy @ numpy_pop
    
    else:
      strat = metanash
      
    test_br = get_br_to_strat(strat, payoffs=payoffs)
    exp = test_br @ payoffs @ strat
    return 2 * exp
    
def compare_exploit(pop, mod, payoffs, K):
  fic_play_strat, _  = fictitious_play(payoffs=pop.get_metagame(payoffs, K).cpu().detach().numpy())
  mod_exp = get_exploitability_direct(mod(pop.get_metagame(payoffs, K).float().cuda()[None,None,]).cpu().detach().numpy()[0], payoffs, pop=pop)
  fic_play_exp = get_exploitability_direct(fic_play_strat[-1], payoffs, popn=True, pop=t_pop)
  true_nash, _ = fictitious_play(payoffs=payoffs)
  true_exp = get_exploitability_direct(true_nash[-1], payoffs, popn=False)
  return [mod_exp, fic_play_exp, true_exp]
  
def compare_pop_strats(pop, mod, payoffs, K):
  fic_n, _ = fictitious_play(payoffs=pop.get_metagame(payoffs, K).cpu().detach().numpy())
  np_pop = pop.pop2numpy(K)
  mod_strat = mod(pop.get_metagame(payoffs, K).float().cuda()[None,None,]).cpu().detach().numpy()[0]
  mod_agg_strat = mod_strat @ np_pop
  fic_play_strat = fic_n[-1] @ np_pop
  true_nash_strat, _ = fictitious_play(payoffs=payoffs)
  return [mod_agg_strat, fic_play_strat, true_nash_strat[-1]]  
    

class psro(nn.Module):
    def __init__(self, pop_size):
        super().__init__()
        self.pop_size = pop_size
        self.maml_pop = [MamlModel().to(args.device) for _ in range(self.pop_size)]
        self.par_pop = [[torch.tensor(np.random.uniform(0.1,5)), torch.tensor(np.random.uniform(0, np.pi))] for _ in range(self.pop_size)]
        print(self.par_pop)
        self.pop_size = len(self.maml_pop)

    #def add_agent(self):
    #    self.maml_pop.append(MamlModel().to(args.device))
    #    self.p_pop.append([np.random.uniform(0.1,5), np.random.uniform(0, np.pi)])
    #    self.pop_size = len(self.pop)
    
    def update(self, new_maml_agent, new_par_agent):
        self.maml_pop.append(new_maml_agent)
        self.par_pop.append(new_par_agent)
        self.pop_size += 1
        
    def get_metagame(self, k=None):
        n = 3
        if k==None:
            k = self.pop_size
        metagame = torch.zeros(k,k)
        for i in range(k):
            for j in range(k):
                loss = 0
                for m in range(n):
                    loss += gd_process(self.maml_pop[i], self.par_pop[j])/n
                metagame[i, j] = loss
        return metagame.detach().numpy()

    def get_exploitability(self, metanash, payoffs, train_steps, lr):
        #print('exp')
        new_agent = torchAgent(self.n_actions)
        #new_agent.pop_logits = torch.zeros_like(new_agent.pop_logits)
        new_agent.start_train()
        for iter in range(train_steps):
            #print(iter)
            exp_payoff = self.payoff_aggregate(new_agent, metanash, payoffs, K = self.pop_size)

            loss = -(exp_payoff)
            #print(exp_payoff)
            grad = torch.autograd.grad(loss, [new_agent.pop_logits], create_graph=True)
            new_agent.pop_logits = new_agent.pop_logits - lr * grad[0]
        #true_exp = self.get_exploitability_direct(metanash, payoffs)
        exp1 = self.payoff_aggregate(new_agent, metanash, payoffs, K=self.pop_size)
        return 2 * exp1

def gd_process(model, par):
    w, b = par
    x_spt = torch.rand(args.k_meta_train, 1) * 10 - 5
    y_spt = w * torch.sin(x_spt - b)
    x_qry = torch.rand(args.k_meta_test, 1) * 10 - 5
    y_qry = w * torch.sin(x_qry - b)

    #maml process
    fast_parameters = model.parameters()
    for weight in model.parameters():
        weight.fast = None
    for k in range(args.num_inner_updates):
        logits = model(x_spt)
        loss = F.mse_loss(logits, y_spt)
        grad = torch.autograd.grad(loss, fast_parameters, create_graph=args.first_order)
        fast_parameters = []
        for k, weight in enumerate(model.parameters()):
            if args.first_order:
                if weight.fast is None:
                    weight.fast = weight - args.lr_inner * grad[k].detach() #create weight.fast 
                else:
                    weight.fast = weight.fast - args.lr_inner * grad[k].detach()
            else:
                if weight.fast is None:
                    weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - args.lr_inner * grad[k]
            fast_parameters.append(weight.fast)         

        logits_q = model(x_qry)
        # loss_q will be overwritten and just keep the loss_q on last update step.
        loss_q = F.mse_loss(logits_q, y_qry)
        
    return loss_q

def train_maml_agent(psro_pop, dis):
    model = MamlModel().to(args.device)
    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    batch_size = args.maml_bs
    for i in range(args.maml_train_iter):
        meta_grad = copy.deepcopy(meta_grad_init)
        #generate batch of tasks according to meta policy distribution
        task_index = np.random.choice(psro_pop.pop_size, batch_size, dis)
        
        for k in range(len(task_index)):
            w, b = psro_pop.par_pop[task_index[k]]
            
            loss_q = gd_process(model, [w, b])
            task_grad_test = torch.autograd.grad(loss_q, model.parameters())

            for g in range(len(task_grad_test)):
                meta_grad[g] += task_grad_test[g].detach()
            # -------------- meta update --------------
        meta_optimiser.zero_grad()

        # set gradients of parameters manually
        for c, param in enumerate(model.parameters()):
            param.grad = meta_grad[c] / float(batch_size)
            param.grad.data.clamp_(-10, 10)

        # the meta-optimiser only operates on the shared parameters, not the context parameters
        meta_optimiser.step()
    model.eval()
    return model
        

def train_par_agent(psro_pop, dis):
    w = torch.tensor(np.random.uniform(0.1, 5))
    w.requires_grad = True
    b = torch.tensor(np.random.uniform(0, np.pi))
    b.requires_grad = True
    meta_optimiser = torch.optim.Adam([w,b], args.lr_par)
    meta_grad_init = [0, 0]
    batch_size = args.par_bs
    for i in range(args.par_train_iter):
        meta_grad = copy.deepcopy(meta_grad_init)
        #generate batch of tasks according to meta policy distribution
        task_index = np.random.choice(psro_pop.pop_size, batch_size, dis)
        
        for k in range(len(task_index)):
            model = psro_pop.maml_pop[task_index[k]]
            
            loss_q = -gd_process(model, [w,b])
            task_grad_test = torch.autograd.grad(loss_q, [w,b])

            for g in range(len(task_grad_test)):
                meta_grad[g] += task_grad_test[g].detach()
                    
            # -------------- meta update --------------
            
        meta_optimiser.zero_grad()

        # set gradients of parameters manually
        for c, param in enumerate([w, b]):
            param.grad = meta_grad[c] / float(batch_size)
            param.grad.data.clamp_(-10, 10)

        # the meta-optimiser only operates on the shared parameters, not the context parameters
        meta_optimiser.step()
        
    w.requires_grad = False
    b.requires_grad = False
    return [w,b]

def metanash(payoff):
    A = - payoff
    B = A
    game = nash.Game(A, B)
    play_counts = tuple(game.fictitious_play(iterations=1000))
    row_play_counts, col_play_counts = play_counts[-1]
    dis_A = row_play_counts / np.sum(row_play_counts)
    dis_B = col_play_counts / np.sum(col_play_counts)
    return dis_A, dis_B
    
def run_psro():
    psro_pop = psro(2)
    for i in range(args.psro_iters):

        # Define the weighting towards diversity as a function of the fixed population size
        payoff = psro_pop.get_metagame()# calculate previous strategies' pay off matrix
        print(payoff)
        meta_nash_maml, meta_nash_par = metanash(payoff)
        new_maml_agent = train_maml_agent(psro_pop, meta_nash_par.tolist())
        new_par_agent = train_par_agent(psro_pop, meta_nash_maml.tolist())
        psro_pop.update(new_maml_agent, new_par_agent)

    #new_payoffs = torch_pop.get_metagame(payoffs, torch_pop.pop_size)
    #new_payoffs = new_payoffs[None,None,].to(device)
    #meta_nash = model(new_payoffs)[0]# calculate nash equillibrium based on meta solver

    #exp_loss = torch_pop.get_exploitability(meta_nash, payoffs, train_steps=args.exploit_train_step, lr=args.exploit_lr)

    #loss = exp_loss# ** 2
    #print('Train iteraion '+ str(iter_counter) + ': ' + str(exp_loss_all))
    ''' 
    if iter_counter % log_interval == 0:
        exp_loss_mean = eval_meta(model, train_iters=train_iters, batch_size=args.eval_bs, iters=iters, lr=inner_lr)
        model.train()
        writer.add_scalar('exploitability_mean', exp_loss_mean, iter_counter)
        #writer.add_scalar('exploitability_std', exp_loss_std, iter_counter)
        print('Test iteraion '+str(iter_counter) + ': ' + str(exp_loss_mean))

    if iter_counter % save_interval == 0:
        model_save = os.path.join(data_dir, 'model')
        if not os.path.exists(model_save):
            os.makedirs(model_save)
        torch.save(model.state_dict(), os.path.join(model_save, str(iter_counter) + '.pth'))
    '''

args = parse_args_psro()
run_psro()