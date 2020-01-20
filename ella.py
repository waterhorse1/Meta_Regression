import copy
import os
import time

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import torch.optim as optim
from gumbel_sample import gumbel_softmax
import utils
from data.task_multi import multi
from model import simple_MLP
from cavia_model_back import place,pool_encoder
from logger import Logger

def get_l(height, width):
    return torch.nn.init.normal_(torch.randn((height,width), requires_grad = True), mean = 0, std = 0.1)
def get_s(width):
    return torch.nn.init.normal_(torch.randn((width), requires_grad = True), mean = 0, std = 0.1)
    
def run(args, log_interval=5000, rerun=False):
    assert not args.maml

    # see if we already ran this experiment
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/{}_result_files/'.format(code_root, args.task))
    path = '{}/{}_result_files/'.format(code_root, args.task) + utils.get_path_from_args(args)

    if os.path.exists(path + '.pkl') and not rerun:
        return utils.load_obj(path)

    start_time = time.time()
    utils.set_seed(args.seed)

    # --- initialise everything ---

    # get the task family
    task_family_train = multi()
    task_family_valid = multi()
    task_family_test = multi()
    
    encoder = pool_encoder().to(args.device)
    encoder_optimiser = optim.Adam(encoder.parameters(), lr=1e-3)
    #encoder.load_state_dict(torch.load('./model/encoder'))
    p_encoder = place().to(args.device)
    p_optimiser = optim.Adam(p_encoder.parameters(), lr=1e-3)
    
    L = get_l(5251,4)
    
    # initialise network
    model = simple_MLP().to(args.device)

    # intitialise meta-optimiser
    # (only on shared params - context parameters are *not* registered parameters of the model)
    L_optimizer = optim.Adam([L], args.lr_L)

    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    # --- main training loop ---

    for i_iter in range(args.n_iter):
        #meta_gradient = [0 for _ in range(len(L))]
        #num = [0 for _ in range(len(L))]
        place_gradient = [0 for _ in range(len(p_encoder.state_dict()))]
        encoder_gradient = [0 for _ in range(len(encoder.state_dict()))]
        # sample tasks
        target_functions,ty = task_family_train.sample_tasks(args.tasks_per_metaupdate,True)

        # --- inner loop ---
        meta_gradient = 0
        for t in range(args.tasks_per_metaupdate):

            # get data for current task
            x = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
            
            '''
            y = target_functions[t](x)
            train_inputs = torch.cat([x,y],dim=1)
            a = encoder(train_inputs)
            #embedding,_ = torch.max(a,dim=0)
            embedding = torch.mean(a,dim=0)
            
            logits = p_encoder(embedding)
            logits = logits.reshape([2])
            
            #y = gumbel_softmax(logits, 0.5, hard=True)
            #s = y[0]
            y = F.softmax(logits,dim=-1)
            
            shape = y.size()#10*10*10
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            # Set gradients w.r.t. y_hard gradients w.r.t. y
            y_hard = (y_hard - y).detach() + y
            '''
            s = torch.zeros([4],requires_grad=False)
            s[ty[t]] = 1
            #s = y
           
            if t==1 and i_iter % 50 == 0:
                print(s)
                print(ty[t])
            
            #initialise st
            #s = get_s(args.n)
            
            new_params = L @ s

            #new_params = 0
            #for i in range(len(L)):
            #    new_params += s[i] * L[i]
            
            for _ in range(args.num_inner_updates):
                # forward through model
                train_outputs = model(x, new_params)

                # get targets
                train_targets = target_functions[t](x)
                # ------------ update on current task ------------
                #print(new_params)
                
                #print(torch.autograd.grad(torch.mean(train_outputs), L, create_graph=not args.first_order)[0])
                # compute loss for current task
                task_loss = F.mse_loss(train_outputs, train_targets)

                # compute gradient wrt context params
                task_gradients = \
                    torch.autograd.grad(task_loss, new_params, create_graph=not args.first_order)[0]

                # update context params (this will set up the computation graph correctly)
                
                new_params = new_params - args.lr_inner * task_gradients
                #print(torch.autograd.grad(task_loss, L, create_graph=not args.first_order)[0])
            

            # ------------ compute meta-gradient on test loss of current task ------------

            # get test data
            test_inputs = task_family_train.sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)

            # get outputs after update
            test_outputs = model(test_inputs, new_params)

            # get the correct targets
            test_targets = target_functions[t](test_inputs)

            # compute loss after updating context (will backprop through inner loop)
            loss_meta = F.mse_loss(test_outputs, test_targets)

            # compute gradient + save for current task
            #loss_meta += - torch.sum(s * torch.log(s + 1e-6)) /4
            
            task_grad = torch.autograd.grad(loss_meta, L,retain_graph=True)[0]

            #for i in range(len(task_grad)):
                # clip the gradient
             #   meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)
            meta_gradient += task_grad.clamp_(-10, 10)
            '''
            task_grad_place = torch.autograd.grad(loss_meta, p_encoder.parameters(),retain_graph=True)
            
            for i in range(len(task_grad_place)):
                # clip the gradient
                place_gradient[i] += task_grad_place[i].detach().clamp_(-10, 10)
            
            task_grad_encoder = torch.autograd.grad(loss_meta, encoder.parameters())
            for i in range(len(task_grad_encoder)):
                # clip the gradient
                encoder_gradient[i] += task_grad_encoder[i].detach().clamp_(-10, 10)
            '''
        # ------------ meta update ------------
        
        # assign meta-gradient

        L_optimizer.zero_grad()
        L.grad = meta_gradient / args.tasks_per_metaupdate

        # do update step on shared model
        L_optimizer.step()
        #L_optimizer.zero_grad()
        '''
        for i, param in enumerate(p_encoder.parameters()):
            param.grad = place_gradient[i] / args.tasks_per_metaupdate
        p_optimiser.step()
        
        for i, param in enumerate(encoder.parameters()):
            param.grad = encoder_gradient[i] / args.tasks_per_metaupdate
        encoder_optimiser.step()
        '''
        # ------------ logging ------------

        if i_iter % log_interval == 0:

            # evaluate on training set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), L, task_family=task_family_train,
                                              num_updates=args.num_inner_updates,encoder=encoder,p_encoder=p_encoder)
            logger.train_loss.append(loss_mean)
            logger.train_conf.append(loss_conf)

            # evaluate on test set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), L, task_family=task_family_valid,
                                              num_updates=args.num_inner_updates,encoder=encoder,p_encoder=p_encoder)
            logger.valid_loss.append(loss_mean)
            logger.valid_conf.append(loss_conf)

            # evaluate on validation set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), L, task_family=task_family_test,
                                              num_updates=args.num_inner_updates,encoder=encoder,p_encoder=p_encoder)
            logger.test_loss.append(loss_mean)
            logger.test_conf.append(loss_conf)

            # save logging results
            utils.save_obj(logger, path)

            # save best model
            if logger.valid_loss[-1] == np.min(logger.valid_loss):
                print('saving best model at iter', i_iter)
                logger.best_valid_model = copy.deepcopy(model)
                logger.best_encoder_valid_model = copy.deepcopy(encoder)
                logger.best_place_valid_model = copy.deepcopy(p_encoder)
                
            if i_iter % (3000) == 0:
                print('saving model at iter', i_iter)
                logger.valid_model.append(copy.deepcopy(model))
                logger.encoder_valid_model.append(copy.deepcopy(encoder))
                logger.place_valid_model.append(copy.deepcopy(p_encoder))

            # print current results
            logger.print_info(i_iter, start_time)
            start_time = time.time()

    return L


def eval_cavia(args, model, L, task_family, num_updates, n_tasks=100, return_gradnorm=False,\
               encoder=False,p_encoder=False):
    # get the task family
    input_range = task_family.get_input_range().to(args.device)

    # logging
    losses = []
    gradnorms = []

    # --- inner loop ---

    for t in range(n_tasks):

        # sample a task
        target_function,ty = task_family.sample_task(True)


        # get data for current task
        curr_inputs = task_family.sample_inputs(args.k_shot_eval).to(args.device)
        curr_targets = target_function(curr_inputs)
        '''
        train_inputs = torch.cat([curr_inputs,curr_targets],dim=1)
        a = encoder(train_inputs)
        #embedding,_ = torch.max(a,dim=0)
        embedding = torch.mean(a,dim=0)

        logits = p_encoder(embedding)
        logits = logits.reshape([2])

        #y = gumbel_softmax(logits, 0.5, hard=True)
        #s = y[0]
        y = F.softmax(logits,dim=-1)
        
        shape = y.size()#10*10*10
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        #s = y_hard
        '''
       # s = torch.tensor([1.])
        #s = y
        s = torch.zeros([4],requires_grad=False)
        s[ty] = 1
        # ------------ update on current task ------------
        
        #s_optimizer = optim.Adam([s], args.lr_s)

        
        new_params = L @ s
        
        for _ in range(1, num_updates + 1):

            # forward pass
            curr_outputs = model(curr_inputs, new_params)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # compute gradient wrt context params
            task_gradients = \
            torch.autograd.grad(task_loss, new_params, create_graph=not args.first_order)[0]

                # update context params (this will set up the computation graph correctly)
            new_params = new_params - args.lr_inner * task_gradients

            # update context params

            # keep track of gradient norms
            gradnorms.append(task_gradients[0].norm().item())

                #task_loss.backward(retain_graph=True)
                #s_optimizer.step()


        # ------------ logging ------------
        
        # compute true loss on entire input range
        with torch.no_grad():
            model.eval()
            losses.append(F.mse_loss(model(input_range, new_params),\
                                     target_function(input_range)).detach().item())
            model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)
