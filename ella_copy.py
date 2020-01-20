import copy
import os
import time

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
from data.task_multi import multi
from model import simple_MLP
from logger import Logger

def get_l(height, width):
    return torch.nn.init.normal_(torch.randn((height,width), requires_grad = True), mean = 0, std = 0.1)
def get_s(width):
    return torch.nn.init.normal_(torch.randn((width), requires_grad = True), mean = 0, std = 1/width)
    
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

    L = get_l(5251, 1)
    
    # initialise network
    model = simple_MLP().to(args.device)

    # intitialise meta-optimiser
    # (only on shared params - context parameters are *not* registered parameters of the model)
    L_optimiser = optim.Adam([L], 0.001)

    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    # --- main training loop ---

    for i_iter in range(args.n_iter):

        # sample tasks
        target_functions = task_family_train.sample_tasks(args.tasks_per_metaupdate)

        # --- inner loop ---
        meta_gradient = 0
        for t in range(args.tasks_per_metaupdate):

            # get data for current task
            train_inputs = task_family_train.sample_inputs(args.k_meta_train).to(args.device)
            
            #initialise st
            #s = get_s(args.n)
           # s_optimizer = optim.Adam([s], args.lr_s)
                      
            
            new_params = L[:,0].clone()

            for _ in range(args.num_inner_updates):
                # forward through model
                train_outputs = model(train_inputs, new_params)

                # get targets
                train_targets = target_functions[t](train_inputs)

                # ------------ update on current task ------------

                # compute loss for current task
                task_loss = F.mse_loss(train_outputs, train_targets)

                # compute gradient wrt context params
                task_gradients = \
                    torch.autograd.grad(task_loss, new_params, create_graph=not args.first_order)[0]

                # update context params (this will set up the computation graph correctly)
                new_params = new_params - args.lr_inner * task_gradients
                #print('l1',L.grad)
                # forward through model
                '''
                train_outputs = model(train_inputs, new_params)
                train_targets = target_functions[t](train_inputs)
                task_loss = F.mse_loss(train_outputs, train_targets)
                task_loss.backward()
                s_optimizer.step()
                L_optimizer.zero_grad()
                '''
                

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
            task_grad = torch.autograd.grad(loss_meta, L)[0]

            #for i in range(len(task_grad)):
                # clip the gradient
             #   meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)
            meta_gradient += task_grad.detach().clamp_(-10, 10)

        # ------------ meta update ------------

        # assign meta-gradient
        L.grad = meta_gradient / args.tasks_per_metaupdate

        # do update step on shared model
        
        L_optimiser.step()
        L.grad = None
        


        # ------------ logging ------------

        if i_iter % log_interval == 0:

            # evaluate on training set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), L, task_family=task_family_train,
                                              num_updates=args.num_inner_updates)
            logger.train_loss.append(loss_mean)
            logger.train_conf.append(loss_conf)

            # evaluate on test set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), L, task_family=task_family_valid,
                                              num_updates=args.num_inner_updates)
            logger.valid_loss.append(loss_mean)
            logger.valid_conf.append(loss_conf)

            # evaluate on validation set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), L, task_family=task_family_test,
                                              num_updates=args.num_inner_updates)
            logger.test_loss.append(loss_mean)
            logger.test_conf.append(loss_conf)

            # save logging results
            utils.save_obj(logger, path)

            # save best model
            if logger.valid_loss[-1] == np.min(logger.valid_loss):
                print('saving best model at iter', i_iter)
                logger.best_valid_model = copy.deepcopy(L)


            # print current results
            logger.print_info(i_iter, start_time)
            start_time = time.time()

    return L


def eval_cavia(args, model, L, task_family, num_updates, n_tasks=100, return_gradnorm=False):
    # get the task family
    input_range = task_family.get_input_range().to(args.device)

    # logging
    losses = []
    gradnorms = []

    # --- inner loop ---

    for t in range(n_tasks):

        # sample a task
        target_function = task_family.sample_task()


        # get data for current task
        curr_inputs = task_family.sample_inputs(args.k_shot_eval).to(args.device)
        curr_targets = target_function(curr_inputs)

        # ------------ update on current task ------------
        #s = get_s(args.n)
        #s_optimizer = optim.Adam([s], args.lr_s)
  
        new_params = L[:,0].clone()

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



        # ------------ logging ------------
        
        # compute true loss on entire input range
        #new_params = L @ s
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
