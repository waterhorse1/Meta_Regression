"""
Regression experiment using CAVIA
"""
import copy
import os
import time

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
import tasks_sine, tasks_celebA
from data.task_multi import multi
from cavia_model import CaviaModel, pool_encoder
from logger import Logger


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
    if args.task == 'sine':
        task_family_train = tasks_sine.RegressionTasksSinusoidal()
        task_family_valid = tasks_sine.RegressionTasksSinusoidal()
        task_family_test = tasks_sine.RegressionTasksSinusoidal()
    elif args.task == 'celeba':
        task_family_train = tasks_celebA.CelebADataset('train', device=args.device)
        task_family_valid = tasks_celebA.CelebADataset('valid', device=args.device)
        task_family_test = tasks_celebA.CelebADataset('test', device=args.device)
    elif args.task == 'multi':
        task_family_train = multi()
        task_family_valid = multi()
        task_family_test = multi()
    else:
        raise NotImplementedError

    # initialise network
    model = CaviaModel(n_in=task_family_train.num_inputs,
                       n_out=task_family_train.num_outputs,
                       num_context_params=args.num_context_params,
                       n_hidden=args.num_hidden_layers,
                       device=args.device
                       ).to(args.device)

    # intitialise meta-optimiser
    # (only on shared params - context parameters are *not* registered parameters of the model)
    #meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)
    #l_optimizer = [optim.SGD(model.fc_layers[0].parameters(), args.lr_inner),\
    #               optim.SGD(model.fc_layers[1].parameters(), args.lr_inner),\
    #               optim.SGD(model.fc_layers[2].parameters(), args.lr_inner)]
    #encoder = pool_encoder()
    #encoder.load_state_dict(torch.load('./model/encoder'))
    # initialise loggers
    encoder = pool_encoder()
    encoder.load_state_dict(torch.load('./model/encoder'))
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    # --- main training loop ---
    
    meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)
    
    for i_iter in range(args.n_iter):

        # initialise meta-gradient
        meta_gradient = [0 for _ in range(len(model.state_dict()))]

        # sample tasks
        target_functions,ty = task_family_train.sample_tasks(args.tasks_per_metaupdate,return_type=True)

        # --- inner loop ---
        
        for t in range(args.tasks_per_metaupdate):

            # reset private network weights
            model.reset_context_params()
            copy_model = copy.deepcopy(model)
            x = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)

            y = target_functions[t](x)
            train_inputs = torch.cat([x,y],dim=1)
            a = encoder(train_inputs)
            #embedding,_ = torch.max(a,dim=0)
            embedding = torch.mean(a,dim=0)
            copy_model.set_context_params(embedding, ty[t])
            #l_optimizer = [optim.SGD([copy_model.param_1], args.lr_inner),\
            #              optim.SGD([copy_model.param_2], args.lr_inner)]
                           #optim.SGD([params], args.lr_inner) for params in copy_model.context_params]
            # get data for current task
            l_optimizer = [optim.SGD([param], args.lr_inner) for param in copy_model.context_params]
            #train_inputs = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
            
            for _ in range(args.num_inner_updates):
                # forward through model
                train_outputs = copy_model(x)

                # get targets
                train_targets = target_functions[t](x)
                type = ty[t]#int(0)
                

                # ------------ update on current task ------------

                # compute loss for current task
                task_loss = F.mse_loss(train_outputs, train_targets)
                #exit()
                task_loss.backward()
                for i in range(len(l_optimizer)):
                    if i == type:
                        l_optimizer[i].step()
                    else:
                        l_optimizer[i].zero_grad()


                # compute gradient wrt context params
                #task_gradients = \
                #    torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]

                # update context params (this will set up the computation graph correctly)
                #model.context_params = model.context_params - args.lr_inner * task_gradients
            
            # ------------ compute meta-gradient on test loss of current task ------------

            # get test data
            test_inputs = task_family_train.sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)          
            # get outputs after update
            test_outputs = copy_model(test_inputs)

            # get the correct targets
            test_targets = target_functions[t](test_inputs)

            # compute loss after updating context (will backprop through inner loop)
            loss_meta = F.mse_loss(test_outputs, test_targets)

            # compute gradient + save for current task
            task_grad = torch.autograd.grad(loss_meta, copy_model.parameters())

            for i in range(len(task_grad)):
                # clip the gradient
                    meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)

        # ------------ meta update ------------

        # assign meta-gradient
        for i, param in enumerate(model.parameters()):
                param.grad = meta_gradient[i] / args.tasks_per_metaupdate
  
        # do update step on shared model
        meta_optimiser.step()

        # reset context params
        #model.reset_context_params()

        # ------------ logging ------------

        if i_iter % log_interval == 0:

            # evaluate on training set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_train,
                                              num_updates=args.num_inner_updates,encoder=encoder)
            logger.train_loss.append(loss_mean)
            logger.train_conf.append(loss_conf)

            # evaluate on test set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_valid,
                                              num_updates=args.num_inner_updates,encoder=encoder)
            logger.valid_loss.append(loss_mean)
            logger.valid_conf.append(loss_conf)

            # evaluate on validation set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_test,
                                              num_updates=args.num_inner_updates,encoder=encoder)
            logger.test_loss.append(loss_mean)
            logger.test_conf.append(loss_conf)

            # save logging results
            utils.save_obj(logger, path)

            # save best model
            if logger.valid_loss[-1] == np.min(logger.valid_loss):
                print('saving best model at iter', i_iter)
                logger.best_valid_model = copy.deepcopy(model)

            # visualise results
            if args.task == 'celeba':
                task_family_train.visualise(task_family_train, task_family_test, copy.deepcopy(logger.best_valid_model),
                                            args, i_iter)

            # print current results
            logger.print_info(i_iter, start_time)
            start_time = time.time()

    return logger


def eval_cavia(args, model, task_family, num_updates, n_tasks=100, return_gradnorm=False, encoder = False):
    # get the task family
    input_range = task_family.get_input_range().to(args.device)

    # logging
    losses = []
    gradnorms = []

    # --- inner loop ---

    for t in range(n_tasks):

        # sample a task
        target_function, ty = task_family.sample_task(return_type = True)
        #ty = 0

        # reset context parameters
        model.reset_context_params()
        copy_model = copy.deepcopy(model)
        # get data for current task
        curr_inputs = task_family.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)
        curr_targets = target_function(curr_inputs)
        
        train_inputs = torch.cat([curr_inputs,curr_targets],dim=1)
        a = encoder(train_inputs)
        #embedding,_ = torch.max(a,dim=0)
        embedding = torch.mean(a,dim=0)
        copy_model.set_context_params(embedding, ty)
        # ------------ update on current task ------------
        #copy_model = copy.deepcopy(model)
        #l_optimizer = [optim.SGD([copy_model.param_1], args.lr_inner),\
        #                  optim.SGD([copy_model.param_2], args.lr_inner)]
        l_optimizer = [optim.SGD([params], args.lr_inner) for params in copy_model.context_params]
        
        for _ in range(1, num_updates + 1):

            # forward pass
            curr_outputs = copy_model(curr_inputs)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # compute gradient wrt context params
            #task_gradients = \
            #    torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]
            task_loss.backward()
            
            for i in range(len(l_optimizer)):
                if i == ty:
                    l_optimizer[i].step()
                    #print(i)
                else:
                    l_optimizer[i].zero_grad()
                    
            # update context params
            '''
            if args.first_order:
                model.context_params = model.context_params - args.lr_inner * task_gradients.detach()
            else:
                model.context_params = model.context_params - args.lr_inner * task_gradients
            '''
            # keep track of gradient norms
            #gradnorms.append(task_gradients[0].norm().item())
        
        # ------------ logging ------------

        # compute true loss on entire input range
        copy_model.eval()
        #print(copy_model.context_params)
        losses.append(F.mse_loss(copy_model(input_range), target_function(input_range)).detach().item())
        copy_model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)
