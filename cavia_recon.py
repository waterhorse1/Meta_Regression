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
from cavia_model_back import CaviaModel, pool_encoder, pool_decoder, place
from logger import Logger
from gumbel_sample import gumbel_softmax

latent_dim = 15
categorical_dim = 2
temp = 1
ANNEAL_RATE = 0.00001

def run(args, log_interval=5000, rerun=False):
    global temp
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
    meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)
    encoder = pool_encoder().to(args.device)
    encoder_optimiser = optim.Adam(encoder.parameters(), lr=1e-3)
    
    decoder = pool_decoder().to(args.device)
    decoder_optimiser = optim.Adam(decoder.parameters(), lr=1e-3)
    #encoder.load_state_dict(torch.load('./model/encoder'))
    p_encoder = place().to(args.device)
    p_optimiser = optim.Adam(p_encoder.parameters(), lr=1e-3)
    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    # --- main training loop ---

    for i_iter in range(args.n_iter):
        # initialise meta-gradient
        meta_gradient = [0 for _ in range(len(model.state_dict()))]
        place_gradient = [0 for _ in range(len(p_encoder.state_dict()))]
        encoder_gradient = [0 for _ in range(len(encoder.state_dict()))]
        decoder_gradient = [0 for _ in range(len(decoder.state_dict()))]
        

        # sample tasks
        target_functions,ty = task_family_train.sample_tasks(args.tasks_per_metaupdate,True)

        # --- inner loop ---

        for t in range(args.tasks_per_metaupdate):
            
            # reset private network weights
            model.reset_context_params()

            # get data for current task
            x = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)

            y = target_functions[t](x)
            train_inputs = torch.cat([x,y],dim=1)
            a, input_embed = encoder(train_inputs, True)
            recon = decoder(a)
            recon_loss = torch.nn.MSELoss()(recon, input_embed)
                       
            embedding = torch.mean(a,dim=0)
            
            logits = p_encoder(embedding)
            logits = logits.reshape([latent_dim, categorical_dim])

            y = gumbel_softmax(logits, temp, hard=True)
            y = y[:,1]
            if t==1 and i_iter % 50 == 0:
                print(y)
                print(ty[t])
            #print(temp)
            
            #model.set_context_params(embedding)
            #print(model.context_params)
            
            for _ in range(args.num_inner_updates):
                # forward through model
                train_outputs = model(x)

                # get targets
                train_targets = target_functions[t](x)

                # ------------ update on current task ------------

                # compute loss for current task
                task_loss = F.mse_loss(train_outputs, train_targets)

                # compute gradient wrt context params
                task_gradients = \
                    torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]

                # update context params (this will set up the computation graph correctly)
                model.context_params = model.context_params - args.lr_inner * task_gradients * y
            
            #print(model.context_params)
            # ------------ compute meta-gradient on test loss of current task ------------

            # get test data
            test_inputs = task_family_train.sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)

            # get outputs after update
            test_outputs = model(test_inputs)

            # get the correct targets
            test_targets = target_functions[t](test_inputs)

            # compute loss after updating context (will backprop through inner loop)
            loss_meta = F.mse_loss(test_outputs, test_targets)
            #print(torch.norm(y,1)/1000)
            #loss_meta += torch.norm(y,1)/700
            qy = F.softmax(logits, dim=-1)
            log_ratio = torch.log(qy * categorical_dim + 1e-20)
            KLD = torch.sum(qy * log_ratio, dim=-1).mean()
                # print(KLD)
            loss_meta = loss_meta + 0.2 * KLD + 0.2 * recon_loss

            # compute gradient + save for current task
            task_grad = torch.autograd.grad(loss_meta, model.parameters(),retain_graph=True)

            for i in range(len(task_grad)):
                # clip the gradient
                meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)
                
            task_grad_place = torch.autograd.grad(loss_meta, p_encoder.parameters(),retain_graph=True)
            
            for i in range(len(task_grad_place)):
                # clip the gradient
                place_gradient[i] += task_grad_place[i].detach().clamp_(-10, 10)
            
            task_grad_encoder = torch.autograd.grad(loss_meta, encoder.parameters(),retain_graph=True)
            for i in range(len(task_grad_encoder)):
                # clip the gradient
                encoder_gradient[i] += task_grad_encoder[i].detach().clamp_(-10, 10)
            
            task_grad_decoder = torch.autograd.grad(loss_meta, decoder.parameters())
            for i in range(len(task_grad_decoder)):
                # clip the gradient
                decoder_gradient[i] += task_grad_decoder[i].detach().clamp_(-10, 10)
        
        # ------------ meta update ------------

        # assign meta-gradient
        for i, param in enumerate(model.parameters()):
            param.grad = meta_gradient[i] / args.tasks_per_metaupdate
        meta_optimiser.step()
        
        # do update step on shared model
        for i, param in enumerate(p_encoder.parameters()):
            param.grad = place_gradient[i] / args.tasks_per_metaupdate
        p_optimiser.step()
        
        for i, param in enumerate(encoder.parameters()):
            param.grad = encoder_gradient[i] / args.tasks_per_metaupdate
        encoder_optimiser.step()
        
        for i, param in enumerate(decoder.parameters()):
            param.grad = decoder_gradient[i] / args.tasks_per_metaupdate
        decoder_optimiser.step()
        # reset context params
        model.reset_context_params()
        
        if i_iter % 350 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i_iter), 0.3)
            print(temp)
        # ------------ logging ------------

        if i_iter % log_interval == 0:

            # evaluate on training set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_train,
                                              num_updates=args.num_inner_updates, encoder=encoder, p_encoder = p_encoder)
            logger.train_loss.append(loss_mean)
            logger.train_conf.append(loss_conf)

            # evaluate on test set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_valid,
                                              num_updates=args.num_inner_updates, encoder=encoder, p_encoder = p_encoder)
            logger.valid_loss.append(loss_mean)
            logger.valid_conf.append(loss_conf)

            # evaluate on validation set
            
            if i_iter % log_interval == 0:
                loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_test,
                                                  num_updates=args.num_inner_updates, encoder=encoder, p_encoder = p_encoder)
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
                
            if i_iter % (4*log_interval) == 0:
                print('saving model at iter', i_iter)
                logger.valid_model.append(copy.deepcopy(model))
                logger.encoder_valid_model.append(copy.deepcopy(encoder))
                logger.place_valid_model.append(copy.deepcopy(p_encoder))

            # visualise results
            if args.task == 'celeba':
                task_family_train.visualise(task_family_train, task_family_test, copy.deepcopy(logger.best_valid_model),
                                            args, i_iter)

            # print current results
            logger.print_info(i_iter, start_time)
            start_time = time.time()

    return logger


def eval_cavia(args, model, task_family, num_updates, n_tasks=100, return_gradnorm=False, encoder = False, p_encoder = False):
    global temp
    # get the task family
    input_range = task_family.get_input_range().to(args.device)

    # logging
    losses = []
    gradnorms = []

    # --- inner loop ---

    for t in range(n_tasks):

        # sample a task
        target_function,ty = task_family.sample_task(True)

        # reset context parameters
        model.reset_context_params()

        # get data for current task
        curr_inputs = task_family.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)
        curr_targets = target_function(curr_inputs)
        
        train_inputs = torch.cat([curr_inputs,curr_targets],dim=1)
        a = encoder(train_inputs)
        #embedding,_ = torch.max(a,dim=0)
        embedding = torch.mean(a,dim=0)
        #model.set_context_params(embedding)
        logits = p_encoder(embedding)
        logits = logits.reshape([latent_dim, categorical_dim])
        y = gumbel_softmax(logits, temp, hard=True)
        y = y[:,1]

        # ------------ update on current task ------------
        
        for _ in range(1, num_updates + 1):

            # forward pass
            curr_outputs = model(curr_inputs)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # compute gradient wrt context params
            task_gradients = \
                torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]

            # update context params
            if args.first_order:
                model.context_params = model.context_params - args.lr_inner * task_gradients.detach() * y
            else:
                model.context_params = model.context_params - args.lr_inner * task_gradients * y

            # keep track of gradient norms
            gradnorms.append(task_gradients[0].norm().item())
        
        # ------------ logging ------------
        
        # compute true loss on entire input range
        model.eval()
        losses.append(F.mse_loss(model(input_range), target_function(input_range)).detach().item())
        model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)

def test(args, n_tasks=5000):
    # copy weights of network
    global temp
    temp = 0.5
    logger = utils.load_obj('./multi_result_files/22d766b4a3616fc920e376a9e6a5d5ee')
    model = logger.best_valid_model
    model.num_context_params = args.num_context_params
    encoder = logger.best_encoder_valid_model
    p_encoder = logger.best_place_valid_model
    utils.set_seed(args.seed*2)
    task_family_test = multi()
    loss_mean, loss_conf = eval_cavia(args, copy.copy(model), task_family=task_family_test,
                                        num_updates=5, n_tasks=n_tasks,encoder = encoder,p_encoder=p_encoder)
    print(loss_mean, loss_conf)