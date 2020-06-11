from tqdm import tqdm
from utils.utils import AverageMeter, channelwise_clamp

import torch
import torch.nn as nn
import torch.nn.functional as F


def standard_epoch_wrapper(model, trainloader, criterion, optimizer, device,
        epsilon, distrib_params, minibatch_replays, QTRADES_lamda=0.0):
    if minibatch_replays > 0: # Implicitly adversarial training
        model = run_standard_adv_epoch(
            model, trainloader, criterion, optimizer, device, 
            epsilon=epsilon, distrib_params=distrib_params,
            minibatch_replays=minibatch_replays
        )
    else:
        model, _ = run_standard_epoch(model, trainloader, criterion, 
            optimizer, device, QTRADES_lamda, distrib_params)

    return model


def run_standard_epoch(model, trainloader, criterion, optimizer, device,
                        QTRADES_lamda=0.0, distrib_params=None):       
    train_loss = AverageMeter()
    model.train()
    for img, label in tqdm(trainloader):
        img, label = img.to(device), label.to(device)
        logits, _ = model(img)
        loss = criterion(logits, label)
        if QTRADES_lamda > 0.0:
            #Generate adversary
            pert_img = generate_consistency(model, criterion, img, distrib_params)
            # make an output step
            cons_logits, _ = model(pert_img) 
            # Add to the loss
            loss = loss + QTRADES_lamda*F.kl_div(F.softmax(cons_logits, dim=-1), F.softmax(logits, dim=-1))

        #Logging Data:
        train_loss.update(loss.item(), img.size(0))
        #SGD Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, train_loss.avg


def run_standard_adv_epoch(model, trainloader, criterion, optimizer, 
        device, epsilon, distrib_params, minibatch_replays):
    # # # # # Start inner functions
    '''
    This function is for 
    ADVERSARIAL TRAINING WITH NO NOTION OF MAGNET LOSS
    '''
    def train_free_batch(model, img, label, criterion, optimizer,
            delta, minibatch_replays, epsilon, llimit, ulimit):
        for _ in range(minibatch_replays): # m is the number of inner restarts
            delta.requires_grad = True
            logits, _ = model(img + delta)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            grad = delta.grad.detach()
            # do gradient ascent step
            delta.data = delta.data + epsilon*grad.sign()
            # project to epsilon energy
            delta.data = channelwise_clamp(delta.data, minima=-epsilon, 
                maxima=epsilon)
            # project to valid noise for each image
            delta.data = channelwise_clamp(delta.data, minima=llimit-img, 
                maxima=ulimit-img)
            optimizer.step()
            delta.grad.zero_()
            
        return model, delta
    # # # # # Finish inner functions

    model.train()
    # Make right sizes
    epsilon_p = epsilon.unsqueeze(1) / distrib_params['std'].unsqueeze(0)
    epsilon_p = epsilon_p.unsqueeze(2).unsqueeze(3).to(device)
    llimit_p, ulimit_p = distrib_params['minima'], distrib_params['maxima']
    llimit_p = llimit_p.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    ulimit_p = ulimit_p.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    # Initialize delta
    delta = None
    for img, label in tqdm(trainloader):
        img, label = img.to(device), label.to(device)
        if delta is None: # for first batch
            delta = torch.zeros_like(img)
        else: # For last batch, since it can have diff dims
            delta = delta.data[:img.size(0)]
        # free training
        model, delta = train_free_batch(
            model, img, label, criterion, optimizer,
            delta=delta, minibatch_replays=minibatch_replays, 
            epsilon=epsilon_p, llimit=llimit_p, ulimit=ulimit_p
        )

    return model

def generate_consistency(model, criterion, img, distrib_params):
    device = img.device

    std, minima, maxima = distrib_params['std'].to(device), distrib_params['minima'].to(device), distrib_params['maxima'].to(device)
    alpha, eps = torch.tensor([10/255], device=device).unsqueeze(1)/std.unsqueeze(0), torch.tensor([8/255],device=device).unsqueeze(1)/std.unsqueeze(0)
    minima, maxima = minima.unsqueeze(0).unsqueeze(2).unsqueeze(3), maxima.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    alpha = alpha.unsqueeze(2).unsqueeze(3).to(device)
    eps = eps.unsqueeze(2).unsqueeze(3).to(device)
   # print(img.shape, eps.shape, alpha.shape, minima.shape, maxima.shape)
     
    original_logits, _ = model(img)

    pert = 2*eps*torch.rand_like(img)-eps
    pert_img = channelwise_clamp(img+pert, 
            minima=minima, maxima=maxima).data.clone()

    pert_img.requires_grad = True
    # Compute forward
    model.zero_grad()
    # Compute label predictions
    logits, _ = model(pert_img)
    loss = F.kl_div(F.softmax(logits, dim=-1), F.softmax(original_logits, dim=-1))
    loss.backward()
    with torch.no_grad():
        pert_img += alpha*pert_img.grad.sign()
        delta = channelwise_clamp(pert_img-img, minima=-eps, maxima=eps)
        pert_img = channelwise_clamp(img + delta, minima=minima, maxima=maxima)
    return pert_img
