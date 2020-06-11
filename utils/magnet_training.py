from tqdm import tqdm
from utils.utils import (AverageMeter, compute_reps, channelwise_clamp, 
    get_softmax_probs)

import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_ent(target_distrib, pred_scores):
    '''
    Compute cross-entropy between two probability distributions, not just
    between a target distribution with one class with probability 1
    See: https://en.wikipedia.org/wiki/Cross_entropy
    The second distribution should be given in terms of scores, so as to
    compute log(softmax(x))
    '''
    # both parameters are of size [batch_size, num_classes]
    # sum in the classes dimension
    log_softmax = F.log_softmax(pred_scores, dim=1)
    cross_entropies = -1.0 * (target_distrib * log_softmax).sum(dim=1)
    # cross_entropies is of size [batch_size]
    return cross_entropies.mean()


def magnet_epoch_wrapper(model, optimizer, trainloader, device, trainset, 
        train_labels, batch_builder, print_freq, cluster_refresh_interval, 
        criterion, eps, magnet_data, distrib_params, minibatch_replays, 
        actual_trades):
    if minibatch_replays > 0: # Implicitly adversarial training
        model, batch_builder, _ = run_magnet_adv_epoch(
            model, optimizer, trainloader, device, trainset, train_labels, 
            batch_builder, print_freq, cluster_refresh_interval, criterion, 
            eps=eps,magnet_data=magnet_data, distrib_params=distrib_params, 
            minibatch_replays=minibatch_replays, 
        )

    else:
        model, batch_builder, _ = run_magnet_epoch(
            model, optimizer, trainloader, device, trainset, train_labels, 
            batch_builder, print_freq, cluster_refresh_interval, criterion, 
            eps=eps, magnet_data=magnet_data, distrib_params=distrib_params, 
            actual_trades=actual_trades
        )

    return model, batch_builder


def run_magnet_epoch(model, optimizer, trainloader, device, trainset, 
        train_labels, batch_builder, print_freq, cluster_refresh_interval, 
        criterion, eps, magnet_data, distrib_params, alpha=10/255, 
        actual_trades=False):

    consist_crit = nn.MSELoss() if magnet_data['mse_consistency'] else cross_ent
    model.train()
    losses = AverageMeter()
    # The next for-loop is actually an epoch
    pbar = tqdm(range(len(trainloader)))
    for iteration in pbar:
        # Get new batch indices and labels
        batch_inds, class_inds, clust_assigns = batch_builder.gen_batch()
        trainloader.sampler.batch_indices = batch_inds
        # The next for-loop (actually) only runs once
        for img, target in trainloader:
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            # # # # # # # # # # # # Forward
            _, embs = model(img)
            # # # # # # # # # # # # Compute loss
            loss, inst_losses, _ = criterion(embs, class_inds, clust_assigns)
            # # # # # # # # # # # # Consistency regularization
            if magnet_data['consistency_lambda'] > 0.0:
                # Get original predictions for class
                probs = get_softmax_probs(embeddings=embs, 
                    magnet_data=magnet_data).to(device)
                _, orig_preds = torch.max(probs.data, 1)
                # Get adversarial instances wrt magnet loss
                eps_p = eps / distrib_params['std']
                alpha_p = alpha / distrib_params['std'].unsqueeze(0)
                # Pass the original embeddings as the 'labels' param
                # or pass the original probs as the 'labels' param
                p_labels = embs if magnet_data['mse_consistency'] else probs
                adv_instances = attack_instances_consistency(
                    model, images=img, labels=p_labels.detach(), 
                    orig_preds=orig_preds, eps=eps_p, 
                    distrib_params=distrib_params, alpha=alpha_p, 
                    criterion=consist_crit, magnet_data=magnet_data, 
                    iterations=1, rand=True, kl_loss = actual_trades
                )
                # Compute representation of adversarial instances
                _, adv_embs = model(adv_instances)
                # Compute the consistency term
                if magnet_data['mse_consistency']:
                    # Loss for consistency between adv and original instances
                    consistency_loss = consist_crit(adv_embs, embs)
                else:
                    # Get predictions for class of adversarial instances
                    adv_scores = get_softmax_probs(embeddings=adv_embs, 
                        magnet_data=magnet_data, return_scores=True)
                    # Loss for consistency between adv instances and clusters 
                    # the original instances were assigned to
                    consistency_loss = consist_crit(probs, adv_scores)

                # Add consistency loss to total loss
                loss = loss + magnet_data['consistency_lambda']*consistency_loss

            # Include cross entropy in the loss
            if magnet_data['xent_lambda'] > 0.0:
                xent_crit = nn.CrossEntropyLoss()
                class_scores = get_softmax_probs(embeddings=embs, 
                    magnet_data=magnet_data, return_scores=True).to(device)
                xent_loss = xent_crit(class_scores, target)
                # Add cross entropy loss to total loss
                loss = loss + magnet_data['xent_lambda']*xent_loss
            # Backward and step
            loss.backward()
            optimizer.step()
            # Update losses in the batch builder
            with torch.no_grad():
                batch_builder.update_losses(batch_inds, inst_losses)
            # Update values
            losses.update(loss.item(), img.size(0))
            # Print
            if iteration % print_freq == 0:
                pbar.set_description(f'Loss: {losses.avg:4.3f}')
            # Refresh clusters
            if iteration % cluster_refresh_interval == 0 and iteration != 0:
                model.eval()
                _, reps = compute_reps(model, trainset, 400)
                batch_builder.update_clusters(reps)
                model.train()

    return model, batch_builder, losses


def run_magnet_adv_epoch(model, optimizer, trainloader, device, trainset, 
        train_labels, batch_builder, print_freq, cluster_refresh_interval, 
        criterion, eps, magnet_data, distrib_params, minibatch_replays, 
        adv_reps=False):
    # # # # # # # # Start inner functions
    def train_free_magnet_batch(model, img, label, classes, 
            assigned_clusters, criterion, optimizer, delta, minibatch_replays, 
            eps, llimit, ulimit, magnet_data):
        '''
        This function is for 
        ADVERSARIAL TRAINING, WHERE:
        BOTH THE NETWORK AND DELTA ARE UPDATED wrt MAGNET LOSS
        '''
        assert magnet_data is None, 'This param only exists for compatibility' \
            ' with the "train_free_magnet_probs_batch" fun (same signature)'
        delta.requires_grad = True
        for _ in range(minibatch_replays): # m is the number of inner restarts
            _, embs = model(img + delta)
            loss, losses, _ = criterion(embs, classes, assigned_clusters)
            # Backward and step
            optimizer.zero_grad()
            loss.backward()
            grad = delta.grad.detach()
            # do gradient ascent step
            delta.data = delta.data + eps*grad.sign()
            # project to eps energy
            delta.data = channelwise_clamp(delta.data, minima=-eps, 
                maxima=eps)
            # project to valid noise for each image
            delta.data = channelwise_clamp(delta.data, minima=llimit-img, 
                maxima=ulimit-img)
            optimizer.step()
            delta.grad.zero_()
        
        # return the last computed batch_example_losses and variance
        return model, delta, losses
    
    # # # # # # # # Finish inner functions
    # free training attacking the magnet loss
    batch_fun = train_free_magnet_batch
    magnet_data_to_use = None

    model.train()
    losses = AverageMeter()
    # Make right sizes
    eps_p = eps.unsqueeze(1) / distrib_params['std'].unsqueeze(0)
    eps_p = eps_p.unsqueeze(2).unsqueeze(3).to(device)
    # The minima
    llimit_p = distrib_params['minima']
    llimit_p = llimit_p.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    # The maxima
    ulimit_p = distrib_params['maxima']
    ulimit_p = ulimit_p.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    # Initialize delta
    delta = None
    # The next for-loop is actually an epoch
    pbar = tqdm(range(len(trainloader)))
    for iteration in pbar:
        # Get new batch indices and labels
        batch_inds, class_inds, clust_assgn = batch_builder.gen_batch()
        trainloader.sampler.batch_indices = batch_inds
        # The next for-loop (actually) only runs once
        for img, label in trainloader:
            img, label = img.to(device), label.to(device)
            # For last batch, since it can have diff dims
            if delta is None:
                delta = torch.zeros_like(img)
            else: 
                delta = delta.data[:img.size(0)]
            # Train batch
            model, delta, _ = batch_fun(
                model, img, label, class_inds, clust_assgn, criterion, 
                optimizer, delta=delta, minibatch_replays=minibatch_replays, 
                eps=eps_p, llimit=llimit_p, ulimit=ulimit_p,
                magnet_data=magnet_data_to_use
            )
            # Update losses in the batch builder
            with torch.no_grad():
                # compute losses on adversarial examples
                _, embs = model(img + delta)
                loss, batch_losses, _ = criterion(embs, class_inds, clust_assgn)
                batch_builder.update_losses(batch_inds, batch_losses)
            # Update values
            losses.update(loss.item(), img.size(0))
            # Print
            if iteration % print_freq == 0:
                pbar.set_description(f'Loss: {losses.avg:4.3f}')
            # Refresh clusters
            if iteration % cluster_refresh_interval == 0 and iteration != 0:
                model.eval()
                _, reps = compute_reps(model, trainset, 400)
                batch_builder.update_clusters(reps)
                model.train()

    return model, batch_builder, losses


def attack_instances_consistency(model, images, labels, orig_preds, eps, alpha,
        distrib_params, criterion, iterations=50, rand=False, magnet_data=None, 
        kl_loss=False):
    # Inner function
    def get_rand_perturb(images, eps):
        # Between 0 and 1
        pert = torch.rand_like(images)
        # Now between -eps and +eps
        pert = 2*eps*pert - eps
        return pert
    
    if kl_loss: #This is the implementation of the Trades paper
        iterations = 10
        alpha[alpha>0] = 0.03
        criterion = torch.nn.KLDivLoss(size_average=False)
    
    assert magnet_data is not None
    # Unsqueeze so that we don't need an extra for-loop
    # (and move it to appropriate device)
    device = images.device
    alphas = alpha.unsqueeze(2).unsqueeze(3).to(device)
    epss = eps.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    # The extrema
    minima, maxima = distrib_params['minima'], distrib_params['maxima']
    minima = minima.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    maxima = maxima.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    # Assume the image is a tensor
    if rand:
        perturbed_images = images.data + get_rand_perturb(images, epss)
        perturbed_images = channelwise_clamp(perturbed_images, 
            minima=minima, maxima=maxima).data.clone()
    else:
        perturbed_images = images.data.clone()

    # Check for which images was the attack already succesful
    for _ in range(iterations): 
        # Gradient for the image
        perturbed_images.requires_grad = True
        # Compute forward
        model.zero_grad()
        # Compute label predictions
        _, embeddings = model(perturbed_images)
        scores = get_softmax_probs(embeddings=embeddings, 
            magnet_data=magnet_data, return_scores=True).to(device)
        # Compute loss
        if kl_loss:
            cost = criterion(F.log_softmax(scores,dim=1), labels)
        else:
            if isinstance(criterion, nn.MSELoss):
                # Compute w.r.t embedding
                cost = criterion(labels, embeddings)
            else:
                # Compute w.r.t probabilities
                cost = criterion(labels, scores)
        # Backward
        cost.backward()
        with torch.no_grad():
            # Sign of gradient times 'learning rate'
            eta = perturbed_images.grad.sign()
            perturbed_images += alphas*eta
            # Project to noise within epsilon ball around original images
            noise = perturbed_images - images
            noise = channelwise_clamp(noise, minima=-epss, maxima=epss)
            # Project to images within space of possible images
            perturbed_images = channelwise_clamp(images + noise, 
                minima=minima, maxima=maxima
            )

    return perturbed_images
