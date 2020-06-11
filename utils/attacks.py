import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import os.path as osp

from utils.utils import (copy_pretrained_model, eval_model, get_softmax_probs,
    channelwise_clamp)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100

import pdb

# Matrix of semantic distances in CIFAR100
SEM_DISTS = torch.from_numpy(np.load('utils/cifar100_dists.npy'))
CLASS_NAMES_FILE = 'utils/class_names_cifar100.txt'
DF_COLS = [
    'label', 'nat_pred', 'adv_pred', 'label_name', 'nat_name', \
    'adv_name', 'nat_h_error', 'adv_h_error', 'linf_norm', 'pgd_iters'
]

def eval_robustness(model, dataloader, dataset, data_labels, checkpoint, 
        distrib_params, device, passed_epsilons, best=True, hardcoded_path=None, 
        rand_restarts=10, iterations=50, save_df=True, alpha=2/255,
        magnet_data=None, check_acc=True, L=None):
    # # # # # # # # # # Load best model and prepare for attack # # # # # # # # 
    if hardcoded_path is not None:
        model_path = hardcoded_path
        print(f'Loading model from (hardcoded path) {model_path}...', end=' ')
    else:
        if best:
            model_path = osp.join(checkpoint, 'model_best.pth')
            print(f'Loading BEST model from {model_path}...', end=' ')
        else:
            model_path = osp.join(checkpoint, 'checkpoint.pth')
            print(f'Loading CURRENT model from {model_path}...', end=' ')

    try:
        print('Regular loading...', end=' ')
        ckpt = torch.load(model_path)
    except RuntimeError:
        print('Pretrained loading...', end=' ')
        model = copy_pretrained_model(model, path_to_copy_from=model_path)
    
    #model.load_state_dict(ckpt['state_dict'])

    try:
        test_acc1 = ckpt['test_acc']
        print(f'done. Loaded test acc: {test_acc1:4.3f}. ' \
            'Comparing with computed acc...', end=' ')
    except:
        pass

    if magnet_data is None:
        try:
            magnet_data = {
                'cluster_centers'   : ckpt['cluster_centers'],
                'cluster_classes'   : ckpt['cluster_classes'],
                'variance'          : ckpt['variance'],
                'normalize_probs'   : ckpt['normalize_probs'],
                'K'                 : ckpt['K'],
                'L'                 : ckpt['L'] if L is None else L
            }
            print('Loaded magnet data from checkpoint...', end=' ')
        except:
            print('Unable to load magnet data from checkpoint...', end=' ')
            pass # keep magnet_data as None, as it is a regular model and we're
                # not loading anything that has to do with magnet loss
    # Compare loaded accuracy with accuracy achieved by model
    model.eval()
    test_acc, _ = eval_model(model=model, dataset=dataset, 
        data_labels=data_labels, magnet_data=magnet_data)
    print(f'computed acc: {test_acc:4.3f}')
    if check_acc:
        assert np.isclose(test_acc1, test_acc), 'Stored acc should == computed acc'
    # # # # # # # # # # # # # Run actual attack # # # # # # # # # # # # # 
    alpha = alpha / distrib_params['std'].unsqueeze(0)
    epsilons = passed_epsilons.unsqueeze(1) / distrib_params['std'].unsqueeze(0)
    minima, maxima = distrib_params['minima'], distrib_params['maxima']
    print(f'Eps: {passed_epsilons} \t Images extrema: max: {maxima} \t ' \
        f'min: {minima}')
    # Initialize dataframe for storing robustness results
    df = {
        'epsilons' 		: [0., ],
        'test_set_accs' : [test_acc, ],
        'flip_rates' 	: [0., ],
    }
    for eps, eps_for_print in zip(epsilons, passed_epsilons):
        print(f'Running attack with epsilon = {eps_for_print:5.3f}')
        acc_test_set, flip_rate = attack_dataset(model, dataloader, device, 
            alpha, eps, distrib_params, rand_restarts, magnet_data, iterations)
        # Append to dataframe
        df['epsilons'].append(eps_for_print.item())
        df['test_set_accs'].append(acc_test_set)
        df['flip_rates'].append(flip_rate)
    
    # Convert dict to dataframe for saving as csv file
    df = pd.DataFrame.from_dict(df)
    print('Overall results: \n', df)
    if save_df:
        df.to_csv(osp.join(checkpoint, 'attack_results.csv'), index=False)
    
    return df


def attack_dataset(model, dataloader, device, alpha, eps, distrib_params,
        rand_restarts=10, magnet_data=None, iterations=50):
    # Reproducibility of robustness assessment
    seed = 222
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        
    model.eval()
    correct, total, flipped_decisions = 0, 0, 0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        total += images.size(0) # batch_size
        # Original predictions
        if magnet_data is not None: # use magnet loss
            _, embeddings = model(images)
            scores = get_softmax_probs(embeddings, magnet_data, 
                return_scores=True)
        else: # they are logits
            scores, _ = model(images)
        _, orig_preds = torch.max(scores.data, 1)
        # Compute adversarial examples
        final_adv_preds = orig_preds.clone()
        inds_left = torch.arange(final_adv_preds.size(0))
        for _ in range(rand_restarts):
            # Current images and labels
            c_ims, c_labs = images[inds_left], labels[inds_left]
            c_orig_preds = orig_preds[inds_left]
            # Compute adversarial examples for images that are left
            adv_images = attack_batch(model, c_ims, c_labs, c_orig_preds, eps, 
                alpha, distrib_params, iterations, magnet_data)
            # Compute prediction on adversarial examples
            if magnet_data is not None: # use magnet loss
                _, embeddings = model(adv_images)
                scores = get_softmax_probs(embeddings, magnet_data,
                    return_scores=True)
            else: # they are logits
                scores, _ = model(adv_images)
            _, adv_preds = torch.max(scores.data, 1)
            # Check instances in which attack was succesful
            where_success = c_orig_preds != adv_preds
            num_inds_where_success = inds_left[where_success]
            # Replace predictions where attack was succesful
            final_adv_preds[num_inds_where_success] = adv_preds[where_success]
            # Remove image indices for which an adversarial example was found
            inds_left = inds_left[~where_success]
            # Check if there are no images left
            if inds_left.size(0) == 0:
                break

        # Compare original predictions with predictions from adv examples
        flipped_idxs = orig_preds != final_adv_preds
        flipped_decisions += flipped_idxs.sum()
        correct += (final_adv_preds == labels).sum()

    # Compute final accuracy on test set and flip rate
    acc_test_set = 100 * float(correct) / total
    flip_rate = 100 * float(flipped_decisions) / total
    print(f'Accuracy of test set: {acc_test_set:5.4f}. '
        f'A total of {flipped_decisions} examples out of {total} were flipped: '
        f'{flipped_decisions}/{total} = {flip_rate:5.4f}%')

    return acc_test_set, flip_rate


def attack_batch(model, images, labels, orig_preds, eps, alpha, distrib_params,
        iterations=50, magnet_data=None, rand=True):
    # Inner function
    def get_rand_perturb(images, eps):
        # Between -eps and +eps
        return 2*eps*torch.rand_like(images) - eps

    criterion = nn.CrossEntropyLoss()
    device = images.device
    # Extract minima and maxima
    minima, maxima = distrib_params['minima'], distrib_params['maxima']
    # Unsqueeze so we don't need extra for-loop (and move to appropriate device)
    alphas = alpha.unsqueeze(2).unsqueeze(3).to(device)
    epss = eps.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    minima = minima.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    maxima = maxima.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    # Assume the image is a tensor
    if rand:
        pert_ims = images.data + get_rand_perturb(images, epss)
        pert_ims = channelwise_clamp(pert_ims, minima=minima, 
            maxima=maxima).data.clone()
    else:
        pert_ims = images.data.clone()

    # Check for which images was the attack already succesful
    inds_left = torch.arange(pert_ims.size(0))
    for _ in range(iterations): 
        # Gradient for the image
        pert_ims.requires_grad = True
        # Compute forward
        model.zero_grad()
        # Compute label predictions
        if magnet_data is not None: # use magnet loss
            _, embeddings = model(pert_ims[inds_left])
            scores = get_softmax_probs(embeddings, magnet_data, 
                return_scores=True)
        else: # they are logits, rather than probs but whatever
            scores, _ = model(pert_ims[inds_left])
        _, pert_preds = torch.max(scores.data, 1)
        # Check where the attack was not successful
        where_not_success = orig_preds[inds_left] == pert_preds
        # Remove image indices for which there's already an adversarial example
        inds_left = inds_left[where_not_success]
        if inds_left.size(0) == 0:
            break
        # Compute cost and do backward
        cost = criterion(scores[where_not_success], labels[inds_left])
        cost.backward()
        with torch.no_grad():
            # Sign of gradient times 'learning rate'
            eta = pert_ims.grad[inds_left].sign()
            pert_ims[inds_left] += alphas*eta
            # Project to noise within epsilon ball around original images
            noise = pert_ims[inds_left] - images[inds_left]
            noise = channelwise_clamp(noise, minima=-epss, maxima=epss)
            # Project to images within space of possible images
            pert_ims[inds_left] = channelwise_clamp(images[inds_left] + noise, 
                minima=minima, maxima=maxima)
            # Gradient to zero
            pert_ims.grad.zero_()

    return pert_ims


def final_attack_eval(model, testloader, testset, test_labels, checkpoint,
        distrib_params, device, standard_epsilons, alpha_step, L, seed, 
        normalize_probs, evaluate_ckpt=None, restarts=10, attack_iters=50, evaluate_only=False):
    # if not evaluate_only:
    #     _ = eval_robustness(
    #         model, testloader, testset, test_labels, checkpoint, 
    #         distrib_params, device, standard_epsilons, best=True,
    #         hardcoded_path=evaluate_ckpt, magnet_data=None,
    #         check_acc=False, alpha=alpha_step, L=L, 
    #         rand_restarts=restarts, iterations=attack_iters
    #     )
    checkpoint1 = evaluate_ckpt if evaluate_ckpt is not None else checkpoint
    external_eval_dataset(
        seed=seed, checkpoint=checkpoint1, device=device, 
        model=model, dataloader=testloader, distrib_params=distrib_params, 
        L=L, normalize_probs=normalize_probs,
        restarts=restarts, attack_iters=attack_iters, save_checkpoint = checkpoint
    )

'''
External utilities taken from
https://github.com/anonymous-sushi-armadillo/fast_is_better_than_free_CIFAR10/blob/master/evaluate_cifar.py#L45
To check our (amazing!) results
Code was copied and then modified
'''
def external_attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
        lower_limit, upper_limit, device, chnls, return_iters=False):
    # 'clamp' function taken from
    # https://github.com/anonymous-sushi-armadillo/fast_is_better_than_free_CIFAR10/blob/master/evaluate_cifar.py#L31
    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    final_iters = torch.zeros_like(y.squeeze())
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        iters = torch.zeros_like(y.squeeze())
        delta = torch.zeros_like(X).to(device)
        delta[:,0,:,:].uniform_(-epsilon[0][0][0].item(), epsilon[0][0][0].item())
        if chnls > 1:
            delta[:,1,:,:].uniform_(-epsilon[1][0][0].item(), epsilon[1][0][0].item())
            delta[:,2,:,:].uniform_(-epsilon[2][0][0].item(), epsilon[2][0][0].item())
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            x = X[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
            # Increase iterations counter
            iters[index[0]] += 1

        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        final_iters[all_loss >= max_loss] = iters.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    
    if return_iters:
        return max_delta, final_iters
    else:
        return max_delta


class MagnetModelWrapper(nn.Module):
    def __init__(self, model, magnet_data, mean, std):
        super(MagnetModelWrapper, self).__init__()
        self.model = model
        self.magnet_data = magnet_data
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.magnet_data is None:
            # If TRADES model -> renormalize data
            x = x * self.std + self.mean if self.mean is not None else x
            scores, _ = self.model(x)
        else:
            _, embeddings = self.model(x)
            scores = get_softmax_probs(embeddings, self.magnet_data, 
                return_scores=True)
        return scores


def external_eval_dataset(seed, checkpoint, device, model, dataloader, 
        distrib_params, attack_iters=50, restarts=10, L=None, 
        normalize_probs=None, save_df=True, save_checkpoint = None):
    if save_checkpoint is None:
        save_checkpoint = checkpoint
    # Initialize seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Load the checkpoint
    if not checkpoint.endswith('.pth'):
        print('Checkpoint isnt ".pth" file: inferring model_best.pth')
        ckpt = torch.load(osp.join(checkpoint, 'model_best.pth'))
    else:
        ckpt = torch.load(checkpoint)
    # Construct magnet data
    try:
        magnet_data = {
            'cluster_classes'   : ckpt['cluster_classes'],
            'cluster_centers'   : ckpt['cluster_centers'],
            'variance'          : ckpt['variance'],
            'L'                 : ckpt['L'] if L is None else L,
            'K'                 : ckpt['K'],
            'normalize_probs'   : ckpt['normalize_probs'] if normalize_probs is None else normalize_probs
        }
        print('Succesfully loaded magnet_data from checkpoint')
    except:
        magnet_data = None
        print('Unable to load magnet_data from checkpoint. '
            'Regular training is inferred')

    # Load and prepare the model
    try:
        state_dict = ckpt['state_dict']
        mean_for_model = None
        std_for_model = None
    except:
        print(f'Checkpoint "{checkpoint}" does not contain a dict. Assuming '
            f'it is the state_dict itself (like TRADES models)')
        state_dict = { k.replace('model.', '') : v for k, v in ckpt.items()}
        # Pass parameters to model -> data is renormalized for TRADES models
        mean_for_model = distrib_params['mean'].view(1,3,1,1).to(device)
        std_for_model = distrib_params['std'].view(1,3,1,1).to(device)
    
    model.load_state_dict(state_dict)
    model_wrapper = MagnetModelWrapper(model, magnet_data, mean_for_model, std_for_model)
    model_wrapper.eval()
    model_wrapper.float()
    # Prepare for dataframe in case it is cifar100
    num_classes = len(dataloader.dataset.classes)
    is_cifar100 = isinstance(dataloader.dataset, CIFAR100) and num_classes == 100
    if is_cifar100:
        class_names = get_classname_list(num_classes, f=CLASS_NAMES_FILE)
    # Clean accuracy
    n, total_acc, total_loss = 0, 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        output = model_wrapper(X)
        preds = output.max(1)[1]
        loss = F.cross_entropy(output, y)
        total_loss += loss.item() * y.size(0)
        total_acc += (preds == y).sum().item()
        n += y.size(0)

    acc, loss = 100.*total_acc/n, total_loss/n
    print(f'eps=0.0: Loss: {loss:.4f}, Acc: {acc:.4f}')
    # Initialize dataframe for storing robustness results
    df = {'epsilons'    : [0.,],    'test_set_accs' : [acc,]}
    # # Attack params
    epsilons = [2, 8, 16, 255*0.1]
    std, mu = distrib_params['std'].squeeze(), distrib_params['mean'].squeeze()
    # Reshape to comply with this external code
    chnls = X.size(1)
    std, mu = std.view(chnls, 1, 1), mu.view(chnls, 1, 1)
    alpha = ((2 / 255.) / std).to(device)
    lower_limit, upper_limit = ((0 - mu)/ std), ((1 - mu)/ std)
    lower_limit, upper_limit = lower_limit.to(device), upper_limit.to(device)
    # Begin all attacks
    for eps in epsilons:
        epsilon = ((eps / 255.) / std).to(device)
        n, total_acc, total_loss = 0, 0, 0
        # Initialize dataframe for this epsilon (a dict, for now)
        if is_cifar100:
            df_eps = { col : [] for col in DF_COLS if 'name' not in col}
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            delta, pgd_iters = external_attack_pgd(
                model_wrapper, X, y, epsilon, alpha, attack_iters, restarts,
                lower_limit, upper_limit, device, chnls, return_iters=True
            )
            with torch.no_grad():
                output = model_wrapper(X + delta)
                adv_pred = output.max(1)[1]
                loss = F.cross_entropy(output, y)
                if is_cifar100:
                    # Compute natural prediction and norm of perturbation
                    nat_pred = model_wrapper(X).max(1)[1]
                    linf_norm = torch.flatten(delta, 1).abs().max(dim=1)[0]
                    # Update the dictionary
                    df_eps = update_dict_info(df_eps, y, nat_pred, adv_pred, 
                        linf_norm, pgd_iters)
                
                total_loss += loss.item() * y.size(0)
                total_acc += (adv_pred == y).sum().item()
                n += y.size(0)
        
        if is_cifar100:
            # Convert df_eps to dataframe
            df_eps = pd.DataFrame.from_dict(df_eps)
            # Compute name columns
            name_map = lambda x: class_names[x]
            df_eps['label_name'] = df_eps['label'].apply(name_map)
            df_eps['nat_name'] = df_eps['nat_pred'].apply(name_map)
            df_eps['adv_name'] = df_eps['adv_pred'].apply(name_map)
            # Save dataframe
            filename = osp.join(save_checkpoint, f'predictions_eps{eps}.csv')
            print(f'Saving predictions DataFrame to "{filename}"...', end=' ')
            df_eps.to_csv(filename, index=False)
            print('done.')

        acc, loss = 100.*total_acc/n, total_loss/n
        print(f'eps={eps:.4f}: Loss: {loss:.4f}, Acc: {acc:.4f}')
        df['epsilons'].append(eps)
        df['test_set_accs'].append(acc)

    # Convert dict to dataframe for saving as csv file
    df = pd.DataFrame.from_dict(df)
    print('Overall results: \n', df)
    if save_df:
        df.to_csv(osp.join(save_checkpoint, 'attack_results_ext.csv'), index=False)


def evaluate_hierarch_err(preds, labels):
    errors = SEM_DISTS[preds, labels]
    return errors


def get_classname_list(num_classes, f):
    # The file is like
    # 0 apple
    # 1 aquarium_fish
    # 2 baby
    # 3 bear
    class_names = []
    with open(f) as fp: 
        for idx, line in enumerate(fp):
            s_line = line.strip()
            class_names.append(s_line[s_line.find(' '):])
            if idx == num_classes-1: break

    return class_names


def update_dict_info(df, y, nat_pred, adv_pred, linf_norm, 
        pgd_iters):
    # Compute natural and adversarial error
    nat_h_error = evaluate_hierarch_err(nat_pred, y)
    adv_h_error = evaluate_hierarch_err(adv_pred, y)
    # column names and values to update
    cols_vals = {
        'label'         : y,
        'nat_pred'      : nat_pred,
        'adv_pred'      : adv_pred,
        'nat_h_error'   : nat_h_error,
        'adv_h_error'   : adv_h_error,
        'linf_norm'     : linf_norm,
        'pgd_iters'     : pgd_iters,
    }
    for col_name, value in cols_vals.items():
        df[col_name].extend(value.cpu().numpy().tolist())

    return df

