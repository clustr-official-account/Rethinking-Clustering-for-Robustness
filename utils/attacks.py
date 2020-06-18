import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import os.path as osp

from utils.utils import get_softmax_probs

import torch
import torch.nn as nn
import torch.nn.functional as F

def final_attack_eval(model, testloader, testset, test_labels, checkpoint,
        distrib_params, device, standard_epsilons, alpha_step, L, seed, 
        normalize_probs, evaluate_ckpt=None, restarts=10, attack_iters=50,
        evaluate_only=False):
    checkpoint1 = evaluate_ckpt if evaluate_ckpt is not None else checkpoint
    external_eval_dataset(
        seed=seed, checkpoint=checkpoint1, device=device, model=model, 
        dataloader=testloader, distrib_params=distrib_params, L=L, 
        normalize_probs=normalize_probs, restarts=restarts, 
        attack_iters=attack_iters, save_checkpoint=checkpoint
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
    num_classes = len(dataloader.dataset.classes)
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
                total_loss += loss.item() * y.size(0)
                total_acc += (adv_pred == y).sum().item()
                n += y.size(0)

        acc, loss = 100.*total_acc/n, total_loss/n
        print(f'eps={eps:.4f}: Loss: {loss:.4f}, Acc: {acc:.4f}')
        df['epsilons'].append(eps)
        df['test_set_accs'].append(acc)

    # Convert dict to dataframe for saving as csv file
    df = pd.DataFrame.from_dict(df)
    print('Overall results: \n', df)
    if save_df:
        df.to_csv(osp.join(save_checkpoint, 'attack_results_ext.csv'), 
            index=False)

