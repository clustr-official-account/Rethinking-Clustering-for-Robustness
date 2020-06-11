import os
import pdb
import copy
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import ceil
from time import time
import os.path as osp
import matplotlib.pyplot as plt

# Torch-related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# change pandas df's format
pd.options.display.float_format = '{:4.3f}'.format

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from 
       https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def channelwise_clamp(images, minima, maxima):
    # torch.clamp does not allow the limits to be 
    # tensors, but torch.min and max DOES!
    images = torch.max(torch.min(images, maxima), minima)
    return images


def compute_reps(model, X, chunk_size):
    """Compute representations for input in chunks."""
    chunks = int(ceil(float(len(X)) / chunk_size))
    reps, logits = [], []

    trainloader = DataLoader(X, batch_size=chunks,
        shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for img, _ in trainloader:
        logit, rep = model(img.to(device))
        reps.append(rep.data)
        logits.append(logit.data)
    
    return torch.cat(logits), torch.cat(reps)


def get_softmax_probs(embeddings, magnet_data, return_scores=False):
    '''
    This code was taken from
    https://github.com/mbanani/pytorch-magnet-loss/blob/9919be9684f765aaaa33ef49ad08502a342bf417/util/kNN_metrics.py#L78
    and then we modified it
    Love you, mbanani
    '''
    device = embeddings.device
    num_clusters = magnet_data['cluster_classes'].size(0)
    num_classes = num_clusters // magnet_data['K']
    batch_size = embeddings.size(0)
    # distances states, for each instance, the distance to each cluster
    # Compute squared distances
    sq_distances = torch.cdist(
        embeddings, magnet_data['cluster_centers'], p=2)**2
    # Scale distances with variances
    sq_distances = sq_distances / magnet_data['variance'].unsqueeze(0)
    # Compute probs
    scores = torch.exp(-0.5 * sq_distances)
    largest_scores, indices = torch.topk(scores, k=magnet_data['L'], 
        largest=True, sorted=False)
    # Reshape to sum across clusters of the same class
    # this is of size [batch_size, num_classes, num_clusters]
    scores = scores.view(batch_size, num_classes, magnet_data['K'])
    # Perform sum (in the cluster dimension)
    scores = scores.sum(dim=2)
    # Normalizing factors
    # Get top-L CLOSEST (highest probabilities) clusters (need not be sorted)
    if magnet_data['normalize_probs']:
        labs_clusters = magnet_data['cluster_classes'].unsqueeze(0)
        labs_clusters = labs_clusters.expand(batch_size, num_clusters)
        labs_clusters = torch.take(labs_clusters, indices)
        scores = torch.zeros(batch_size, num_classes, device=device)
        unique_labels = torch.unique(labs_clusters)
        for label in unique_labels:
            label_mask = labs_clusters == label
            scores[:, label] = (largest_scores * label_mask).sum(dim=1)

    # Normalize probabilities in the class dimension (rows)
    to_return = scores if return_scores else F.normalize(scores, p=1, dim=1)
    return to_return


def accuracy(probs, target, topk=(1,)):
    # Taken from
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L407
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = probs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred).to(pred.device))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def magnet_accuracy(embeddings, target, magnet_data, topk=(1,)):
    """Compute the accuracy over k top predictions for specified values of k"""
    softmax_probs = get_softmax_probs(embeddings, magnet_data)
    return accuracy(probs=softmax_probs, target=target, topk=topk)


def eval_model(model, dataset, data_labels, magnet_data=None):
    # default values are filled if standard training, ie magnet_data is None
    # Evaluate using kNC
    # https://github.com/mbanani/pytorch-magnet-loss/blob/master/train_magnet.py#L274
    loss = None
    model.eval()
    # Train set
    if magnet_data is not None: # dealing with magnet loss
        _, data_reps = compute_reps(model, dataset, 400)
        acc = magnet_accuracy(embeddings=data_reps, target=data_labels, 
            magnet_data=magnet_data)
    else: # for regular training, compute accuracy AND loss
        logits, _ = compute_reps(model, dataset, 400)
        # accuracies
        acc = accuracy(probs=logits, target=data_labels)
        # losses
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            loss = criterion(logits, data_labels.to(logits.device)).item()

    return acc[0].item(), loss


def compute_real_losses(model, criterion, dataset, data_labels, magnet_data,
        compute_variance):
    variance = None if compute_variance else magnet_data['variance']
    _, rs = compute_reps(model, dataset, 400)
    loss, var = criterion.compute_total_loss(r=rs, labels=data_labels, 
        magnet_data=magnet_data, variance=variance)
    
    return loss, var


def copy_pretrained_model(model, 
        path_to_copy_from='./pretrained_models/resnet18.pt'):
    resnet = torch.load(path_to_copy_from, map_location='cuda')
    if 'state_dict' in resnet.keys():
        resnet = resnet['state_dict']
    keys = list(resnet.keys())
    count = 0
    for key in model.state_dict().keys():
        model.state_dict()[key].copy_(resnet[keys[count]].data)
        count +=1
    
    print('Pretrained model is loaded successfully')
    return model
