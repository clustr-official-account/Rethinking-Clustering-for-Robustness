from utils.magnet_tools import ClusterBatchBuilder
from utils.utils import compute_reps, compute_real_losses

import torch

def magnet_assertions(train_labels, k, L, m):
    tot_clusts = torch.unique(train_labels).size(0) * k
    assert L <= tot_clusts, f'NNs ({L}) <= tot clusts ({tot_clusts})'
    assert m <= tot_clusts, f'Samp. clusts ({m}) ≤ tot clusts ({tot_clusts})'


def get_batch_builders(model, trainset, train_labels, K, M, D, device, 
        dataset_name):
    _, reps = compute_reps(model, trainset, 400)
    # Kmeans parameters for the batch builder
    max_iter = 20 if dataset_name == 'cifar10' else 40
    kmeans_inits = 1 if dataset_name == 'cifar10' else 10
    # Create batcher
    batch_builder = ClusterBatchBuilder(train_labels, K, M, D, device, 
        max_iter=max_iter, kmeans_inits=kmeans_inits)
    batch_builder.update_clusters(reps)
    
    return batch_builder


def get_magnet_data(batch_builder, device, args, model, criterion, trainset, 
        train_labels):
    print('Computing initial magnet data (variance and centroids)...', end=' ')
    magnet_data = { # don't include variance yet, as it hasnt been computed
        'cluster_classes' 	: batch_builder.cluster_classes.to(device),
        'cluster_centers' 	: batch_builder.centroids.to(device), 
        'L' 				: args.L, 
        'K' 				: args.k,
        'consistency_lambda': args.consistency_lambda,
        'mse_consistency'   : args.mse_consistency,
        'normalize_probs'   : not args.not_normalize,
        'xent_lambda'       : args.xent_lambda
    }
    # Compute variance in train set and store it in 'magnet_data' dict
    _, train_variance = compute_real_losses(model, criterion, trainset, 
        train_labels, magnet_data=magnet_data, compute_variance=True)
    magnet_data.update({'variance' : train_variance})
    print('done.')

    return magnet_data
