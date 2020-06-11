"""
ClusterBatchBuilder framework ported from 
https://github.com/pumpikano/tf-magnet-loss/blob/master/magnet_tools.py.
"""
import time
import numpy as np
from tqdm import tqdm
from math import ceil

from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import pdb, traceback, sys

class ClusterBatchBuilder(object):
    """
    Sample minibatches for magnet loss.
    k: clusters per class
    m: number of clusters to sample (1 is the seed cluster, the 
        remaining m-1 are the 'impostor' clusters)
    d: samples per cluster
    """
    def __init__(self, labels, k, m, d, device, kmeans_inits=1, max_iter=20, 
            eps=1e-8):

        if isinstance(labels, np.ndarray):
            self.num_classes = np.unique(labels).shape[0]
            self.labels = labels
        else:
            self.num_classes = np.unique(labels.numpy()).shape[0]
            self.labels = labels.numpy()

        self.k = k
        self.m = m
        self.d = d
        self.centroids = None
        self.rng = np.random.default_rng()
        self.expected_batch_size = self.m * self.d
        self.sampl_fun = self.orig_sampling
        self.max_iter = max_iter # 20 if cifar10 else 40
        self.kmeans_inits = kmeans_inits # 1 if cifar10 else 10

        # Assignments for each instance
        if isinstance(labels, np.ndarray):
            self.assignments = np.zeros_like(labels, int)
        else:
            self.assignments = np.zeros_like(labels.numpy(), int)

        # cluster_assignments maps from the (index of the) cluster 
        # to the array of indices of samples that belong to THAT cluster
        self.cluster_assignments = {}
        # This is the behavior of np.repeat:
        # np.repeat(range(3), 2)                                                                                                         
        # >> array([0, 0, 1, 1, 2, 2])
        self.cluster_classes = torch.from_numpy(
            np.repeat(range(self.num_classes), k)
        )
        self.device = device
        self.example_losses = None
        self.cluster_losses = None
        self.has_loss = None
        self.eps = eps


    def update_clusters(self, rep_data):
        """Given an array of representations for the entire training set,
        recompute clusters and store example cluster assignments in a
        quickly sampleable form."""
        # Lazily allocate array for centroids
        if self.centroids is None:
            # There are (num classes) * (num cluster per class) clusters,
            # each of the same dimension as the representation space
            self.centroids = torch.zeros([self.num_classes * self.k, rep_data.shape[1]])
            self.centroids = self.centroids.to(self.device)

        # Iterate through all classes
        for c in range(self.num_classes):
            # For a certain class, check which labels are of that class
            class_mask = self.labels == c
            # Extract the representation of the samples of that class
            class_examples = rep_data[class_mask].cpu().numpy()
            # Perform kmeans on those samples
            kmeans = KMeans(n_clusters=self.k, init='k-means++', random_state=1,
                n_init=self.kmeans_inits, max_iter=self.max_iter)
            kmeans.fit(class_examples)
            # Save centroids for finding impostor clusters
            # Note we save cluster inds, which are global, rather local ones
            start = self.get_cluster_ind(c, 0)
            stop = self.get_cluster_ind(c, self.k)
            # Assign centroids (in all columns - representation dimension)
            self.centroids[start:stop] = torch.from_numpy(
                kmeans.cluster_centers_).to(self.device)
            # Update assignments with new (global) cluster indexes
            self.assignments[class_mask] = self.get_cluster_ind(c, 
                kmeans.labels_)
            # self.assignments maps from each sample to its global cluster 
            # index, that is, the index of the cluster to which it belongs

        # Construct a map: cluster to example indexes for fast batch creation
        for cluster in range(self.k * self.num_classes):
            # Use global cluster indices
            cluster_mask = self.assignments == cluster
            # Save indices (for self.assignments) that correspond to each cluster
            # each entry of self.cluster_assignments can serve as index for self.assignments
            self.cluster_assignments[cluster] = np.flatnonzero(cluster_mask)
            # self.cluster_assignments is something like this:
            # {0: array([   29,    49,   115, ..., 49856, 49863, 49930]), 
            #  1: array([   30,    35,    77, ..., 49941, 49992, 49994]),...}


    def update_losses(self, indexes, losses):
        """Given a list of examples indexes and corresponding losses
        store the new losses and update corresponding cluster losses."""

        # Lazily allocate structures for losses
        if self.example_losses is None:
            # Loss of each sample
            self.example_losses = torch.zeros_like(torch.from_numpy(self.labels), 
                dtype=torch.float32, device=self.device)
            # Loss of each cluster (counting all clusters of all classes)
            self.cluster_losses = torch.zeros([self.k * self.num_classes], 
                dtype=torch.float32, device=self.device)
            # A binary mask keeping count of whose loss has/n't been computed
            self.has_loss = torch.zeros_like(torch.from_numpy(self.labels), 
                dtype=bool, device=self.device)

        # Update losses of the examples that arrived
        self.example_losses[indexes] = losses
        # Update which examples have a loss
        self.has_loss[indexes] = True

        # Find affected clusters and update the corresponding cluster losses
        # Note that self.assignments maps from each sample to its global cluster index
        clusters = np.unique(self.assignments[indexes])
        for cluster in clusters:
            # The loss of the samples that belong to the current cluster
            cluster_inds = self.assignments == cluster
            cluster_example_losses = self.example_losses[cluster_inds]

            # Take the average loss in the cluster of examples for which we have measured a loss
            self.cluster_losses[cluster] = torch.mean(cluster_example_losses[self.has_loss[cluster_inds]])

    def gen_batch(self):
        return self.sampl_fun()

    def orig_sampling(self):
        """Sample a batch by first sampling a seed cluster proportionally to
        the mean loss of the clusters, then finding nearest neighbor
        "impostor" clusters, then sampling (at most) d examples uniformly 
        from each cluster.

        The generated batch will consist of m clusters each with d consecutive
        examples."""

        # Sample seed cluster proportionally to cluster losses if available
        p = None # means cluster choice will assume uniform distribution
        if self.cluster_losses is not None:
            denominator = torch.sum(self.cluster_losses)
            # Check denominator
            if torch.allclose(denominator, torch.tensor(0.0)):
                denominator = self.eps
            p = self.cluster_losses / denominator
            p = p.cpu().numpy()
            
        seed_cluster = np.random.choice(self.num_classes * self.k, p=p)
        # Assure only clusters of different class from seed are chosen
        where_diff_class = self.get_class_ind(seed_cluster) != self.cluster_classes
        diff_class_clusters = torch.nonzero(where_diff_class).squeeze()

        # Get impostor clusters by ranking centroids by distance
        # sq_dists.shape == diff_class_clusters.shape, that is, the distances
        # from the seed cluster to all the other clusters
        sq_dists = torch.cdist(
            self.centroids[seed_cluster].unsqueeze(0), 
            self.centroids[where_diff_class],
            p=2
        ).squeeze()

        # Get top impostor clusters and add seed
        _, closest_inds = torch.topk(sq_dists, k=self.m-1, largest=False, sorted=False)
            
        clusters = diff_class_clusters[closest_inds]
        clusters = torch.tensor([seed_cluster, *clusters])

        # Sample examples uniformly from all clusters to be sampled
        batch_indexes, clust_assigns = [], []
        for c in clusters:
            # Need to take 'd' samples per cluster and store them in batch_indexes
            n_clust_samples = len(self.cluster_assignments[c.item()])
            assert n_clust_samples > 1, 'Num of samples in cluster must be larger than 1'
            if n_clust_samples < self.d:
                print(
                    f'Num. of samples ({n_clust_samples}) for cluster {c.item()} is '
                    f'insufficient for taking {self.d} samples. Sampling ENTIRE cluster.'
                )
            x = np.random.choice(
                self.cluster_assignments[c.item()], 
                size=min(n_clust_samples, self.d), 
                replace=False
            ).tolist()
            # Save indices of samples
            batch_indexes.extend(x)
            # Save the cluster assignment of each sample
            clust_assigns.extend([c for _ in range(len(x))])

        # Convert to numpy array
        clust_assigns = np.array(clust_assigns)
        batch_indexes = np.array(batch_indexes)
        # Get class inds
        class_inds = self.get_class_ind(clust_assigns)

        return batch_indexes, class_inds, clust_assigns

    def rand_sampling(self):
        batch_indexes = self.rng.choice(
            self.labels.shape[0], 
            size=self.expected_batch_size, 
            replace=False
        )
        class_inds = self.labels[batch_indexes]
        clust_assigns = self.assignments[batch_indexes]
        return batch_indexes, class_inds, clust_assigns


    def get_cluster_ind(self, c, i):
        """Given a class index and a cluster index within the class
        return the global cluster index"""
        return c * self.k + i

    def get_class_ind(self, c):
        """Given a cluster index return the class index."""
        # Make sure it is integer division
        return c // self.k
