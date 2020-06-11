import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb

class MagnetLoss(nn.Module):
    """
    Magnet loss technique presented in the paper:
    ''Metric Learning with Adaptive Density Discrimination'' by Oren Rippel, Manohar Paluri, Piotr Dollar, Lubomir Bourdev in
    https://research.fb.com/wp-content/uploads/2016/05/metric-learning-with-adaptive-density-discrimination.pdf?

    Args:
        r: A batch of features.
        classes: Class labels for each example.
        clusters: Cluster labels for each example.
        cluster_classes: Class label for each cluster.
        n_clusters: Total number of clusters.
        alpha: The cluster separation gap hyperparameter.

    Returns:
        total_loss: The total magnet loss for the batch.
        losses: The loss for each example in the batch.
    """
    def __init__(self, alpha=1.0, epsilon=1e-8):
        super(MagnetLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, r, classes, assigned_clusters):
        # This is the STOCHASTIC APPROXIMATION to the loss of the paper
        # 'within_batch_classes' need not be the actual classes. They are just 
        # arbitrary IDs that say, in the batch of features, r, which examples 
        # correspond to the same class

        # Need to compute the centroids we can from the representations passed
        # Take cluster means within the batch
        # r.shape == [batch_size, emb_dimension]
        # This line breaks r into groups belonging to the same cluster
        lengths, unique_clusts = compute_partition(assigned_clusters)
        cluster_examples = torch.split(r, lengths)
        # cluster_examples[i].shape == [num of instances sampled for ith cluster, emb_dimension]
        # By taking the mean of each entry in cluster_examples
        # we get the center of mass of each cluster
        cluster_means = torch.stack([torch.mean(x, dim=0) for x in cluster_examples])
        # cluster_means.shape == [m (clusters sampled), emb_dimension]
            
        # Compute distance from all instances to all clusters
        sample_costs = torch.cdist(r, cluster_means, p=2)**2
        # sample_costs.shape == [batch_size, self.tot_clusters]
        clusters_tensor = torch.from_numpy(assigned_clusters).to(r.device)
        n_clusters_tensor = torch.from_numpy(unique_clusts).to(r.device)

        # Kind-of binary mask that states, for each instance, the cluster it belongs to
        intra_cluster_mask = comparison_mask(clusters_tensor, n_clusters_tensor)
        # (Squared) Distance from each instance to its own cluster
        # intra_cluster_costs.shape == [batch_size]
        intra_cluster_costs = torch.sum(intra_cluster_mask * sample_costs, dim=1)

        # This is the batch size
        N = r.size(0)
        # Average variance (mean of the variances of the points to their corresponding mean)
        # variance is a SCALAR!
        variance = torch.sum(intra_cluster_costs) / float(N - 1)
        var_normalizer = -1 / (2 * variance)

        # Compute numerator
        # numerator.shape == [batch_size]
        numerator = torch.exp(var_normalizer * intra_cluster_costs - self.alpha)

        # Get the class for each cluster
        # like [4, 0, 1, 2, 3, 5, 5, 6, 7, 8, 9, 8] (12 clusters here)
        cluster_classes = torch.from_numpy(classes[np.cumsum(lengths)-1]).to(r.device)
        # The classes for each instance
        classes = torch.from_numpy(classes).to(r.device)
        # Compute denominator (compare in terms of class -- not cluster)
        # diff_class_mask.shape == [batch_size, n_clusters]
        equal_class_mask = comparison_mask(classes, cluster_classes)
        # diff_class_mask says, for each instance, which clusters are 
        # of a different class than that of the instance 
        diff_class_mask = torch.logical_not(equal_class_mask) # logical not
        denom_sample_costs = torch.exp(var_normalizer * sample_costs)
        denominator = torch.sum(diff_class_mask * denom_sample_costs, dim=1)
        
        # Compute main term. losses.shape == [batch_size]
        losses = torch.relu(-torch.log(numerator / (denominator + self.epsilon) + self.epsilon))

        # Compute the mean (Eq. 5)
        total_loss = torch.mean(losses)
        return total_loss, losses, variance

    def compute_total_loss(self, r, labels, magnet_data, variance=None):
        centroids = magnet_data['cluster_centers']
        centroid_classes = magnet_data['cluster_classes']
        # centroids.shape == [k*c, emb_dim]
        # where k = num of clusters per class
        # and c = num of classes
        
        # the first k clusters belong to label 0, the 
        # second k to label 1 and so on

        # for each instance in r we must find the closest 
        # centroid OF ITS OWN CLASS

        # Compute assignments
        # r.shape == [num_samples, emb_dim]
        # centroids.shape == [num_centroids (classes*args.k), emb_dim]
        # sq_dists.shape == [r.size(0) (batch_size), num_centroids]
        sq_dists = torch.cdist(r, centroids, p=2)**2
        # Check, for each instance, the centroids that are of its same class
        labels = labels.to(r.device)
        where_same_class = comparison_mask(labels, centroid_classes)
        # Extract those distance (view to make them of appropiate size)
        # dists_to_same_class_clust.shape == [r.size(0), num_cluster per class]
        dists_to_same_class_clust = sq_dists[where_same_class].view(r.size(0), -1)
        dist_to_assigned_clust, assigned_cluster_ind = dists_to_same_class_clust.min(dim=1)
        N = r.size(0)
        # Compute index of the assigned cluster for each instance
        clust_inds = torch.arange(centroid_classes.size(0), device=r.device).unsqueeze(0)
        clust_inds = clust_inds.expand(where_same_class.size(0), clust_inds.size(1))
        clust_inds = clust_inds[where_same_class].view(r.size(0), -1)
        # These are the indices of the clusters to which each instance belongs
        assigned_cluster_ind = assigned_cluster_ind.unsqueeze(1)
        clust_inds = torch.gather(clust_inds, dim=1, index=assigned_cluster_ind).squeeze()
        # If the variance was not given, compute it
        if variance is None:
            assert N != 1, 'Number of samples must be != 1 for variance computation'
            variance = dist_to_assigned_clust.sum() / float(N - 1)
            # Expand so that dimensions are consistent when computing individual variances
            variance = variance.unsqueeze(0).expand(centroids.size(0))
            # this is equivalent to having all the clusters with the SAME variance
            
            variance = variance.detach()

        # # Compute numerator
        var_normalizer = -1 / (2 * variance)
        assigned_clust_var_normalizer = torch.gather(var_normalizer, dim=0, index=clust_inds)
        numerator = torch.exp(assigned_clust_var_normalizer * dist_to_assigned_clust - self.alpha)
        # numerator.shape == [batch_size]

        # # Compute denominator
        # Check which clusters are from OTHER CLASSES
        num_centroids = centroids.size(0)
        repeated_classes = centroid_classes.unsqueeze(0).expand(N, num_centroids)
        labels = labels.unsqueeze(1).expand(N, num_centroids).to(repeated_classes.device)
        other_class_mask = labels != repeated_classes

        denominator_terms = torch.exp(var_normalizer.unsqueeze(0) * (other_class_mask * sq_dists))
        denominator = torch.sum(denominator_terms, dim=1)

        # Compute loss
        losses = torch.relu(-torch.log(numerator / (denominator + self.epsilon) + self.epsilon))

        return losses.mean(), variance


def expand_dims(var, dim=0):
    """ Is similar to [numpy.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html).
        var = torch.range(0, 9).view(-1, 2)
        torch.expand_dims(var, 0).size()
        # (1, 5, 2)
    """
    sizes = list(var.size())
    sizes.insert(dim, 1)
    return var.view(*sizes)

def comparison_mask(a_labels, b_labels):
    """Computes boolean mask for distance comparisons"""
    return torch.eq(expand_dims(a_labels, 1),
                    expand_dims(b_labels, 0))

def dynamic_partition(X, n_clusters):
    """Partitions the data into the number of cluster bins"""
    cluster_bin = torch.chunk(X, n_clusters)
    return cluster_bin

def compute_partition(clust_assigns):
    # clust_assign is something like
    # Y == np.array([4,6,6,6,7,7,1])
    where_change = np.nonzero(np.diff(clust_assigns))[0]
    # where_change = np.nonzero(np.diff(Y))[0] == array([0, 3, 5])
    # add index for last group -> array([0, 3, 5, 6])
    where_change = np.append(where_change, len(clust_assigns)-1)
    # extract unique clusters
    unique_clusts = clust_assigns[where_change]
    # add index for first group -> array([-1,  0,  3,  5,  6])
    where_change = np.append(-1, where_change)
    # compute lengths -> array([1, 3, 2, 1])
    lengths = np.diff(where_change)
    return lengths.tolist(), unique_clusts


def compute_euclidean_distance(x, y):
    return torch.sum((x - y)**2, dim=2)
