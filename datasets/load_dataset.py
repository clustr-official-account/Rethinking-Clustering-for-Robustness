import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

import torchvision
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torchvision.transforms import (Compose, RandomCrop, RandomHorizontalFlip, 
    ToTensor, Normalize)

# from datasets.fashion import FASHION

import numpy as np
from utils.sampler import SubsetSequentialSampler as SubSeqSamp

BEST_MODEL_THRESHOLDS = {
    'cifar10'   : 80.0,
    'cifar100'  : 60.0,
    'svhn'      : 85.0
}
DATASET_CLASSES = {
    'cifar10'   : 10,
    'cifar100'  : 100,
    'svhn'      : 10
}
# Statistics for SVHN taken from
# https://www.programcreek.com/python/example/105105/torchvision.datasets.SVHN
DATASET_MEANS = {
    'cifar10'   : [0.4914, 0.4822, 0.4465],
    'cifar100'  : [0.4914, 0.4822, 0.4465],
    'svhn'      : [0.4378, 0.4439, 0.4729]
}
DATASET_STDS = {
    'cifar10'   : [0.2023, 0.1994, 0.2010],
    'cifar100'  : [0.2023, 0.1994, 0.2010],
    'svhn'      : [0.1980, 0.2011, 0.1971]
}
DATASET_ROOT = './data'

def load_dataset(args, magnet_training):
    ''' Loads the dataset specified '''
    print('==> Preparing data...')
    mean = torch.tensor(DATASET_MEANS[args.dataset])
    std = torch.tensor(DATASET_STDS[args.dataset])
    # CIFAR-10,100 datasets
    if 'cifar' in args.dataset:
        # Data
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean, std),
        ])
        dataset = CIFAR10 if args.dataset == 'cifar10' else CIFAR100
    # SVHN
    elif args.dataset == 'svhn':
        transform_train = Compose([ToTensor(), Normalize(mean, std)])
        dataset = SVHN

    transform_test = Compose([ToTensor(), Normalize(mean, std)])
    # Call to datasets.SVHN is different from the others (because of reasons)
    if args.dataset == 'svhn':
        trainset = dataset(root=DATASET_ROOT, split='train', 
            transform=transform_train, download=True)
        testset = dataset(root=DATASET_ROOT, split='test', 
            transform=transform_test, download=True)
    else:
        trainset = dataset(root=DATASET_ROOT, train=True, 
            transform=transform_train, download=True)
        testset = dataset(root=DATASET_ROOT, train=False, 
            transform=transform_test, download=True)

    # Deep Metric Learning
    if magnet_training:
        print('Dataloading process for Magnet Loss training')
        train_sampler = SubSeqSamp(range(len(trainset)), range(args.batch_size))
        trainloader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=False, num_workers=1,
            sampler=train_sampler, pin_memory=True, drop_last=True
        )
        testloader = DataLoader(
            testset, batch_size=args.test_batch, shuffle=False, num_workers=1, 
            pin_memory=True, drop_last=False
        )
    # Random sampling
    else:
        print('Dataloading process for standard training')
        trainloader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4,
            pin_memory=True, drop_last=True
        )
        testloader = DataLoader(
            testset, batch_size=args.test_batch, shuffle=False, num_workers=4,
            pin_memory=True, drop_last=False
        )

    # Compute attack-related params
    distrib_params = {
        'minima'    : (0.0 - mean)/ std,
        'maxima'    : (1.0 - mean)/ std,
        'mean'      : mean,
        'std'       : std
    }
    n_classes = DATASET_CLASSES[args.dataset]
    # The labels
    target_name = 'labels' if args.dataset == 'svhn' else 'targets'
    train_labels = torch.tensor(getattr(trainset, target_name))
    test_labels = torch.tensor(getattr(testset, target_name))

    return (trainloader, testloader, trainset, testset, train_labels, 
        test_labels, distrib_params, n_classes)
