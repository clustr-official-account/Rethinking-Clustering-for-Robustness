import torch
import argparse
import os.path as osp
# The models
import sys
sys.path.append('../models') # This is gross, I know
from models.resnet import ResNet18

# Constants
LOG_HEADERS = ['Epoch', 'LR', 'Train loss', 'Train acc.', 'Test loss', \
    'Test acc.']
BEST_MODEL_THRESHOLDS = {
    'cifar10'   : 80.0,
    'cifar100'  : 60.0,
    'svhn'      : 85.0
}
MODEL_INITS = {
    'resnet18'  : ResNet18
}
ALPHA_STEP = 2/255
HARDCODED_EPS = torch.tensor([8/255,])
STANDARD_EPSILONS = torch.tensor([2/255, 8/255, 16/255, .1])

def parse_settings(magnet_training):
    # Training settings
    dataset_choices = ['cifar10','cifar100','svhn']
    arch_choices = ['resnet18']
    parser = argparse.ArgumentParser(description='PyTorch Magnet Loss')
    parser.add_argument('--epochs', type=int, default=90,
        help='training epochs (def.: 90)')
    parser.add_argument('--batch-size', type=int, default=256,
        help='batch size for standard (regular) training')
    parser.add_argument('--test-batch', type=int, default=64,
        help='batch size for testing')
    parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate (default: 1e-3)')
    parser.add_argument('--dataset', type=str, default='cifar10',
        help='dataset to use', choices=dataset_choices)
    parser.add_argument('--checkpoint', required=True, type=str,
        help='name of experiment')
    parser.add_argument('--seed', type=int, help='manual seed', default=111)
    parser.add_argument('--arch', type=str, default='resnet18',
        help='architecture to use', choices=arch_choices)
    parser.add_argument('--evaluate-ckpt', type=str, default=None,
        help='path to ckpt to evaluate')
    # Magnet-loss params
    parser.add_argument('--k', type=int, default=2,
        help='clusters per class (def.: 2)')
    parser.add_argument('--d', type=int, default=20,
        help='samples to take per cluster (def.: 20) -- bs = m*d')
    parser.add_argument('--m', type=int, default=12,
        help='clusters to sample per iteration (def.: 12) -- bs = m*d')
    parser.add_argument('--L', type=int, default=20,
        help='nearest neighbors to consider (def.: 20)')
    parser.add_argument('--alpha', type=float, default=12.5,
        help='alpha for magnet loss (def.: 12.5)')
    # Adversarial training param
    parser.add_argument('--minibatch-replays', type=int, default=0,
        help='M parameter for free adversarial training')
    # For scheduler
    parser.add_argument('--milestones', type=int, nargs='+', default=[30, 60], 
        help='milestones for scheduler')
    # Other parameters
    parser.add_argument('--consistency-lambda', type=float, default=0.0,
        help='multiplier for combination of magnet- and consistency-loss')
    parser.add_argument('--xent-lambda', type=float, default=0.0,
        help='multiplier for combination of magnet- and crossentropy-loss')
    parser.add_argument('--mse-consistency', action='store_true', default=False,
        help='use mean-squared error consistency')
    parser.add_argument('--not-normalize', action='store_true', default=False,
        help='not normalize scores for probabilities computation')
    parser.add_argument('--actual-trades', action='store_true', default=False,
        help='run with actual trades implementation')
    parser.add_argument('--pretrained-path', type=str, default=None,
        help='path to pretrained model')
    parser.add_argument('--save-all', action='store_true', default=False,
        help='save all checkpoints (every epoch)')
    # Attack-related params
    parser.add_argument('--restarts', type=int, default=10,
        help='num of PGD restarts for evaluation')
    parser.add_argument('--iterations', type=int, default=20,
        help='num of PGD iterations for evaluation')
    args = parser.parse_args()
    # Batch size
    args.batch_size = args.m * args.d if magnet_training else args.batch_size
    print(f'Effective batch size: {args.batch_size}')
    # Add checkpoint namespace
    args.checkpoint = "experiments/%s/"%(args.checkpoint)

    LOG_PATH = osp.join(args.checkpoint, 'log.txt')
    MODEL_INIT = MODEL_INITS[args.arch]
    BEST_MODEL_THRESH = BEST_MODEL_THRESHOLDS[args.dataset]

    return (args, LOG_PATH, LOG_HEADERS, BEST_MODEL_THRESH, MODEL_INIT, 
        ALPHA_STEP, HARDCODED_EPS, STANDARD_EPSILONS)
