import os
import random
import numpy as np
import os.path as osp
from tqdm import tqdm
# Torch-related imports
import torch
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
# From magnet_loss
from utils.magnet_loss import MagnetLoss
from utils.attacks import eval_robustness, final_attack_eval
from utils.magnet_training import magnet_epoch_wrapper
from utils.setups import magnet_assertions, get_batch_builders, get_magnet_data
from utils.logging import (report_epoch_and_save, print_to_log, update_log,
    print_training_params, check_best_model, copy_best_checkpoint)
from utils.utils import eval_model, compute_real_losses, copy_pretrained_model
from utils.train_settings import parse_settings
from datasets.load_dataset import load_dataset

# For deterministic behavior
cudnn.deterministic = True
cudnn.benchmark = False

args, LOG_PATH, LOG_HEADERS, BEST_MODEL_THRESH, MODEL_INIT, ALPHA_STEP, \
    HARDCODED_EPS, STANDARD_EPSILONS = parse_settings(magnet_training=True)

def train_epoch(model, epoch, optimizer, trainloader, device, trainset, 
        train_labels, testset, test_labels, batch_builder, print_freq, 
        cluster_refresh_interval, criterion, magnet_data, distrib_params, 
        hardcoded_eps, minibatch_replays, actual_trades):
    
    model, batch_builder = magnet_epoch_wrapper(
        model, optimizer, trainloader, device, trainset, train_labels, 
        batch_builder, print_freq, cluster_refresh_interval, criterion, 
        eps=hardcoded_eps, magnet_data=magnet_data, 
        distrib_params=distrib_params, minibatch_replays=minibatch_replays, 
        actual_trades=actual_trades
    )
    # Extract centroid and centroid classes and update magnet_data dict
    magnet_data.update({
        'cluster_classes' : batch_builder.cluster_classes.to(device),
        'cluster_centers' : batch_builder.centroids.to(device)
    })
    # Trainset: compute loss and variance. Testset: compute loss
    train_loss, train_variance = compute_real_losses(model, criterion, 
        trainset, train_labels, magnet_data, compute_variance=True)
    magnet_data.update({'variance' : train_variance})
    test_loss, _ = compute_real_losses(model, criterion, 
        testset, test_labels, magnet_data, compute_variance=False)
    # Eval model in train and test sets
    train_acc, _ = eval_model(model=model, dataset=trainset, 
        data_labels=train_labels, magnet_data=magnet_data)
    test_acc, _ = eval_model(model=model, dataset=testset, 
        data_labels=test_labels, magnet_data=magnet_data)

    return (model, batch_builder, magnet_data, train_acc, test_acc, train_loss,
        test_loss)


def main():
    # Decide device to use
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Log path: verify existence of checkpoint dir, or create it
    if not osp.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    print_to_log('\t '.join(LOG_HEADERS), LOG_PATH)
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    # txt file with all params
    print_training_params(args, osp.join(args.checkpoint, 'params.txt'))
    # Get datasets and dataloaders
    trainloader, testloader, trainset, testset, train_labels, test_labels, \
        distrib_params, num_classes = load_dataset(args, magnet_training=True)
    # Params for each epoch
    epoch_steps = len(trainloader)
    print_freq = int(np.ceil(0.01 * epoch_steps))
    cluster_refresh_interval = epoch_steps - 1
    # Create model
    model = MODEL_INIT(num_classes=num_classes).to(device)
    print(model)
    if args.pretrained_path is not None:
        model = copy_pretrained_model(model, args.pretrained_path)
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones)
    # Define criterion
    criterion = MagnetLoss(args.alpha)
    # Get batch builders and magnet_data dict
    magnet_assertions(train_labels, args.k, L=args.L, m=args.m)
    # Get batch builders
    batch_builder = get_batch_builders(model, trainset, train_labels, 
        args.k, args.m, args.d, device, dataset_name=args.dataset)
    # Get magnet_data dict
    magnet_data = get_magnet_data(batch_builder, device, args, model, 
        criterion, trainset, train_labels)

    if args.evaluate_ckpt is not None:
        final_attack_eval(
            model, testloader, testset, test_labels, checkpoint=args.checkpoint, 
            distrib_params=distrib_params, device=device, 
            standard_epsilons=STANDARD_EPSILONS, 
            evaluate_ckpt=args.evaluate_ckpt, alpha_step=ALPHA_STEP, L=args.L, 
            seed=args.seed, normalize_probs=not args.not_normalize,
            restarts=args.restarts, attack_iters=args.iterations, evaluate_only=True
        )
        return

    best_acc = -np.inf
    # Iterate through epochs
    for epoch in range(args.epochs):
        model, batch_builder, magnet_data, train_acc, test_acc, train_loss, \
            test_loss = train_epoch(
                model, epoch, optimizer, trainloader, device, trainset, 
                train_labels, testset, test_labels, batch_builder, print_freq, 
                cluster_refresh_interval, criterion, magnet_data, 
                distrib_params, hardcoded_eps=HARDCODED_EPS, 
                minibatch_replays=args.minibatch_replays,
                actual_trades=args.actual_trades
        )
        best_acc = max(best_acc, test_acc)
        # Report epoch and save current model
        report_epoch_and_save(args.checkpoint, epoch, model, train_acc, 
            test_acc, train_loss, test_loss, magnet_data, args.save_all)
        # Update log with results
        update_log(optimizer, epoch, train_loss, train_acc, test_loss, test_acc,
            LOG_PATH)
        # Update scheduler
        scheduler.step()

    # Report best accuracy of all training
    print(f'Best accuracy: {best_acc:4.3f}')
    # Run attack on best model
    final_attack_eval(
        model, testloader, testset, test_labels, checkpoint=args.checkpoint, 
        distrib_params=distrib_params, device=device, 
        standard_epsilons=STANDARD_EPSILONS, alpha_step=ALPHA_STEP, L=args.L, 
        seed=args.seed, normalize_probs=not args.not_normalize,
        restarts=args.restarts, attack_iters=args.iterations,
        evaluate_ckpt=os.path.join(args.checkpoint, 'checkpoint.pth')
    )

if __name__ == '__main__':
    main()
