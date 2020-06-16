import os
import shutil
import os.path as osp

import torch

def print_to_log(text, txt_file_path):
    with open(txt_file_path, 'a') as text_file:
        print(text, file=text_file)


def print_training_params(args, txt_file_path):
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    # Print to log and console
    print_to_log(text, txt_file_path)


def update_log(optimizer, epoch, train_loss, train_acc, test_loss, test_acc,
        log_path):
    lr = get_lr(optimizer)
    print_to_log(
        f'{epoch+1}\t {lr:1.0E}\t {train_loss:5.4f} \t '
        f'{train_acc:4.3f}\t\t {test_loss:5.4f}\t {test_acc:4.3f}\t\t',
        log_path
    )

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(model_info, checkpoint, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    torch.save(model_info, osp.join(checkpoint + filename))


def copy_best_checkpoint(checkpoint, filename='checkpoint.pth'):
    filename = checkpoint + filename
    shutil.copyfile(filename, osp.join(checkpoint, 'model_best.pth'))


def report_epoch_and_save(checkpoint, epoch, model, train_acc, test_acc, 
        train_loss, test_loss, magnet_data, save_all):
    text_for_print = f'>> Epoch {epoch} finished. ' \
        f'Train: top-1 acc: {train_acc:4.3f} -- loss: {train_loss:4.3f} ' \
        f'Test: top-1 acc: {test_acc:4.3f} -- loss: {test_loss:4.3f} '
    print(text_for_print)
    # dict for saving
    model_info = {
        'state_dict' 		: model.state_dict(),
        'test_acc' 			: test_acc,
        'epoch' 			: epoch
    }
    if magnet_data is not None:
        magnet_data_stuff = {
            'cluster_centers'	: magnet_data['cluster_centers'], 
            'cluster_classes' 	: magnet_data['cluster_classes'],
            'variance'			: magnet_data['variance'],
            'normalize_probs'   : magnet_data['normalize_probs'],
            'K'                 : magnet_data['K'],
            'L'                 : magnet_data['L']
        }
        model_info.update(magnet_data_stuff)

    save_checkpoint(model_info=model_info, checkpoint=checkpoint)
    if save_all:
        save_checkpoint(model_info=model_info, checkpoint=checkpoint,
            filename=f'checkpoint_{epoch}.pth')


def check_best_model(best_acc, test_acc, best_pgd_acc, pgd_acc, threshold=80):
    if best_acc > threshold and test_acc > threshold:
        # if both are above threshold, compare in terms of estimated pgd acc
        is_best = pgd_acc > best_pgd_acc
    else:
        # in any other case, compare in terms of acc
        is_best = test_acc > best_acc
    
    if is_best:
        best_acc = test_acc
        best_pgd_acc = pgd_acc
        text_for_print = \
            f'>>>> Better model achieved. Test-set acc: {best_acc:4.3f} ' \
            f'-- (estimated) PGD acc: {best_pgd_acc:4.3f}'
        print(text_for_print)

    return best_acc, best_pgd_acc, is_best
