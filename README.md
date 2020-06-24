# ClusTR: Clustering Training for Robustness - PyTorch implementation
This repository implements "ClusTR: Clustering Training for Robustness" in PyTorch. Running this code succesfully reproduces the results in the [manuscript](https://arxiv.org/abs/2006.07682).

ClusTR is a theoretically-motivated framework for training Deep Neural Networks (DNNs) for adversarial robustness.

ClusTR is based on using a Clustering Loss, _i.e._ a loss that encourages clustering of semantically-similar instances in representation space, for inducing semantics into the training of the network. ClusTR harnesses this loss, together with standard techniques for DNN training, to train robust networks without the need of conducting training on adversaries.

As a Clustering Loss, ClusTR employs the Magnet Loss, introduced in [Metric Learning with Adaptive Density Discrimination](https://research.fb.com/wp-content/uploads/2016/05/metric-learning-with-adaptive-density-discrimination.pdf?), Rippel _et al._, ICLR 2016. Since there is no official implementation of the Magnet Loss, we borrow parts of the Magnet Loss implementation from [MagnetLossPyTorch, Vithursan Thangarasa, 2018, GitHub](https://github.com/vithursant/MagnetLoss-PyTorch).

## Installation

This repository has several dependencies. These dependencies can be easily installed by using [Anaconda](https://docs.anaconda.com/anaconda/install/) to create an environment based on the file `utils/environment.yml`. Downloading all requirements and creating an environment based on this yml file is achieved by running:

```bash
cd utils
```
And then
```bash
conda env create -f environment.yml
```
These lines create an environment called `pytorch`. To activate this environment, run:
```bash
conda activate pytorch
```

## Repository structure
This repository has three main folders, as described next:
* `datasets`: for management of the datasets. Currently supports CIFAR10, CIFAR100 and SVHN.
* `models`: holds the code for the DNN architectures. All experiments were run with [ResNet18](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), as implemented in `models/resnet.py`.
* `utils`: all utilities for training and testing.

The main file for running training/testing in this repository is `main_magnet.py`. The arguments this file takes as input are established in `utils/train_setting.py`.

## Training ClusTR+QTRADES
To train ClusTR+QTRADES, pre-trained weights are required. These pre-trained weights are in the `pretrained_weights` directory.

To train, run

```bash
python main_clustr.py --checkpoint clustr_qtrades --pretrained-path pretrained_weights/resnet18.pt --epochs 25 --consistency-lambda 8
```

This command will run training of ClusTR+QTRADES on CIFAR10 for 25 epochs, starting from the pre-trained weights at `pretrained_weights/resnet18.pt`, with a coefficient for the TRADES loss of 8, _i.e._ the <img src="https://render.githubusercontent.com/render/math?math=\lambda"> in Equation (5) in the [manuscript](https://arxiv.org/abs/2006.07682). The results will be saved at directory `clustr_qtrades`. 

When the command finishes running, the directory `clustr_qtrades` will have three files, described next.
* `checkpoint.pth`: a dictionary with the information from the model at the _last_ epoch. This dictionary contains the _state_dict_ of the model, which is found under the key `'state_dict'`. To see the _state_dict_ of the model, run `torch.load('checkpoint.pth', map_location='cpu')['state_dict']` on a python terminal.
* `log.txt`: the log of the training procedure. It contains 8 columns, namely: (1) Epoch, (2) LR, (3) Train loss, (4) Train acc., (5) Test loss, and (6) Test acc. Each row corresponds to a different epoch. For that epoch, we report: the number of the epoch, the learning rate during the epoch, the loss in the training set, the accuracy in the train set, the loss in the test set, and the accuracy in the test set.
* `params.txt`: a `txt` file with a single line reporting all the parameters with which the experiment was run, as given by the parameters needed by the parser defined in `utils/train_setting.py`.

Please refer to the script `utils/train_settings.py` or run `python main_clustr.py --help` for details of the possible arguments to pass to the `main_clustr.py` script.


## Evaluating ClusTR+QTRADES
To evaluate the checkpoint obtained from the training procedure, run

```bash
python main_clustr.py --checkpoint clustr_qtrades_pgd20 --evaluate-ckpt experiments/clustr_qtrades/checkpoint.pth --iterations 20 --restarts 10 --L 20
```

This evaluation procedure will consider the closest 20 clusters (the L parameter) and run PGD <img src="https://render.githubusercontent.com/render/math?math=\ell_\infty"> attacks (l-infinity norm bounded attacks) with 20 iterations and 10 restarts will be run for assessing robustness. When the procedure finishes there will be a `csv` file called `attack_results_ext.csv` under the `experiments/clustr_qtrades_pgd20` directory. This file has the results from the PGD attack. There are two columns: _epsilons_ and _test_set_accs_. Each row of the file shows the resulting PGD accuracy at the corresponding value of epsilon (the strength of the attack).

You can also run stronger PGD-100 attacks by running (this may take _very_ long, depending on your GPU).

```bash
python main_clustr.py --checkpoint clustr_qtrades_pgd100 --evaluate-ckpt experiments/clustr_qtrades/checkpoint.pth --iterations 100 --restarts 10 --L 20
```

## Citation
If you find our work useful, please consider citing it as
```bash
@article{alfarra2020clusttr,
    title={ClusTR: Clustering Training for Robustness},
    author={Motasem Alfarra and Juan C. Pérez and Adel Bibi and Ali Thabet and Pablo Arbeláez and Bernard Ghanem},
    year={2020},
    journal={arXiv:2006.07682}
}
```
