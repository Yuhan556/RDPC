# Task-Oriented Lossy Compression with Data, Perception, and Classification Constraints

This is the official Pytorch implementation of the proposed __rate-distortion-perception-classification (RDPC) framework__ in the paper *Task-Oriented Lossy Compression with Data, Perception, and Classification Constraints* [[paper](https://ieeexplore.ieee.org/document/10960410)] [[arXiv](https://arxiv.org/abs/2405.04144)]. 

This repository is partially based on the [PyTorch Code](https://github.com/zhanggq96/uRDP) for the paper *Universal Rate-Distortion-Perception Representations for Lossy Compression* by [Zhang et al. (2021)](https://arxiv.org/abs/2106.10311).



<p align="center">
  <img src="/Figs/Network-s.pdf" width="700" />
</p>

## Introduction

In this paper, we integrate traditional reconstruction tasks and generative tasks into the [information bottleneck](https://arxiv.org/abs/physics/0004057) (IB) principle and investigate both the theoretical and practical roles of the multi-task IB framework. 

Specifically, we derived the *closed-form expressions* for both __rate-distortion-classification (RDC)__ and __rate-perception-classification (RPC)__ functions in binary and scalar Gaussian cases. We discussed the operational meaning of RDPC with common randomness and proved the convexity of the RDC function.  We also investigate the decisive role of the *source noise* towards the existence of tradeoffs in both lossy compression and signal restoration problems.

<p align="center">
  <img src="/Figs/RDC-PC-theory.pdf" width="700" />
</p>

We conduct a series of experiments by implementing this DL-based image compression framework incorporating multiple tasks. The experimental outcomes validate our theoretical results on the RDC, RPC tradeoffs, as well as RPC with certain levels of distortion.



## Requirements

The codes are compatible with the following packages:

- numpy==1.21.6
- tensorflow==2.11.0
- torch==1.8.0+cu111
- torchsummary==1.5.1
- torchvision==0.9.0+cu111

## Usage

### Pre-trained classifiers

We have provided pre-trained classifiers for MNIST and SVHN datasets under ```experiments``` folder. If you want to retrain the classifiers, please un-comment ```lines 185-211``` in ```train.py```, and then save the corresponding checkpoints named as `q-mnist.ckpt` or `q-svhn.ckpt` under ```experiments``` folder.

### Set Hyperparameters

The overall loss function is
$$
\mathcal L=\lambda_d \mathbb{E}(||X-\hat{X}||^2) + \lambda_p W_1(p_X,p_{\hat{X}})+\lambda_c \text{CE}(s,\mathbf{\hat{s}}),
$$
where $\lambda_d$, $\lambda_p$, and $\lambda_c$ are hyperparameters to control weights of distortion, perception, and cross-entropy losses, respectively. It should be specified by the variables ```Lambda_distortion```, ```Lambda_perception```, and ```Lambda_classification``` in the ```run_with_params.py```. You may refer the following choices of the hyperparameter to reproduce the results in the paper. 

#### MNIST dataset

For the RDC tradeoff shown in Fig. 9(a), we set

```python
mode = 'D'
Lambda_distortion = [0.50000, 0.80000,  1.00000, 1.20000]
Lambda_perception = [0]
Lambda_classification = [0.0025, 0.0050, 0.00600, 0.00800, 0.01000, 0.01100, 0.01300, 0.01500]
```

For the RPC tradeoff shown in Fig. 9(b), we set

```python
mode = 'P'
Lambda_distortion = [0]
Lambda_perception = [0.0025, 0.0040, 0.0050, 0.0100, 0.0130, 0.015]
Lambda_classification = [0.0050, 0.00600, 0.00800, 0.01000, 0.01100, 0.01300, 0.01500]
```

For the RDC given P and RPC given D shown in Fig. 10(a) and Fig. 11(a) respectively, we set

```python
mode = 'P'
Lambda_distortion = [1]
Lambda_perception = [0, 0.0025, 0.0040, 0.0050, 0.0100, 0.0130, 0.015]
Lambda_classification = [0, 0.0025, 0.0050, 0.00600, 0.00800, 0.01000, 0.01100, 0.01300, 0.01500]
```

Rates are controlled by $R = \text{dim} \times \log_2(L)$ with $(\text{dim}, L)$ pairs specified by `latent_dim_1` and `L_1` in ```run_with_parameters.py``` to be $(3, 3), (3, 4), (4, 4), (5, 4)$.

Use the following code block (`Line 102-104`) in ```run_with_params.py``` for experiments on MNIST dataset:

```python
settings.append({'dataset': 'mnist', 'latent_dim_1': latent_dim_1, 'L_1': L_1,
                  'Lambda_d': Lambda_d, 'Lambda_c': Lambda_c, 'Lambda_p': Lambda_p, 'n_critic': 1, 'n_epochs': 30, 'progress_intervals': 6,
                  'enc_layer_scale': 1, 'initialize_mse_model': 0, 'test_batch_size': 1000})
```

#### SVHN dataset

For the RDC given P and RPC given D shown in Fig. 10(b) and Fig. 11(b) respectively, we set

```python
mode = 'P'
Lambda_distortion = [1]
Lambda_perception = [0, 0.00025, 0.00075, 0.00125, 0.00200]
Lambda_classification = [0, 0.00025, 0.00050, 0.00075, 0.00100, 0.00125, 0.00150, 0.00200]
```

Rates are controlled by $R = \text{dim} \times \log_2(L)$ with $(\text{dim}, L)$ pairs specified by `latent_dim_1` and `L_1` in ```run_with_parameters.py``` to be $(10, 8), (15, 8), (20, 8)$.

Use the following code block (`Line 106-111`) in ```run_with_params.py``` for experiments on MNIST dataset:

```python
settings.append({'dataset': 'svhn', 'latent_dim_1': latent_dim_1, 'L_1': L_1,
                 'Lambda_d': Lambda_d, 'Lambda_c': Lambda_c, 'Lambda_p': Lambda_p, 'n_critic': 1, 'n_epochs': 80, 'progress_intervals': 10,
                 'enc_layer_scale': 1, 'initialize_mse_model': 0, 'test_batch_size': 1000,
                 'lr_encoder': 1e-4, 'lr_decoder': 1e-4, 'lr_critic': 1e-4,
                 'beta1_encoder': 0.5, 'beta1_decoder': 0.5, 'beta1_critic': 0.5,
                 'beta2_encoder': 0.999, 'beta2_decoder': 0.999, 'beta2_critic': 0.999})
```

### Run Experiments

After setting the hyperparamenters properly, you can use the following command line to run experiements

```shell
python run_with_params.py
```

Nested folders will be created for each choice of $(\lambda_d, \lambda_c)$ or $(\lambda_p,\lambda_c)$ under `experiments` folder. The results for all epochs will be stored in the `__losses.csv` file in each sub-folder.



## Citation

```tex
@ARTICLE{RDPC_JSAC2025,
  author={Wang, Yuhan and Wu, Youlong and Ma, Shuai and Zhang, Ying-Jun Angela},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Task-Oriented Lossy Compression with Data, Perception, and Classification Constraints}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JSAC.2025.3559164}}
```
