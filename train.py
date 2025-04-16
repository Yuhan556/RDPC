import argparse
import os
import sys
import math
import json
from shutil import copyfile
from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from models import *
from utils import *


def is_progress_interval(args, epoch):
    return epoch == args.n_epochs-1 or (args.progress_intervals > 0 and epoch % args.progress_intervals == 0)

def _lr_factor(epoch, dataset, mode=None):
    if dataset == 'mnist':
        if epoch < 20:
            return 1
        elif epoch < 40:
            return 1/5
        else:
            return 1/50
    elif dataset == 'fashion_mnist':
        if epoch < 20:
            return 1
        elif epoch < 35:
            return 1/5
        else:
            return 1/50
    elif dataset == 'svhn':
        if epoch < 25:
            return 1
        else:
            return 1/5
    else:
        return 1

def compute_lambda_anneal(Lambda, epoch, Lambda_init=0.0005, end_epoch=12):
    assert Lambda == 0 and epoch >= 0
    e = min(epoch, end_epoch)

    return Lambda_init*(end_epoch-e)/end_epoch

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Source: https://github.com/andreaferretti/wgan/blob/master/train.py
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Evaluate accuarcy of the classification task, where the predict is the probability vector for 10 classes, and the label is a 0-9 number indicating the class.
def evaluate_accuracy(label, y_recon):
    _, predict = torch.max(y_recon, dim=1)
    correct = 0
    for i in range(len(label)):
        if label[i] == predict[i]:
            correct += 1
    acc = correct / len(label)
    print(acc)
    return acc


def train_base(args, device):
    experiment_path = args.experiment_path
    classi_path = 'experiments'
    N = 60000

    assert (args.L_1 > 0 or not args.quantize) and args.latent_dim_1 > 0
    assert not (args.L_1 > 0 and not args.quantize), f'Quantization disabled, yet args.L_1={args.L_1}'

    # Loss weight for gradient penalty
    lambda_gp = args.Lambda_gp

    # Initialize decoder and discriminator
    encoder1 = Encoder1(args).to(device)
    decoder1 = Decoder1(args).to(device)
    discriminator1 = Discriminator(args).to(device)
    netQ = NetQ(args).to(device)

    ## load pre-trained model for MNIST or SVHN dataset
    if args.dataset == 'mnist':
        classifier_file = f'{classi_path}/q-mnist.ckpt'
    else:
        classifier_file = f'{classi_path}/q-svhn.ckpt'
    netQ.load_state_dict(torch.load(classifier_file))

    alpha1 = encoder1.alpha

    # if args.initialize_mse_model:
    #     # Load pretrained models to continue from if directory is provided
    #     if args.Lambda_base > 0:
    #         assert isinstance(args.load_mse_model_path, str)

    #         # Check args match
    #         with open(os.path.join(args.load_mse_model_path, '_settings.json'), 'r') as f:
    #             mse_model_args = json.load(f)
    #             assert_args_match(mse_model_args, vars(args), ('L_1', 'latent_dim_1', 'limits', 'enc_layer_scale'))
    #             assert mse_model_args['Lambda_base'] == 0
    #             # No need to assert args match for "stochastic" and "quantize"?

    #     if isinstance(args.load_mse_model_path, str):
    #         assert args.Lambda_base > 0, args.load_mse_model_path
    #         encoder1.load_state_dict(torch.load(os.path.join(args.load_mse_model_path, 'encoder1.ckpt')))
    #         decoder1.load_state_dict(torch.load(os.path.join(args.load_mse_model_path, 'decoder1.ckpt')))
    #         discriminator1.load_state_dict(torch.load(os.path.join(args.load_mse_model_path, 'discriminator1.ckpt')))

    # Configure data loader
    train_dataloader, test_dataloader, unnormalizer = \
        load_dataset(args.dataset, args.batch_size, args.test_batch_size, shuffle_train=True)
    test_set_size = len(test_dataloader.dataset)

    # Optimizers
    optimizer_E1 = torch.optim.Adam(encoder1.parameters(), lr=args.lr_encoder, betas=(args.beta1_encoder, args.beta2_encoder))
    optimizer_G1 = torch.optim.Adam(decoder1.parameters(), lr=args.lr_decoder, betas=(args.beta1_decoder, args.beta2_decoder))
    optimizer_Q1 = torch.optim.Adam(netQ.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=args.lr_critic, betas=(args.beta1_critic, args.beta2_critic))

    lr_factor = lambda epoch: _lr_factor(epoch, args.dataset)

    scheduler_E1 = LambdaLR(optimizer_E1, lr_factor)
    scheduler_G1 = LambdaLR(optimizer_G1, lr_factor)
    scheduler_D1 = LambdaLR(optimizer_D1, lr_factor)
    scheduler_Q1 = LambdaLR(optimizer_Q1, lr_factor)

    criterion = nn.MSELoss()

    # ----------
    #  Prep
    # ----------
    # copyfile('models.py', 'experiments/models.txt')

    os.makedirs(f"{experiment_path}", exist_ok=True)
    with open(f'{experiment_path}/_settings.json', 'w') as f:
        json.dump(vars(args), f)

    with open(f'{experiment_path}/_losses.csv', 'w') as f:
        f.write('epoch,distortion_loss,perception_loss, classification_loss, acc\n')

    # ----------
    #  Training
    # ----------

    batches_done = 0
    n_cycles = 1 + args.n_critic
    disc_loss = torch.Tensor([-1])
    distortion_loss = torch.Tensor([-1])
    classi_loss = torch.Tensor([-1])
    saved_original_test_image = False

    # ----------------------
    # Train a classifier (Q)
    # ----------------------
    print('Start training Q')

    ''' ###------ Un-comment this part if you want to re-train the classifier Q ------###
    class_criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        for i, (x,y) in enumerate(train_dataloader):
            x = x.to(device)
            # print(x.shape)
            x = x.view(x.shape[0], -1)
            # print(x.shape)
            #convert labels into one hot of dim 10
            y_dim10 = to_categorical(y,num_classes=10)
            y_dim10 = torch.from_numpy(y_dim10)

            y_recon = netQ(x)

            class_loss = class_criterion(y_recon, y_dim10)

            class_loss.backward()
            optimizer_Q1.step()
            optimizer_Q1.zero_grad()
            if batches_done % 100 == 0:
                with torch.no_grad(): # use most recent results
                    print('Recon:{0}, y:{1}'.format(y_recon, y_dim10))
                    acc = evaluate_accuracy(y, y_recon)
                print('Epoch:{0}, Batches_done:{1}, Acc:{2}, classi_loss:{3}'.format(epoch, batches_done, acc, class_loss))
            batches_done += 1
    classifier_file = f'{classi_path}/q-mnist.ckpt'
    torch.save(netQ.state_dict(), classifier_file)
    '''

    print('Start training GAN')
    batches_done = 0
    for epoch in range(args.n_epochs):
        Lambda_distort = args.Lambda_d
        # if Lambda_distort == 0:
        #     # Give an early edge to training discriminator for Lambda = 0
        #     Lambda_distort = compute_lambda_anneal(Lambda, epoch)

        Lambda_classi = args.Lambda_c
        Lambda_percep = args.Lambda_p

        for i, (x, y) in enumerate(train_dataloader):
            # Configure input
            x = x.to(device)
            #convert labels into one hot of dim 10
            y_dim10 = to_categorical(y, num_classes=10)
            y_dim10 = torch.from_numpy(y_dim10).to(device).long()
            y = y.to(device)

            if i % n_cycles == 1:

                # ---------------------
                #  Train Discriminator
                # ---------------------

                free_params(discriminator1)
                frozen_params(encoder1)
                frozen_params(decoder1)
                frozen_params(netQ)

                optimizer_D1.zero_grad()

                # Noise batch_size x latent_dim
                u1 = uniform_noise([x.size(0), args.latent_dim_1], alpha1).to(device)
                z1 = encoder1(x, u1)
                x_recon = decoder1(z1, u1)
                # Real images and labels
                real_validity = discriminator1(x)
                # Fake images and labels
                fake_validity = discriminator1(x_recon)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator1, x.data, x_recon.data)
                # Adversarial loss
                disc_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                disc_loss.backward()

                optimizer_D1.step()

            else: # if i % n_cycles == 0:

                # -----------------
                #  Train Generator
                # -----------------
                frozen_params(discriminator1)
                frozen_params(netQ)
                free_params(encoder1)
                free_params(decoder1)

                optimizer_E1.zero_grad()
                optimizer_G1.zero_grad()
               
                u1 = uniform_noise([x.size(0), args.latent_dim_1], alpha1).to(device)
                z1 = encoder1(x, u1)
                x_recon = decoder1(z1, u1)
                x_flat = x_recon.view(x_recon.shape[0], -1)
                if args.dataset == 'mnist':
                    y_1 = netQ(x_flat)
                else:
                    y_1 = netQ(x_recon)

                fake_validity = discriminator1(x_recon)

                perception_loss = -torch.mean(fake_validity)
                distortion_loss = criterion(x, x_recon)
                # classi_loss = F.cross_entropy(y_1, y_dim10).div(math.log(2))
                classi_loss = F.cross_entropy(y_1, y).div(math.log(2))

                loss = Lambda_distort*distortion_loss + Lambda_percep*perception_loss  + Lambda_classi*classi_loss

                loss.backward()

                optimizer_G1.step()
                optimizer_E1.step()
                # optimizer_Q1.step()

            if batches_done % 100 == 0:
                with torch.no_grad(): # use most recent results
                    real_validity = discriminator1(x)
                    perception_loss = -torch.mean(fake_validity) + torch.mean(real_validity)
                print(
                    "[Epoch %d/%d] [Batch %d/%d (batches_done: %d)] [Disc loss: %f] [Perception loss: %f] [Distortion loss: %f] [Classification loss: %f]"
                    % (epoch, args.n_epochs, i, len(train_dataloader), batches_done, disc_loss.item(),
                    perception_loss.item(), distortion_loss.item(), classi_loss.item())
                )

            batches_done += 1

        # ---------------------
        # Evaluate losses on test set
        # ---------------------
        with torch.no_grad():
            encoder1.eval()
            decoder1.eval()
            discriminator1.eval()
 
            distortion_loss_avg, perception_loss_avg, class_loss_avg, accuracy = 0, 0, 0, 0

            for j, (x_test, y) in enumerate(test_dataloader):
                #convert labels into one hot of dim 10
                y_dim10 = to_categorical(y, num_classes=10)
                y_dim10 = torch.from_numpy(y_dim10).to(device)
                y = y.to(device)
                x_test = x_test.to(device)

                u1_test = uniform_noise([x_test.size(0), args.latent_dim_1], alpha1).to(device)
                z1 = encoder1(x_test, u1_test)
                x_test_recon = decoder1(z1, u1_test)    
                distortion_loss, perception_loss, classi_loss, y_recon = evaluate_losses(args.dataset, x_test, x_test_recon, discriminator1, netQ, y)
                #####
                distortion_loss_avg += x_test.size(0) * distortion_loss
                perception_loss_avg += x_test.size(0) * perception_loss
                class_loss_avg += x_test.size(0) * classi_loss
                accuracy += x_test.size(0) * evaluate_accuracy(y, y_recon)

                if j == 0 and is_progress_interval(args, epoch):
                    save_image(unnormalizer(x_test_recon.data[:120]), f"{experiment_path}/{epoch}_recon.png", nrow=10, normalize=True)
                    if not saved_original_test_image:
                        save_image(unnormalizer(x_test.data[:120]), f"{experiment_path}/{epoch}_real.png", nrow=10, normalize=True)
                        saved_original_test_image = True

            distortion_loss_avg /= test_set_size
            perception_loss_avg /= test_set_size
            class_loss_avg /= test_set_size
            accuracy /= test_set_size

            with open(f'{experiment_path}/_losses.csv', 'a') as f:
                f.write(f'{epoch},{distortion_loss_avg},{perception_loss_avg},{class_loss_avg},{accuracy}\n')

            encoder1.train()
            decoder1.train()
            discriminator1.train()

        scheduler_E1.step()
        scheduler_D1.step()
        scheduler_G1.step()

    # ---------------------
    #  Save
    # ---------------------

    encoder1_file = f'{experiment_path}/encoder1.ckpt'
    decoder1_file = f'{experiment_path}/decoder1.ckpt'
    discriminator1_file = f'{experiment_path}/discriminator1.ckpt'

    torch.save(encoder1.state_dict(), encoder1_file)
    torch.save(decoder1.state_dict(), decoder1_file)
    torch.save(discriminator1.state_dict(), discriminator1_file)



if __name__ == '__main__':
    os.makedirs("experiments", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=25, help="number of epochs of training")
    parser.add_argument("--n_channel", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument('--quantize', type=int, default=1, help='DOES NOTHING RIGHT NOW')
    parser.add_argument('--stochastic', type=int, default=1, help='add noise below quantization threshold (default: True)')
    parser.add_argument("--latent_dim_1", type=int, default=8, help="dimensionality of the latent space")
    parser.add_argument("--latent_dim_2", type=int, default=-1, help="dimensionality of the latent space for refinement model")
    parser.add_argument("--latent_dim_0", type=int, default=-1, help="dimensionality of the latent space for reduced model")
    parser.add_argument("--latent_dim_M1", type=int, default=-1, help="dimensionality of the latent space for joint_reduced_reduced model")
    parser.add_argument('--L_1', type=int, default=-1, help='number of quantization levels for base model (default: -1)')
    parser.add_argument('--L_2', type=int, default=-1, help='number of quantization levels for refined model (default: -1)')
    parser.add_argument('--L_0', type=int, default=-1, help='number of quantization levels for reduced model (default: -1)')
    parser.add_argument('--L_M1', type=int, default=-1, help='number of quantization levels for joint_reduced_reduced model (default: -1)')
    parser.add_argument('--limits', nargs=2, type=float, default=[-1,1], help='quanitzation limits (default: (-1,1))')
    parser.add_argument("--Lambda_d", type=float, default=1.0, help="coefficient for distortion loss (default: 1.0)")
    parser.add_argument("--Lambda_c", type=float, default=0.0, help="coefficient for distortion loss (default: 1.0)")
    parser.add_argument("--Lambda_p", type=float, default=0.0, help="coefficient for distortion loss (default: 1.0)")
    parser.add_argument("--Lambda_gp", type=float, default=10.0, help="coefficient for gradient penalty")
    #### Insertion of paras
    parser.add_argument("--lr_encoder", type=float, default=1e-2, help="encoder learning rate")
    parser.add_argument("--lr_decoder", type=float, default=1e-2, help="decoder learning rate")
    parser.add_argument("--lr_critic", type=float, default=2e-4, help="critic learning rate")
    parser.add_argument("--lr_classifier", type=float, default=1e-2, help="classifier learning rate")
    parser.add_argument("--beta1_encoder", type=float, default=0.5, help="encoder beta 1")
    parser.add_argument("--beta1_decoder", type=float, default=0.5, help="decoder beta 1")
    parser.add_argument("--beta1_classifier", type=float, default=0.5, help="classifier beta 1")
    parser.add_argument("--beta1_critic", type=float, default=0.5, help="critic beta 1")
    parser.add_argument("--beta2_encoder", type=float, default=0.9, help="encoder beta 2")
    parser.add_argument("--beta2_decoder", type=float, default=0.9, help="decoder beta 2")
    parser.add_argument("--beta2_critic", type=float, default=0.9, help="critic beta 2")
    parser.add_argument("--beta2_classifier", type=float, default=0.9, help="classifier beta 2")
    parser.add_argument("--test_batch_size", type=int, default=5000, help="test set batch size (default: 5000)")
    parser.add_argument("--load_mse_model_path", type=str, default=None, help="directory from which to preload enc1/dec1+disc1 models to start training at")
    parser.add_argument("--load_base_model_path", type=str, default=None, help="directory from which to preload enc1/dec1+disc1 models to start training at")
    parser.add_argument("--load_reduced_model_path", type=str, default=None, help="(for joint_reduced_reduced) directory from which to preload discriminator models to start training at")
    parser.add_argument("--initialize_base_discriminator", type=int, default=0, help="For refined or reduced models: whether to start from base model disc.")
    parser.add_argument("--initialize_mse_model", type=int, default=0, help="For base model: whether or not to continue training from Lambda=0 model.")
    parser.add_argument("--enc_layer_scale", type=float, default=1.0, help="Scale layer size of encoder by factor")
    parser.add_argument("--enc_2_layer_scale", type=float, default=1.0, help="Scale layer size of encoder 2 by factor")
    parser.add_argument("--reduced_dims", type=str, default='', help="Reduced dims")
    parser.add_argument("--reduced_path", type=str, help="If joint reduced training, where to save the secondary model")
    parser.add_argument("--refined_path", type=str, help="If joint refined training, where to save the secondary model")
    parser.add_argument("--dataset", type=str, default='mnist', help="dataset to use (default: mnist)")
    parser.add_argument("--progress_intervals", type=int, default=-1, help="periodically show progress of training")
    parser.add_argument("--entropy_intervals", type=int, default=-1, help="periodically calculate entropy of model. -1 only end, -2 for never")
    parser.add_argument("--submode", type=str, default=None, help="generic submode of mode")
    parser.add_argument("-mode", type=str, default='base', help="base, refined or reduced training mode")
    parser.add_argument("-experiment_path", type=str, help="name of the subdirectory to save")

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device]: {device}')

    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        vars(args)['input_size'] = 784
    elif args.dataset == 'svhn':
        vars(args)['input_size'] = 3*32*32
        vars(args)['n_channel'] = 3
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    train_base(args, device)

