"""
This code uses the explicit Domain-Agnostic Fourier Neural Operator (eDAFNO) on the hyperelasticity problem described
in the paper "Domain Agnostic Fourier Neural Operators"
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

import matplotlib.pyplot as plt
from utilities import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os
import sys
from itertools import chain


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class eDAFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, nlayers):
        super(eDAFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.inp_size = 2
        self.nlayers = nlayers

        self.fc0 = nn.Linear(self.inp_size, self.width)

        for i in range(self.nlayers):
            self.add_module('conv%d' % i, SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.add_module('w%d' % i, nn.Conv2d(self.width, self.width, 1))

        self.fc1 = torch.nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, chi):
        batchsize = chi.shape[0]
        size_x = size_y = chi.shape[1]

        grid = self.get_grid(batchsize, size_x, size_y, chi.device)

        chi = chi.permute(0, 3, 1, 2)
        chi = chi.expand(batchsize, self.width, size_x, size_y)
        x = grid

        # Lifting layer:
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.nlayers):
            conv_chi = self._modules['conv%d' % i](chi)
            conv_chix = self._modules['conv%d' % i](chi * x)
            xconv_chi = x * conv_chi
            wx = self._modules['w%d' % i](x)
            x = chi * (conv_chix - xconv_chi + wx)
            if i < self.nlayers - 1: x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)

        # Projection Layer:
        x = self.fc1(x)

        return x

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(-0.5, 0.5, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(-0.5, 0.5, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def smooth_chi(mask, dist, smooth_coef):
    return torch.mul(torch.tanh(dist * smooth_coef), (mask - 0.5)) + 0.5


if __name__ == '__main__':
    # define hyperparameters
    lrs = [5.0e-2]
    gammas = [0.5]
    wds = [1e-5]
    smooth_coefs = [10.]
    smooth_coef = smooth_coefs[0]

    # experiments to be replicated with different seeds
    seeds = [0]

    ################################################################
    #                       configs
    ################################################################
    Ntotal = 2000
    ntrain = 1000
    ntest = 200

    batch_size = 20

    epochs = 501
    scheduler_step = 100
    tol_early_stop = 500

    modes = 12
    width = 32
    nlayers = 4

    ################################################################
    # load data and data normalization
    ################################################################
    t1 = default_timer()

    sub = 1
    Sx = int(((41 - 1) / sub) + 1)
    Sy = Sx

    mask = np.load('./data/elas/Random_UnitCell_mask_10_interp.npy')
    mask = torch.tensor(mask, dtype=torch.float).permute(2, 0, 1)
    dist_in = np.load('./data/elas/Random_UnitCell_dist_10_interp.npy')
    dist_in = torch.tensor(dist_in, dtype=torch.float).permute(2, 0, 1)
    input = smooth_chi(mask, dist_in, smooth_coef)
    output = np.load('./data/elas/Random_UnitCell_sigma_10_interp.npy')
    output = torch.tensor(output, dtype=torch.float).permute(2, 0, 1)

    mask_train = mask[:Ntotal][:ntrain, ::sub, ::sub][:, :Sx, :Sy]
    mask_test = mask[:Ntotal][-ntest:, ::sub, ::sub][:, :Sx, :Sy]

    mask_train = mask_train.reshape(ntrain, Sx, Sy, 1)
    mask_test = mask_test.reshape(ntest, Sx, Sy, 1)

    chi_train = input[:Ntotal][:ntrain, ::sub, ::sub][:, :Sx, :Sy]
    chi_test = input[:Ntotal][-ntest:, ::sub, ::sub][:, :Sx, :Sy]

    chi_train = chi_train.reshape(ntrain, Sx, Sy, 1)
    chi_test = chi_test.reshape(ntest, Sx, Sy, 1)

    y_train = output[:Ntotal][:ntrain, ::sub, ::sub][:, :Sx, :Sy]
    y_test = output[:Ntotal][-ntest:, ::sub, ::sub][:, :Sx, :Sy]

    # Normalizers
    y_normalizer = GaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    y_train = y_train.reshape(ntrain, Sx, Sy, 1)
    y_test = y_test.reshape(ntest, Sx, Sy, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_train, chi_train, y_train),
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_test, chi_test, y_test),
                                              batch_size=batch_size,
                                              shuffle=False)

    op_type = 'elas'
    res_dir = './res_eDAFNO_%s' % op_type
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    f = open("%s/n%d.txt" % (res_dir, ntrain), "w")
    f.write(f'ntrain, seed, learning_rate, scheduler_gamma, weight_decay, smooth_coef, '
            f'best_train_loss, best_valid_loss, best_epoch\n')
    f.close()

    t2 = default_timer()
    print(f'>> Preprocessing finished, time used: {(t2 - t1):.2f}s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

    if torch.cuda.is_available():
        y_normalizer.cuda()
    else:
        y_normalizer.cpu()

    for learning_rate in lrs:
        for scheduler_gamma in gammas:
            for wd in wds:
                for isd in seeds:
                    torch.manual_seed(isd)
                    np.random.seed(isd)

                    print(f'>> random seed: {isd}')

                    base_dir = './eDAFNO_2d_%s/n%d_lr%e_gamma%e_wd%e_seed%d' % (op_type, ntrain, learning_rate,
                                                                                scheduler_gamma, wd, isd)
                    if not os.path.exists(base_dir):
                        os.makedirs(base_dir)

                    ################################################################
                    #                      train and eval
                    ################################################################
                    myloss = LpLoss(size_average=False)
                    print("-" * 100)
                    model = eDAFNO2d(modes, modes, width, nlayers).to(device)
                    print(f'>> Total number of model parameters: {count_params(model)}')

                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
                    model_filename = '%s/model_depth4.ckpt' % base_dir

                    ttrain, ttest = [], []
                    best_train_loss = best_test_loss = 1e8
                    best_epoch = 0
                    early_stop = 0
                    for ep in range(epochs):
                        t1 = default_timer()
                        optimizer = scheduler(optimizer,
                                              LR_schedule(learning_rate, ep, scheduler_step, scheduler_gamma))
                        model.train()
                        train_l2 = 0
                        for mm, xx, yy in train_loader:
                            mm, xx, yy = mm.to(device), xx.to(device), yy.to(device)

                            optimizer.zero_grad()
                            out = model(xx)
                            out = y_normalizer.decode(out) * mm
                            yy = y_normalizer.decode(yy) * mm

                            loss = myloss(out, yy)
                            train_l2 += loss.item()

                            loss.backward()
                            optimizer.step()

                        train_l2 /= ntrain
                        ttrain.append([ep, train_l2])

                        model.eval()
                        test_l2 = 0
                        with torch.no_grad():
                            for mm, xx, yy in test_loader:
                                mm, xx, yy = mm.to(device), xx.to(device), yy.to(device)

                                out = model(xx)
                                out = y_normalizer.decode(out) * mm
                                yy *= mm

                                test_l2 += myloss(out, yy).item()

                        test_l2 /= ntest
                        ttest.append([ep, test_l2])
                        if test_l2 < best_test_loss:
                            early_stop = 0
                            best_train_loss = train_l2
                            best_test_loss = test_l2
                            best_epoch = ep
                            torch.save(model.state_dict(), model_filename)
                            t2 = default_timer()
                            print(f'>> s: {smooth_coef}, '
                                  f'epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                                  f'runtime: {(t2 - t1):.2f}s, train error: {train_l2:.5f}, '
                                  f'validation error: {test_l2:.5f}')
                        else:
                            early_stop += 1
                            t2 = default_timer()
                            print(f'>> s: {smooth_coef}, '
                                  f'epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}](best:{best_epoch + 1}), '
                                  f'runtime: {(t2 - t1):.2f}s, train error: {train_l2:.5f} (best: '
                                  f'{best_train_loss:.5f}/{best_test_loss:.5f})')

                        if early_stop > tol_early_stop: break

                    with open('%s/loss_train.txt' % base_dir, 'w') as file:
                        np.savetxt(file, ttrain)
                    with open('%s/loss_test.txt' % base_dir, 'w') as file:
                        np.savetxt(file, ttest)

                    print("-" * 100)
                    print("-" * 100)
                    print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, weight decay: {wd}')
                    print(f'>> Best train error: {best_train_loss: .5f}')
                    print(f'>> Best validation error: {best_test_loss: .5f}')
                    print(f'>> Best epochs: {best_epoch}')
                    print("-" * 100)
                    print("-" * 100)

                    f = open("%s/n%d.txt" % (res_dir, ntrain), "a")
                    f.write(f'{ntrain}, {isd}, {learning_rate}, {scheduler_gamma}, {wd}, {smooth_coef}, '
                            f'{best_train_loss}, {best_test_loss}, {best_epoch}\n')
                    f.close()

    print(f'********** Training completed! **********')
