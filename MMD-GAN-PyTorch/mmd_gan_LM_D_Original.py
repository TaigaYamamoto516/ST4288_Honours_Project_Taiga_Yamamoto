#!/usr/bin/env python
# encoding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
import os
import timeit

from PIL import Image
from torch.optim import Adagrad, Adam, RMSprop, SGD, NAdam
from Optimizer.Adagrad import Adagrad_IKSA_Min, Adagrad_IKSA_Max
from Optimizer.Adam import Adam_IKSA_Min, Adam_IKSA_Max
from Optimizer.RMSprop import RMSprop_IKSA_Min, RMSprop_IKSA_Max
from Optimizer.SGD import SGD_IKSA_Min, SGD_IKSA_Max
from Optimizer.NAdam import NAdam_IKSA_Min, NAdam_IKSA_Max

import util
import numpy as np

import base_module
from mmd import mix_rbf_mmd2


# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder, decoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        f_enc_X = self.encoder(input)
        f_dec_X = self.decoder(f_enc_X)

        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X


class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output


# Get argument
parser = argparse.ArgumentParser()
parser = util.get_args(parser)
args = parser.parse_args()
print(args)

if args.experiment is None:
    args.experiment = 'samples'
os.system('mkdir {0}'.format(args.experiment))

args.manual_seed = 2024
np.random.seed(seed=args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed(args.manual_seed)
cudnn.benchmark = True

# Get data
trn_dataset = util.get_data(args, train_flag=True)
print(trn_dataset)
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))

# construct encoder/decoder modules
hidden_dim = args.nz
G_decoder = base_module.Decoder(args.image_size, args.nc, k=args.nz, ngf=64)
D_encoder = base_module.Encoder(args.image_size, args.nc, k=hidden_dim, ndf=64)
D_decoder = base_module.Decoder(args.image_size, args.nc, k=hidden_dim, ngf=64)

netG = NetG(G_decoder)
netD = NetD(D_encoder, D_decoder)
one_sided = ONE_SIDED()
print("netG:", netG)
print("netD:", netD)
print("oneSide:", one_sided)

netG.apply(base_module.weights_init)
netD.apply(base_module.weights_init)
one_sided.apply(base_module.weights_init)

if torch.cuda.is_available():
    args.cuda = True
    print("Using GPU device", torch.cuda.current_device())
else:
    raise EnvironmentError("GPU device not available!")

# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]

# put variable into cuda device
noise_image = torch.randn(10000, args.nz, 1, 1, device='cuda', requires_grad=False)
fixed_noise = torch.randn(64, args.nz, 1, 1, device='cuda', requires_grad=False)
one = torch.tensor(1, dtype=torch.float, device='cuda')
mone = -one

if args.cuda:
    netG.cuda()
    netD.cuda()
    one_sided.cuda()
fixed_noise.requires_grad = False

# setup optimizer
if args.optimizer == 'Adagrad':
  if args.model == 'Original':
    optimizerG = Adagrad(netG.parameters(), lr=args.glr)
    optimizerD = Adagrad(netD.parameters(), lr=args.dlr)
  else:
    optimizerG = Adagrad(netG.parameters(), lr=args.glr)
    optimizerD = Adagrad_IKSA_Max(netD.parameters(), lr=args.dlr, function=args.LM_f)
elif args.optimizer == 'Adam':
  if args.model == 'Original':
    optimizerG = Adam(netG.parameters(), lr=args.glr, betas=(0.1, 0.99))
    optimizerD = Adam(netD.parameters(), lr=args.dlr, betas=(0.1, 0.99))
  else:
    optimizerG = Adam(netG.parameters(), lr=args.glr, betas=(0.1, 0.99))
    optimizerD = Adam_IKSA_Max(netD.parameters(), lr=args.dlr, function=args.LM_f)
elif args.optimizer == 'NAdam':
  if args.model == 'Original':
    optimizerG = NAdam(netG.parameters(), lr=args.glr)
    optimizerD = NAdam(netD.parameters(), lr=args.dlr)
  else:
    optimizerG = NAdam(netG.parameters(), lr=args.glr, betas=(0.1, 0.99))
    optimizerD = NAdam_IKSA_Max(netD.parameters(), lr=args.dlr, function=args.LM_f)
elif args.optimizer == 'RMSprop':
  if args.model == 'Original':
    optimizerG = RMSprop(netG.parameters(), lr=args.glr, momentum=args.momentum)
    optimizerD = RMSprop(netD.parameters(), lr=args.dlr, momentum=args.momentum)
  else:
    optimizerG = RMSprop(netG.parameters(), lr=args.glr, momentum=args.momentum)
    optimizerD = RMSprop_IKSA_Max(netD.parameters(), lr=args.dlr, momentum=args.momentum, function=args.LM_f)
else:
  if args.model == 'Original':
    optimizerG = SGD(netG.parameters(), lr=args.glr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizerD = SGD(netD.parameters(), lr=args.dlr, momentum=args.momentum, weight_decay=args.weight_decay)
  else:
    optimizerG = SGD(netG.parameters(), lr=args.glr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizerD = SGD_IKSA_Max(netD.parameters(), lr=args.dlr, momentum=args.momentum, weight_decay=args.weight_decay, function=args.LM_f)

lambda_MMD = 1.0
lambda_AE_X = 8.0
lambda_AE_Y = 8.0
lambda_rg = 16.0
c_G_25 = []
c_list_D = []
average_loss_list_G = []
average_errG_list = []
time_list_25 = []
iterations_list = []
LM_c_D = -1
LM_c_G = 100
RL_G = 0
RE_G = 0
gen_iterations = 0
running_loss_G = 0.0
running_errG = 0.0
time = timeit.default_timer()
time_list = []
g_max = args.g_max_iteration

for t in range(args.max_iter):
  data_iter = iter(trn_loader)
  i = 0
  while i < len(trn_loader) and gen_iterations < g_max:
      LM_c_D = -1
      
      # ---------------------------
      #        Optimize over NetD
      # ---------------------------
      for p in netD.parameters():
          p.requires_grad = True

      if gen_iterations < 25 or gen_iterations % 500 == 0:
          Diters = 100
          Giters = 1
      else:
          Diters = 5
          Giters = 1

      for j in range(Diters):
          if i == len(trn_loader):
              break

          x_cpu, _ = next(data_iter)
          i += 1
          netD.zero_grad()

          x = x_cpu.to('cuda')
        
          batch_size = x.size(0)

          f_enc_X_D, f_dec_X_D = netD(x)

          noise = torch.randn(batch_size, args.nz, 1, 1, device='cuda')
          
          with torch.no_grad():
            noise = noise
          
          y = netG(noise).detach()

          f_enc_Y_D, f_dec_Y_D = netD(y)

          # compute biased MMD2 and use ReLU to prevent negative value
          mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, sigma_list)
          mmd2_D = F.relu(mmd2_D)
          
          # compute rank hinge loss
          #print('f_enc_X_D:', f_enc_X_D.size())
          #print('f_enc_Y_D:', f_enc_Y_D.size())
          one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))

          # compute L2-loss of AE
          L2_AE_X_D = util.match(x.view(batch_size, -1), f_dec_X_D, 'L2')
          L2_AE_Y_D = util.match(y.view(batch_size, -1), f_dec_Y_D, 'L2')
          
          errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D
          #errD_list.append(errD.item())
          #torch.sqrt(mmd2_D):torch.size([])
          #lambda_rg:float
          #one_side_errD:float
          #lambda_AE_X:float
          #L2_AE_X_D:float
          #lambda_AE_Y:float
          #L2_AE_Y_D:torch.size([])
          errD.backward(mone)

          if args.model == 'IKSA': 
            if gen_iterations > 5:
              if errD.item()  > LM_c_D:
                LM_c_D = errD.item()
            c_list_D.append(LM_c_D)
            optimizerD.step(c=LM_c_D, running_loss=errD)
          else:
            optimizerD.step()
          

      # ---------------------------
      #        Optimize over NetG
      # ---------------------------
      for p in netD.parameters():
          p.requires_grad = False

      for j in range(Giters):
          if i == len(trn_loader):
              break
          
          x_cpu, _ = next(data_iter)
          i += 1
          netG.zero_grad()

          x = x_cpu.to('cuda')
          batch_size = x.size(0)

          f_enc_X, f_dec_X = netD(x)

          noise = torch.cuda.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1)
          noise = Variable(noise)
          y = netG(noise)

          f_enc_Y, f_dec_Y = netD(y)

          # compute biased MMD2 and use ReLU to prevent negative value
          mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, sigma_list)
          mmd2_G = F.relu(mmd2_G)
          #MMD2_G_list.append(mmd2_G.item())

          # compute rank hinge loss
          one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))
        
          errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG
          #errG_list.append(errG.item())

          errG.backward(one)
          
          optimizerG.step()

          running_loss_G += mmd2_G.item()
          running_errG += errG.item()
          RL_G += 1
          RE_G += 1
          gen_iterations += 1

          if gen_iterations % 25 == 0:
            average_errG_list.append(running_errG / RE_G)
            running_errG = 0.0
            RE_G = 0
            average_loss_list_G.append(running_loss_G / RL_G)
            running_loss_G = 0.0
            RL_G = 0
            time_25 = (timeit.default_timer() - time) / 60
            time_list_25.append(time_25)
            c_G_25.append(LM_c_G)
            iterations_list.append(gen_iterations)


      run_time = (timeit.default_timer() - time) / 60.0
      print('[%3d/%3d][%3d/%3d] [%5d] (%.2f m) min_D %.6f MMD2_D %.6f hinge %.6f L2_AE_X %.6f L2_AE_Y %.6f loss_D %.6f Loss_G %.6f f_X %.6f f_Y %.6f |gD| %.4f |gG| %.4f'
            % (t, args.max_iter, i, len(trn_loader), gen_iterations, run_time, LM_c_D,
                mmd2_D.data.item(), one_side_errD.data.item(),
                L2_AE_X_D.data.item(), L2_AE_Y_D.data.item(),
                errD.data.item(), errG.data.item(),
                f_enc_X_D.mean().data.item(), f_enc_Y_D.mean().data.item(),
                base_module.grad_norm(netD), base_module.grad_norm(netG)))
                #[0] to .item()

      if gen_iterations % 500 == 0:
          y_fixed = netG(fixed_noise)
          y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
          f_dec_X_D = f_dec_X_D.view(f_dec_X_D.size(0), args.nc, args.image_size, args.image_size)
          f_dec_X_D.data = f_dec_X_D.data.mul(0.5).add(0.5)
          vutils.save_image(y_fixed.data, '{0}/fake_samples_{1}.png'.format(args.experiment, gen_iterations))
          vutils.save_image(f_dec_X_D.data, '{0}/decode_samples_{1}.png'.format(args.experiment, gen_iterations))

  if t % 50 == 0:
      torch.save(netG.state_dict(), '{0}/netG_iter_{1}.pth'.format(args.experiment, t))
      torch.save(netD.state_dict(), '{0}/netD_iter_{1}.pth'.format(args.experiment, t))

  run_time = (timeit.default_timer() - time) / 60.0
  time_list.append(run_time)

data = [average_errG_list, average_loss_list_G, time_list_25, iterations_list]
df = pd.DataFrame(data, index =['Ave_errG', 'Ave_MMD2_G', 'time', 'Generator Iterations'])
df.to_csv(f'{args.data_file}.csv', na_rep='NA')

y_image = netG(noise_image)
y_image.data = y_image.data.mul(0.5).add(0.5)
generated_images = y_image.cpu()
folder_path = f'Generarted_png/{args.dataset}/{args.optimizer}/{args.model}_D_Original_g={gen_iterations}'
os.makedirs(folder_path, exist_ok=True)

for i, image in enumerate(generated_images):
  # The `save_image` utility expects tensor data in the range [0, 1], so we normalize the image if necessary
  save_image(image, os.path.join(folder_path, f'image_{i}.png'), normalize=True)