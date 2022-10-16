from __future__ import print_function
from itertools import islice
import math
from random import random
from random import seed
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from skimage import io

from skimage import img_as_ubyte
import torchvision.transforms as transforms
import matplotlib.pyplot as mplot
import os

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, GaussianDiffusion_enhance

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4)
).cuda()

diffusion = GaussianDiffusion_enhance(
    model,
    image_size = 28,
    timesteps = 1000,           # number of steps
    enhancesteps = 60,
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
).cuda()



trainer = Trainer(
    diffusion,
    './datasets_folder/mnist/train_diffusion',
    train_batch_size = 16,
    train_lr = 8e-5,
    train_num_steps = 100000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()

trainer.save("Enhance")
denoise_batch = 8

transform = transforms.Compose([
        transforms.ToTensor()
    ])
mnist_train = torchvision.datasets.MNIST('./datasets_folder/MNIST_data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size = denoise_batch, shuffle = False, num_workers=2)

mnist_test = torchvision.datasets.MNIST('./datasets_folder/MNIST_data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size = denoise_batch, shuffle = False, num_workers=2)


for batch_idx, (data, target) in enumerate(train_loader):
    img = trainer.model.denoise(data, batch = denoise_batch)
    for i in range(denoise_batch):
        img_path = "./datasets_folder/mnist/mnist_train_d"+"/"+str(target[i].item())
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        io.imsave(img_path+"/" + str(batch_idx * denoise_batch + i) + ".jpg", img_as_ubyte(img[i].cpu()[0]))

for batch_idx, (data, target) in enumerate(test_loader):
    img = trainer.model.denoise(data, batch = denoise_batch)
    for i in range(denoise_batch):
        img_path = "./datasets_folder/mnist/mnist_test_d"+"/"+str(target[i].item())
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        io.imsave(img_path+"/" + str(batch_idx * denoise_batch + i) + ".jpg", img_as_ubyte(img[i].cpu()[0]))

