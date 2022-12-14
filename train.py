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
import torchvision.transforms as transforms
import matplotlib.pyplot as mplot
import os

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 28,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
).cuda()



trainer = Trainer(
    diffusion,
    './datasets_folder/mnist/mnist_train/0',
    train_batch_size = 16,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)


sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)

trainer.train()

mnist_train = torchvision.datasets.MNIST('./datasets_folder/MNIST_data', train=True, download=True)

for i, (img, label) in enumerate(mnist_train):
    if label == 0:
        img_path = "./datasets_folder/mnist/mnist_train_denoise"+"/"+str(label)
        img = Trainer.model.forward(img, -1, None)