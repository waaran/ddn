# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch
from torch.autograd import Variable, grad
from torch.nn.init import xavier_normal
from torchvision import datasets, transforms
import torchvision.utils as vutils
from models import *


def xavier_init(model):
    for param in model.parameters():
        if len(param.size()) == 2:
            xavier_normal(param)

class Trainer:
    def __init__(self):
        self.batch_size = 128
        self.z_dim = 100
        self.h_dim = 128
        self.y_dim = 784
        self.max_epochs = 1000
        self.lambda_ = 10
        self.num_gpu = 1
        begin_num = 0
    # build model with the corresponding parameters
    def build_model(self):
        self.generator = GeneratorDRAGAN(self.z_dim, self.y_dim, self.h_dim, self.num_gpu)
        self.discriminator = DiscriminatorGRAGAN(self.z_dim, self.y_dim, self.h_dim, self.num_gpu)
        # Init weight matrices (xavier_normal)
        xavier_init(self.generator)
        xavier_init(self.discriminator)

    # assign running devices for models
    def assign_device(self):
        if self.num_gpu == 1:
            self.generator.cuda()
            self.discriminator.cuda()
        elif self.num_gpu > 1:
            self.generator = torch.nn.DataParallel(self.generator.cuda(),device_ids=range(self.num_gpu))
            self.discriminator = torch.nn.DataParallel(self.discriminator.cuda(),device_ids=range(self.num_gpu))

    # assign device for Variables
    def _get_variable(self, inputs):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out

    def train(self):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=self.batch_size, shuffle=False, drop_last=True)

        # generator = torch.nn.Sequential(torch.nn.Linear(z_dim, h_dim),
        #         torch.nn.Sigmoid(),
        #         torch.nn.Linear(h_dim, y_dim),
        #         torch.nn.Sigmoid()).cuda()

        # discriminator = torch.nn.Sequential(torch.nn.Linear(y_dim, h_dim),
        #         torch.nn.Sigmoid(),
        #         torch.nn.Linear(h_dim, 1),
        #         torch.nn.Sigmoid()).cuda()
        self.build_model()
        self.assign_device()

        generator = self.generator
        discriminator = self.discriminator

        opt_g = torch.optim.Adam(generator.parameters())
        opt_d = torch.optim.Adam(discriminator.parameters())

        criterion = torch.nn.BCELoss()
        # X = Variable(torch.cuda.FloatTensor(batch_size, y_dim))
        # z = Variable(torch.cuda.FloatTensor(batch_size, z_dim))
        # labels = Variable(torch.cuda.FloatTensor(batch_size))
        X = self._get_variable(torch.FloatTensor(self.batch_size, self.y_dim))
        z = self._get_variable(torch.FloatTensor(self.batch_size, self.z_dim))
        labels = self._get_variable(torch.FloatTensor(self.batch_size))

        # Train
        for epoch in range(self.max_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                X.data.copy_(data)

                # Update discriminator
                # train with real
                discriminator.zero_grad()
                pred_real = discriminator(X)
                labels.data.fill_(1.0)
                loss_d_real = criterion(pred_real, labels)
                loss_d_real.backward()

                # train with fake
                z.data.normal_(0, 1)
                fake = generator.forward(z).detach()
                pred_fake = discriminator(fake)
                labels.data.fill_(0.0)
                loss_d_fake = criterion(pred_fake, labels)
                loss_d_fake.backward()

                # gradient penalty
                alpha = torch.rand(self.batch_size, 1).type(dtype).expand(X.size())
                x_hat = Variable(alpha * X.data + (1 - alpha) * (X.data + 0.5 * X.data.std() * torch.rand(X.size()).type(dtype)), requires_grad=True)
                pred_hat = discriminator(x_hat)
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).type(dtype),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = self.lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                gradient_penalty.backward()

                loss_d = loss_d_real + loss_d_fake + gradient_penalty
                opt_d.step()

                # Update generator
                generator.zero_grad()
                z.data.normal_(0, 1)
                gen = generator(z)
                pred_gen = discriminator(gen)
                labels.data.fill_(1.0)
                loss_g = criterion(pred_gen, labels)
                loss_g.backward()
                opt_g.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                      % (epoch, self.max_epochs, batch_idx, len(train_loader),
                         loss_d.data[0], loss_g.data[0]))

                if batch_idx % 100 == 0:
                    vutils.save_image(data,
                            'samples/real_samples.png')
                    fake = generator(z)
                    vutils.save_image(gen.data.view(self.batch_size, 1, 28, 28),
                            'samples/fake_samples_epoch_%03d.png' % epoch)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
