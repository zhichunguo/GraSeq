import copy
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
import models

import pickle
import random
import numpy as np

import time
from tqdm import tqdm

class LINEAR_LOGSOFTMAX(nn.Module):

    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction = nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


class VAE(nn.Module):

    def __init__(self, args):
        super(VAE,self).__init__()

        self.device = args.device
        self.latent_size = args.latent_size
        self.hidden_size = args.hidden_size

        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.warmup = args.warmup

        self.encoder = models.encoder(self.input_dim, self.hidden_size, self.latent_size, self.device)

        self.decoder = models.decoder(self.latent_size, self.input_dim, self.device)

        self.reconstruction_criterion = nn.MSELoss(reduction='mean').to(self.device)
        self.reparameterize_with_noise = False

        self.reconstruction = args.recons

    def reparameterize(self, mu, logvar):

        if self.reparameterize_with_noise:

            sigma = torch.exp(logvar)
            eps = torch.FloatTensor(logvar.size()[0],1).normal_(0,1).to(self.device)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps

        else:
            return mu

    def map_label(self, label, classes):

        mapped_label = torch.LongTensor(label.size()).to(self.device)

        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i

        return mapped_label

    def train_step(self, smile):

        molecule_emb, mu, logvar = self.encoder(smile)

        z = self.reparameterize(mu, logvar)

        if self.reconstruction:

            soss = nn.Embedding(1, self.input_dim).to(self.device)
            sos = soss(torch.LongTensor([0]).to(self.device))
            re_smile = self.decoder(z, sos, smile).to(self.device)

            # Reconstruction Loss
            reconstruction_loss = self.reconstruction_criterion(smile, re_smile)

            # KL-Divergence
            KLD = (0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
            f2 = 1.0 * (self.current_epoch - 0)
            f2 = f2 * (1.0 * self.warmup)
            beta =  torch.FloatTensor([min(max(f2, 0), self.warmup)]).to(self.device)
            recons_loss =  0.1 * self.batch_size * (reconstruction_loss - beta * KLD)

        else:
            recons_loss = torch.tensor([0])

        return recons_loss, molecule_emb

    def forward(self, smile, epoch):

        self.train()
        self.current_epoch = epoch

        loss, molecule_emb = self.train_step(smile)
        return loss, molecule_emb

    def test_vae(self, smile):

        self.eval()

        molecule_emb, mu, logvar = self.encoder(smile)

        return molecule_emb


