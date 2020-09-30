import copy
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data

import pickle
import random
import numpy as np


import time
from tqdm import tqdm

import graph_models
import seq_models
import models


class Model(nn.Module):
    def __init__(self, args):
        super(Model,self).__init__()

        self.device = args.device
        self.layers = args.num_layers
        self.input_size_graph = args.input_size_graph
        self.output_size_graph = args.output_size_graph
        self.train_data = args.train_data
        self.test_data = args.test_data
        self.train_labels = args.train_labels
        self.test_labels = args.test_labels
        self.latent_size = args.latent_size
        self.hidden_size = args.hidden_size
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.warmup = args.warmup
        self.num_labels = args.num_labels

        self.graph = args.graph
        self.sequence = args.sequence
        self.recons = args.recons
        self.use_attn = args.attn
        self.use_fusion = args.fusion

        self.graph_pretrain = graph_models.GraphSage(self.layers,
                                                     self.input_size_graph,
                                                     self.output_size_graph,
                                                     device=self.device,
                                                     gcn="True",
                                                     agg_func="MEAN")

        self.VAE = seq_models.VAE(args)

        self.AtomEmbedding = nn.Embedding(self.input_size_graph,
                                          self.hidden_size).to(self.device)
        self.AtomEmbedding.weight.requires_grad = True

        self.output_layer = models.classifier(self.latent_size, self.num_labels, self.device)

        self.label_criterion = nn.BCEWithLogitsLoss(reduction = "none")

        if self.use_attn:
            self.attention = models.SelfAttention(self.hidden_size)

        self.optimizer  = optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=1e-8,
                                     amsgrad=True)

        for name, para in self.named_parameters():
            if para.requires_grad:
                print(name, para.data.shape)

    def train(self, graph_index, epoch):

        nodes_emb = self.AtomEmbedding(self.train_data['sequence'][graph_index])

        if self.graph:
            nodes_emb = self.graph_pretrain(graph_index, self.train_data)
            graph_emb = nodes_emb

        if self.sequence:
            recons_loss, nodes_emb = self.VAE(nodes_emb, epoch)
            seq_emb = nodes_emb

        if self.use_fusion:
            molecule_emb = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)
        else:
            molecule_emb = torch.mean(nodes_emb, dim=0, keepdim=True)

        pred = self.output_layer(molecule_emb)
        label = torch.tensor(self.train_labels[graph_index]).to(self.device)

        self.optimizer.zero_grad()

        label = label.view(pred.shape)

        #Whether y is non-null or not.
        is_valid = label ** 2 > 0
        loss_mat = self.label_criterion(pred, (label + 1) / 2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        label_loss = torch.sum(loss_mat) / torch.sum(is_valid)


        if self.recons:
            loss = label_loss + recons_loss
        else:
            loss = label_loss

        loss.backward()
        self.optimizer.step()

        return loss

    def test(self, graph_index):

        nodes_emb = self.AtomEmbedding(self.test_data['sequence'][graph_index])

        if self.graph:
            nodes_emb = self.graph_pretrain(graph_index, self.test_data)
            graph_emb = nodes_emb

        if self.sequence:
            nodes_emb = self.VAE.test_vae(nodes_emb)
            seq_emb = nodes_emb

        if self.use_fusion:
            molecule_emb = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)
        else:
            molecule_emb = torch.mean(nodes_emb, dim=0, keepdim=True)

        pred = self.output_layer(molecule_emb)

        return pred