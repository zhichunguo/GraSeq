import os
import sys
import torch
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from rdkit.Chem import AllChem
from rdkit import Chem

from joint_models import *
torch.backends.cudnn.enabled=False

localtime = time.asctime(time.localtime(time.time()))
localtime = '-'.join(localtime.split())

from load_data import load_data, load_data_long

def write_record(args, message):

    fw = open('outputs/data_{}_graph_{}_seq_{}_fu_{}_recons_{}_{}.txt'.format(args.dataset, args.graph, args.sequence, args.fusion, args.recons, localtime), 'a')
    fw.write('{}\n'.format(message))
    fw.close()

def get_args():
    parser = argparse.ArgumentParser(description='pytorch version of GraSeq')

    ''' Graph Settings '''
    parser.add_argument('--agg_func', type=str, default='MEAN')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--b_sz', type=int, default=20)
    parser.add_argument('--seed', type=int, default=824)
    parser.add_argument('--cuda', type=bool, help='use CUDA')
    parser.add_argument('--gcn', action='store_true')
    parser.add_argument('--unsup_loss', type=str, default='margin')
    parser.add_argument('--max_vali_f1', type=float, default=0)
    parser.add_argument('--name', type=str, default='debug')

    ''' Sequence Settings '''
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--warmup', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output_size_graph', type=int, default=64)


    ''' Options '''
    parser.add_argument('--graph', type=bool, default=True)
    parser.add_argument('--sequence', type=bool, default=True)
    parser.add_argument('--recons', type=bool, default=True)
    parser.add_argument('--fusion', type=bool, default=False)

    parser.add_argument('--attn', type=bool, default=False)

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--load_long', type=bool, default=True)

    args = parser.parse_args()
    return args

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc


def evlauation(pred, label):

    macro_precision = precision_score(label, pred)
    macro_recall = recall_score(label, pred)
    macro_f1 = f1_score(label, pred, average='macro')
    micro_f1 = f1_score(label, pred, average='micro')

    return macro_precision, macro_recall, macro_f1, micro_f1


def evlauation_auc(pred, label):

    fpr, tpr, _  = roc_curve(label, pred[:, 1], pos_label=1)
    auc_score = auc(fpr, tpr)

    fpr2, tpr2, _  = roc_curve(label, pred[:, 0], pos_label=0)
    auc_score2 = auc(fpr2, tpr2)
    # print('auc_score for label == 0: {}, auc_score for label == 1: {}'.format(auc_score, auc_score2))
    return auc_score, fpr, tpr


def main(options, d_name):

    ''' set parameters '''
    args = get_args()

    ''' option settings '''
    args.graph = options[0]
    args.sequence = options[1]
    args.fusion = options[2]
    args.recons = options[3]
    args.load_long = options[4]

    args.dataset = d_name

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device_id = torch.cuda.current_device()
            print('using device', device_id, torch.cuda.get_device_name(device_id))

    write_record(args, '{}'.format(str(args)))
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print('current running on the device: {} with loading type: {}'.format(args.device, args.load_long))
    write_record(args, 'current running on the device: {} with loading type: {}'.format(args.device, args.load_long))

    print('data: {} graph: {} seq: {} attn: {} recons: {} fusion: {}'.format(args.dataset, args.graph, args.sequence, args.attn, args.recons, args.fusion))
    write_record(args, 'data: {} graph: {} seq: {} attn: {} recons: {} fusion: {}'.format(args.dataset, args.graph, args.sequence, args.attn, args.recons, args.fusion))

    if args.unsup_loss == 'margin':
        num_neg = 6
    elif args.unsup_loss == 'normal':
        num_neg = 100

    if args.load_long:
        args.train_data, args.train_labels, args.test_data, args.test_labels = load_data_long(args.dataset, args.device)
    else:
        args.train_data, args.train_labels, args.test_data, args.test_labels = load_data(args.dataset, args.device)

    args.input_size_graph = args.train_data['features'][0].size(1)

    model = Model(args)
    model.to(args.device)

    multiclass_metrics = []

    for epoch in range(1, args.epochs):
        train_graphs = np.arange(len(args.train_data['adj_lists']))
        np.random.shuffle(train_graphs)

        losses = 0.0
        for graph_index in tqdm(train_graphs, desc=f'Epoch {epoch}', ascii=True, leave=False):
            loss = model.train(graph_index, epoch)
            losses += loss.item()

        test_graphs = np.arange(len(args.test_data['adj_lists']))
        np.random.shuffle(test_graphs)

        outs = torch.zeros(args.test_labels.shape[0], 2)
        for graph_index in tqdm(test_graphs, desc=f'Epoch {epoch}', ascii=True, leave=False):
            out = model.test(graph_index)
            outs[graph_index, :] = out

        test_pred = F.softmax(outs, dim=1)
        test_pred_label = torch.max(test_pred, 1)[1]

        test_pred = test_pred.cpu().detach().numpy()
        test_pred_label = test_pred_label.cpu().detach().numpy()

        pre, rec, maf1, mif1 = evlauation(test_pred_label, args.test_labels)
        auc_score, fpr, tpr = evlauation_auc(test_pred, args.test_labels)

        multiclass_metrics.append([auc_score, pre, rec, maf1, mif1])
        best_auc = sorted(multiclass_metrics, key=lambda x: x[0], reverse=True)[0][0]

        # if auc_score == best_auc:
        #     fw = open('rocs/data_{}_graph_{}_seq_{}_fu_{}_recons_{}_{}.tsv'.format(args.dataset, args.graph, args.sequence, args.fusion, args.recons, localtime), 'w')
        #     for i, j in zip(fpr, tpr):
        #         fw.write('{}\t{}\n'.format(i, j))
        #     fw.close()

        print('-------------' * 5)
        print('epoch: {} loss: {:.4f} prec.:{:.4f} rec.:{:.4f} ma-f1:{:.4f} acc:{:.4f} auc:{:.4f} ## best_auc:{:.4f}'.format(epoch, losses, pre, rec, maf1, mif1, auc_score, best_auc))
        write_record(args, 'epoch: {} loss: {:.4f} prec.:{:.4f} rec.:{:.4f} ma-f1:{:.4f} acc:{:.4f} auc:{:.4f} ## best_auc:{:.4f}'.format(epoch, losses, pre, rec, maf1, mif1, auc_score, best_auc))


if __name__ == '__main__':
    # args.graph = options[0]
    # args.sequence = options[1]
    # args.fusion = options[2]
    # args.recons = options[3]
    # args.load_long = option[4]

    d_name = "bbbp"
    # option_list = [[True, False, False, False, False], [False, True, False, False, False], [False, True, False, True, False], [True, True, False, False, False]]
    option_list = [[True, True, False, True, False], [True, True, True, False, False], [True, True, True, True, False]]
    # option_list = [[True, False, False, False, True], [False, True, False, False, True], [False, True, False, True, True], [True, True, False, False, True]]
    # option_list = [[True, True, False, True, True], [True, True, True, False, True], [True, True, True, True, True]]

    for op in option_list:
        main(op, d_name)
