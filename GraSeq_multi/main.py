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

def write_record(message):

    fw = open('outputs/{}.txt'.format(localtime), 'a')
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
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--output_size_graph', type=int, default=64)
    parser.add_argument('--num_labels', type=int, help='set num of labels')


    ''' Options '''
    parser.add_argument('--graph', type=bool, default=True)
    parser.add_argument('--sequence', type=bool, default=True)
    parser.add_argument('--recons', type=bool, default=True)
    parser.add_argument('--fusion', type=bool, default=False)

    parser.add_argument('--attn', type=bool, default=False)

    parser.add_argument('--dataset', type=str, default='tox21')
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

    fpr, tpr, _  = roc_curve(label, pred, pos_label=1)
    # print(fpr, tpr)
    auc_score = auc(fpr, tpr)
    # print('auc1', auc_score)

    fpr2, tpr2, _  = roc_curve(label, pred, pos_label=0)
    auc_score2 = auc(fpr2, tpr2)
    # print('auc2', auc_score2)
    return auc_score


def eval_epoch(args, model, test_graphs, labels):

    y_scores = []
    with torch.no_grad():
        for graph_index in tqdm(test_graphs, desc=f'Epoch', ascii=True, leave=False):

            pred = model.test(graph_index)

            # y_true.append(batch.y.view(pred.shape))
            y_scores.append(pred)

        y_true = labels
        y_scores = torch.cat(y_scores, dim = 0).numpy()

        roc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

        return sum(roc_list)/len(roc_list) #y_true.shape[1]


def main(options, d_name):

    ''' set parameters '''
    args = get_args()
    write_record('{}'.format(str(args)))

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

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print('current running on the device: {} with loading type: {}'.format(args.device, args.load_long))
    write_record('current running on the device: {} with loading type: {}'.format(args.device, args.load_long))

    print('data: {} graph: {} seq: {} attn: {} recons: {} fusion: {}'.format(args.dataset, args.graph, args.sequence, args.attn, args.recons, args.fusion))
    write_record('data: {} graph: {} seq: {} attn: {} recons: {} fusion: {}'.format(args.dataset, args.graph, args.sequence, args.attn, args.recons, args.fusion))

    if args.unsup_loss == 'margin':
        num_neg = 6
    elif args.unsup_loss == 'normal':
        num_neg = 100

    if args.load_long:
        args.train_data, args.train_labels, args.test_data, args.test_labels = load_data_long(args.dataset, args.device)
    else:
        args.train_data, args.train_labels, args.test_data, args.test_labels = load_data(args.dataset, args.device)

    args.input_size_graph = args.train_data['features'][0].size(1)
    args.num_labels = args.train_labels.shape[1]

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

        auc_score = eval_epoch(args, model, test_graphs, args.test_labels)

        multiclass_metrics.append([auc_score])
        best_auc = sorted(multiclass_metrics, key=lambda x: x[0], reverse=True)[0][0]

        print('-------------' * 5)
        print('epoch: {} loss: {:.4f} auc:{:.4f} ## best_auc:{:.4f}'.format(epoch, losses, auc_score, best_auc))
        write_record('epoch: {} loss: {:.4f} auc:{:.4f} ## best_auc:{:.4f}'.format(epoch, losses, auc_score, best_auc))


        '''

        test_graphs = np.arange(len(args.test_data['adj_lists']))
        np.random.shuffle(test_graphs)

        outs = torch.zeros(args.test_labels.shape[0], args.num_labels)
        for graph_index in tqdm(test_graphs, desc=f'Epoch {epoch}', ascii=True, leave=False):
            out = model.test(graph_index)
            outs[graph_index, :] = out

        test_pred = F.softmax(outs, dim=1)
        test_pred_label = torch.max(test_pred, 1)[1]

        test_pred = test_pred.cpu().detach().numpy()
        test_pred_label = test_pred_label.cpu().detach().numpy()

        pre, rec, maf1, mif1 = evlauation(test_pred_label, args.test_labels)
        auc_score = evlauation_auc(test_pred_label, args.test_labels)

        multiclass_metrics.append([auc_score, pre, rec, maf1, mif1])
        best_auc = sorted(multiclass_metrics, key=lambda x: x[0], reverse=True)[0][0]

        print('-------------' * 5)
        print('epoch: {} loss: {:.4f} prec.:{:.4f} rec.:{:.4f} ma-f1:{:.4f} acc:{:.4f} auc:{:.4f} ## best_auc:{:.4f}'.format(epoch, losses, pre, rec, maf1, mif1, auc_score, best_auc))
        write_record('epoch: {} loss: {:.4f} prec.:{:.4f} rec.:{:.4f} ma-f1:{:.4f} acc:{:.4f} auc:{:.4f} ## best_auc:{:.4f}'.format(epoch, losses, pre, rec, maf1, mif1, auc_score, best_auc))

        '''

if __name__ == '__main__':
    # args.graph = options[0]
    # args.sequence = options[1]
    # args.fusion = options[2]
    # args.recons = options[3]
    # args.load_long = option[4]

    d_name = "tox21"
    option_list = [[True, False, False, False, False], [False, True, False, False, False], [False, True, False, True, False], [True, True, False, False, False]]
    # option_list = [[True, True, False, True, False], [True, True, True, False, False], [True, True, True, True, False]]
    # option_list = [[True, False, False, False, True], [False, True, False, False, True], [False, True, False, True, True], [True, True, False, False, True]]
    # option_list = [[True, True, False, True, True], [True, True, True, False, True], [True, True, True, True, True]]
    for op in option_list:
        main(op, d_name)
