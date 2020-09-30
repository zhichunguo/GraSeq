import sys, os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):

    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        self.fc1 = nn.Linear(emb_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                # initialize all bias as zeros
                nn.init.constant_(param, 0.0)

    def forward(self, embeds):
        graph_embs = torch.mean(embeds, 0)
        x = F.elu(self.fc1(graph_embs))
        x = F.elu(self.fc2(x))
        logists = torch.log_softmax(x, 0)
        return logists

class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size


        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.

        nodes	 -- list of nodes
        """
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined

class GraphSage(nn.Module):
    """docstring for GraphSage"""
    def __init__(self, num_layers, input_size, out_size, device, raw_features_all=None, adj_lists_all=None, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func

        # self.raw_features_all = raw_features_all
        # self.adj_lists_all = adj_lists_all
        self.raw_features = None
        self.adj_lists = None

        for index in range(1, num_layers+1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

    def forward(self, graph_index, data_set):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings
        """
        self.raw_features = data_set['features'][graph_index]
        self.adj_lists = data_set['adj_lists'][graph_index]
        nodes_batch = list()
        for key in self.adj_lists:
            if len(self.adj_lists[key]) > 0:
                nodes_batch.append(key)
        lower_layer_nodes = nodes_batch
        nodes_batch_layers = [(lower_layer_nodes,)]
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index-1]
            # self.dc.logger.info('aggregate_feats.')
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            new_aggregate_features = torch.zeros(pre_hidden_embs.shape[0], aggregate_feats.shape[1])
            record = 0
            # print(nb)
            # print(pre_neighs)
            # print(pre_hidden_embs)

            for i in range(pre_hidden_embs.shape[0]):
                if i not in nb:
                    new_aggregate_features[i,:] = pre_hidden_embs[i,:]
                else:
                    # print(new_aggregate_features[i,:])
                    # print("######")
                    # print(aggregate_feats[record,:])
                    new_aggregate_features[i,:] = aggregate_feats[record,:]
                    record += 1

            # print(new_aggregate_features)

            sage_layer = getattr(self, 'sage_layer' + str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)

            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs,
                                        aggregate_feats=new_aggregate_features)

            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _list = list
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        # print(nodes)
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_list(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        # print(len(samp_neighs))
        sam = []
        for i, samp_neigh in enumerate(samp_neighs):
            sam.append(samp_neigh + [nodes[i]])
        # print(sam)
        samp_neighs = sam
        _unique_nodes_list = list(set().union(*samp_neighs))
        # print(_unique_nodes_list)
        # raise TypeError
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]
        # self.dc.logger.info('2')
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # self.dc.logger.info('3')
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # self.dc.logger.info('4')

        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh).to(embed_matrix.device)
            # print(embed_matrix)
            aggregate_feats = mask.mm(embed_matrix)

        elif self.agg_func == 'MAX':
            # print(mask)
            indexs = [x.nonzero() for x in mask==1]
            aggregate_feats = []
            # self.dc.logger.info('5')
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        # self.dc.logger.info('6')

        return aggregate_feats
