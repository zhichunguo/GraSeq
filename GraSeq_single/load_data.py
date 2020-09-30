import torch
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from collections import defaultdict


def load_data_long(dataset, device):
    mole_dict = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: " Ne",
                11: "Na", 12:"Mg", 13: "Al", 14:"Si", 15:"P", 16: "S", 17: "Cl", 18:"Ar", 19:"K", 20:"Ca", 22:"Ti", 24:"Cr", 26:"Fe", 28:"Ni",
                29:"Cu", 31:"Ga", 32:"Ge", 34:"Se", 35:"Br", 40:"Zr", 44:"Ru", 45:"Rh", 46:"Pd", 47:"Ag", 50:"Sn", 51:"Sb", 52:"Te", 53: "I", 65:"Tb", 75:"Re", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg",
                81:"Tl", 82:"Pb", 83:"Bi"}

    pair_list = ["Br", "Cl", "Si", "Na", "Ca", "Ge", "Cu", "Au", "Sn", "Tb", "Pt", "Re", "Ru", "Bi", "Li", "Fe", "Sb", "Hg","Pb", "Se", "Ag","Cr","Pd","Ga","Mg","Ni","Ir","Rh","Te","Ti","Al","Zr","Tl"]

    data_file = f"../original_datasets/{dataset}/{dataset}_train"
    file = open(data_file, "r")
    node_types = set()
    label_types = set()
    tr_len = 0
    for line in file:
        tr_len += 1
        smiles = line.split("\t")[1]
        label = line.split("\t")[2][:-1]
        i = 0
        s = []
        while i < len(smiles):
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                s.append(smiles[i] + smiles[i+1])
                i += 2
            else:
                s.append(smiles[i].upper())
                i += 1
        node_types |= set(s)
        label_types.add(label)
    file.close()

    te_len = 0
    data_file = f"../original_datasets/{dataset}/{dataset}_test"
    file = open(data_file, "r")
    for line in file:
        te_len += 1
        smiles = line.split("\t")[1]
        label = line.split("\t")[2][:-1]
        i = 0
        s = []
        while i < len(smiles):
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                s.append(smiles[i] + smiles[i+1])
                i += 2
            else:
                s.append(smiles[i].upper())
                i += 1
        node_types |= set(s)
        label_types.add(label)
    file.close()

    print(tr_len)
    print(te_len)

    node2index = {n: i for i, n in enumerate(node_types)}
    label2index = {l: i for i, l in enumerate(label_types)}

    print(node2index)
    print(label2index)

    data_file = f"../original_datasets/{dataset}/{dataset}_train"
    file = open(data_file, "r")
    train_adjlists = []
    train_features = []
    train_sequence = []
    train_labels = torch.zeros(tr_len)
    for line in file:
        smiles = line.split("\t")[1]
        label = line.split("\t")[2][:-1]
        mol = AllChem.MolFromSmiles(smiles)
        graph_nodes = []
        for atom in mol.GetAtoms():
            graph_nodes.append(mole_dict[atom.GetAtomicNum()])
        # print(graph_nodes)
        i = 0
        s = 0
        while i < len(smiles):
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                i += 2
            else:
                i += 1
            s += 1

        feature = torch.zeros(s, len(node_types))

        map = {}
        se_num = 0
        gr_num = 0
        i = 0
        smiles_seq = []
        while i < len(smiles):
            this_str = smiles[i]
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                this_str = smiles[i] + smiles[i+1]
                i += 2
            else:
                this_str = this_str.upper()
                i += 1
            smiles_seq.append(node2index[this_str])
            if this_str in graph_nodes and this_str == mole_dict[mol.GetAtoms()[gr_num].GetAtomicNum()]:
                map[gr_num] = se_num
                gr_num += 1
            feature[se_num, node2index[this_str]] = 1
            se_num += 1

        adj_list = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # print(i,j)
            typ = bond.GetBondType()
            adj_list[map[i]].append(map[j])
            adj_list[map[j]].append(map[i])
            if typ == Chem.rdchem.BondType.DOUBLE:
                adj_list[map[i]].append(map[j])
                adj_list[map[j]].append(map[i])
            elif typ == Chem.rdchem.BondType.TRIPLE:
                adj_list[map[i]].append(map[j])
                adj_list[map[j]].append(map[i])
                adj_list[map[i]].append(map[j])
                adj_list[map[j]].append(map[i])

        # train_labels[len(train_adjlists)]= int(label2index[label])
        train_labels[len(train_adjlists)]= int(label)
        train_adjlists.append(adj_list)
        train_features.append(torch.FloatTensor(feature).to(device))
        train_sequence.append(torch.tensor(smiles_seq))
    file.close()

    data_file = f"../original_datasets/{dataset}/{dataset}_test"
    file = open(data_file, "r")
    test_adjlists = []
    test_features = []
    test_sequence = []
    test_labels = np.zeros(te_len)
    for line in file:
        smiles = line.split("\t")[1]
        # print(smiles)
        label = line.split("\t")[2][:-1]
        mol = AllChem.MolFromSmiles(smiles)
        graph_nodes = []
        for atom in mol.GetAtoms():
            graph_nodes.append(mole_dict[atom.GetAtomicNum()])
        # print(graph_nodes)
        i = 0
        s = 0
        while i < len(smiles):
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                i += 2
            else:
                i += 1
            s += 1

        feature = torch.zeros(s, len(node_types))

        map = {}
        se_num = 0
        gr_num = 0
        i = 0
        smiles_seq = []
        while i < len(smiles):
            this_str = smiles[i]
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                this_str = smiles[i] + smiles[i+1]
                i += 2
            else:
                this_str = this_str.upper()
                i += 1
            smiles_seq.append(node2index[this_str])
            if this_str in graph_nodes and this_str == mole_dict[mol.GetAtoms()[gr_num].GetAtomicNum()]:
                map[gr_num] = se_num
                gr_num += 1
            feature[se_num, node2index[this_str]] = 1
            se_num += 1

        adj_list = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # print(i,j)
            typ = bond.GetBondType()
            adj_list[map[i]].append(map[j])
            adj_list[map[j]].append(map[i])
            if typ == Chem.rdchem.BondType.DOUBLE:
                adj_list[map[i]].append(map[j])
                adj_list[map[j]].append(map[i])
            elif typ == Chem.rdchem.BondType.TRIPLE:
                adj_list[map[i]].append(map[j])
                adj_list[map[j]].append(map[i])
                adj_list[map[i]].append(map[j])
                adj_list[map[j]].append(map[i])

        # test_labels[len(test_adjlists)] = int(label2index[label])
        test_labels[len(test_adjlists)] = int(label)
        test_adjlists.append(adj_list)
        test_features.append(torch.FloatTensor(feature).to(device))
        test_sequence.append(torch.tensor(smiles_seq))
    file.close()

    train_data = {}
    train_data['adj_lists'] = train_adjlists
    train_data['features'] = train_features
    train_data['sequence'] = train_sequence

    test_data = {}
    test_data['adj_lists'] = test_adjlists
    test_data['features'] = test_features
    test_data['sequence'] = test_sequence

    return train_data, train_labels, test_data, test_labels

def load_data(dataset, device):

    data_file = f"../original_datasets/{dataset}/{dataset}_train"
    file = open(data_file, "r")
    node_types = set()
    label_types = set()
    tr_len = 0
    for line in file:
        tr_len += 1
        smiles = line.split("\t")[1]
        s = []
        mol = AllChem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            s.append(atom.GetAtomicNum())
        node_types |= set(s)
        label = line.split("\t")[2][:-1]
        label_types.add(label)
    file.close()

    te_len = 0
    data_file = f"../original_datasets/{dataset}/{dataset}_test"
    file = open(data_file, "r")
    for line in file:
        te_len += 1
        smiles = line.split("\t")[1]
        s = []
        mol = AllChem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            s.append(atom.GetAtomicNum())
        node_types |= set(s)
        label = line.split("\t")[2][:-1]
        label_types.add(label)
    file.close()

    print(tr_len)
    print(te_len)

    node2index = {n: i for i, n in enumerate(node_types)}
    label2index = {l: i for i, l in enumerate(label_types)}

    print(node2index)
    print(label2index)

    data_file = f"../original_datasets/{dataset}/{dataset}_train"
    file = open(data_file, "r")
    train_adjlists = []
    train_features = []
    train_sequence = []
    train_labels = torch.zeros(tr_len)
    for line in file:
        smiles = line.split("\t")[1]
        label = line.split("\t")[2][:-1]
        mol = AllChem.MolFromSmiles(smiles)
        feature = torch.zeros(len(mol.GetAtoms()), len(node_types))

        l = 0
        smiles_seq = []
        for atom in mol.GetAtoms():
            feature[l, node2index[atom.GetAtomicNum()]] = 1
            smiles_seq.append(node2index[atom.GetAtomicNum()])
            l += 1
        adj_list = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            typ = bond.GetBondType()
            adj_list[i].append(j)
            adj_list[j].append(i)
            if typ == Chem.rdchem.BondType.DOUBLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
            elif typ == Chem.rdchem.BondType.TRIPLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
                adj_list[i].append(j)
                adj_list[j].append(i)

        train_labels[len(train_adjlists)]= int(label2index[label])
        train_adjlists.append(adj_list)
        train_features.append(torch.FloatTensor(feature).to(device))
        train_sequence.append(torch.tensor(smiles_seq))
    file.close()

    data_file = f"../original_datasets/{dataset}/{dataset}_test"
    file = open(data_file, "r")
    test_adjlists = []
    test_features = []
    test_sequence = []
    test_labels = np.zeros(te_len)
    for line in file:
        smiles = line.split("\t")[1]
        # print(smiles)
        label = line.split("\t")[2][:-1]
        mol = AllChem.MolFromSmiles(smiles)
        feature = torch.zeros(len(mol.GetAtoms()), len(node_types))
        l = 0
        smiles_seq = []
        for atom in mol.GetAtoms():
            feature[l, node2index[atom.GetAtomicNum()]] = 1
            smiles_seq.append(node2index[atom.GetAtomicNum()])
            l += 1
        adj_list = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            typ = bond.GetBondType()
            adj_list[i].append(j)
            adj_list[j].append(i)
            if typ == Chem.rdchem.BondType.DOUBLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
            elif typ == Chem.rdchem.BondType.TRIPLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
                adj_list[i].append(j)
                adj_list[j].append(i)

        test_labels[len(test_adjlists)] = int(label2index[label])
        test_adjlists.append(adj_list)
        test_features.append(torch.FloatTensor(feature).to(device))
        test_sequence.append(torch.tensor(smiles_seq))
    file.close()

    train_data = {}
    train_data['adj_lists'] = train_adjlists
    train_data['features'] = train_features
    train_data['sequence'] = train_sequence

    test_data = {}
    test_data['adj_lists'] = test_adjlists
    test_data['features'] = test_features
    test_data['sequence'] = test_sequence

    return train_data, train_labels, test_data, test_labels

# def load_data(dataset, device):
#     pair_list = ["Br", "Cl", "Si", "Na", "Ca", "Ge", "Cu", "Au", "Tb", "Pt", "Re", "Ru", "Bi", "Li", "Fe", "Sb", "Hg","Pb", "Se", "Ag","Cr","Pd","Ga","Mg","Ni","Ir","Rh","Te","Ti","Al","Zr","Tl"]
#     data_file = f"../original_datasets/{dataset}/{dataset}_train"
#     file = open(data_file, "r")
#     node_types = set()
#     label_types = set()
#     tr_len = 0
#     for line in file:
#         smiles = line.split("\t")[1]
#         label = line.split("\t")[2][0]
#         try:
#             mol = read_smiles(smiles)
#             tr_len += 1
#         except:
#             continue
#         i = 0
#         s = []
#         while i < len(smiles):
#             if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
#                 s.append(smiles[i] + smiles[i+1])
#                 i += 2
#             else:
#                 s.append(smiles[i])
#                 i += 1
#         node_types |= set(s)
#         label_types.add(label)
#     file.close()

#     te_len = 0
#     data_file = f"../original_datasets/{dataset}/{dataset}_test"
#     file = open(data_file, "r")
#     for line in file:
#         smiles = line.split("\t")[1]
#         label = line.split("\t")[2][0]
#         try:
#             mol = read_smiles(smiles)
#             te_len += 1
#         except:
#             continue
#         i = 0
#         s = []
#         while i < len(smiles):
#             if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
#                 s.append(smiles[i] + smiles[i+1])
#                 i += 2
#             else:
#                 s.append(smiles[i])
#                 i += 1
#         node_types |= set(s)
#         label_types.add(label)
#     file.close()

#     print(tr_len)
#     print(te_len)
#     # print(kskjd)
#     print(node_types)

#     node2index = {n: i for i, n in enumerate(node_types)}
#     label2index = {l: i for i, l in enumerate(label_types)}

#     print(node2index)
#     print(label2index)

#     data_file = f"../original_datasets/{dataset}/{dataset}_train"
#     file = open(data_file, "r")
#     train_adjlists = []
#     train_features = []
#     train_labels = torch.zeros(tr_len)
#     for line in file:
#         smiles = line.split("\t")[1]
#         print(smiles)
#         label = line.split("\t")[2][0]
#         try:
#             mol = read_smiles(smiles)
#         except:
#             continue
#         edges = mol.edges
#         nodes = mol.nodes(data='element')
#         lists = []
#         for i,n in nodes:
#             lists.append(n)
#         adjlist = defaultdict(set)
#         i = 0
#         s = 0
#         while i < len(smiles):
#             if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
#                 i += 2
#             else:
#                 i += 1
#             s += 1
#         feature = torch.zeros(s, len(node_types))
#         map = {}
#         se_num = 0
#         gr_num = 0
#         i = 0
#         while i < len(smiles):
#             this_str = smiles[i]
#             if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
#                 this_str = smiles[i] + smiles[i+1]
#                 i += 2
#             else:
#                 i += 1
#             if this_str in lists and this_str == list(nodes)[gr_num][1]:
#                 map[list(nodes)[gr_num][0]] = se_num
#                 gr_num += 1
#             elif this_str == 'c' or this_str == 'n' or this_str == 's' or this_str == 'o' or this_str == "b" or this_str == "p":
#                 map[list(nodes)[gr_num][0]] = se_num
#                 gr_num += 1
#             feature[se_num, node2index[this_str]] = 1
#             se_num += 1

#         for i, j in edges:
#             adjlist[map[i]].add(map[j])
#             adjlist[map[j]].add(map[i])

#         train_labels[len(train_adjlists)] = int(label2index[label])
#         train_adjlists.append(adjlist)
#         train_features.append(torch.FloatTensor(feature).to(device))

#     file.close()

#     data_file = f"../original_datasets/{dataset}/{dataset}_test"
#     file = open(data_file, "r")
#     test_adjlists = []
#     test_features = []
#     test_labels = np.zeros(te_len)
#     for line in file:
#         smiles = line.split("\t")[1]
#         print(smiles)
#         label = line.split("\t")[2][0]
#         try:
#             mol = read_smiles(smiles)
#         except:
#             continue
#         edges = mol.edges
#         nodes = mol.nodes(data='element')
#         lists = []
#         for i,n in nodes:
#             lists.append(n)
#         adjlist = defaultdict(set)
#         i = 0
#         s = 0
#         while i < len(smiles):
#             if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
#                 i += 2
#             else:
#                 i += 1
#             s += 1
#         feature = torch.zeros(s, len(node_types))
#         map = {}
#         se_num = 0
#         gr_num = 0
#         i = 0
#         while i < len(smiles):
#             this_str = smiles[i]
#             if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
#                 this_str = smiles[i] + smiles[i+1]
#                 i += 2
#             else:
#                 i += 1
#             if this_str in lists and this_str == list(nodes)[gr_num][1]:
#                 map[list(nodes)[gr_num][0]] = se_num
#                 gr_num += 1
#             elif this_str == 'c' or this_str == 'n' or this_str == 's' or this_str == 'o' or this_str == "b" or this_str == "p":
#                 map[list(nodes)[gr_num][0]] = se_num
#                 gr_num += 1
#             feature[se_num, node2index[this_str]] = 1
#             se_num += 1

#         for i, j in edges:
#             adjlist[map[i]].add(map[j])
#             adjlist[map[j]].add(map[i])

#         test_labels[len(test_adjlists)] = int(label2index[label])
#         test_adjlists.append(adjlist)
#         test_features.append(torch.FloatTensor(feature).to(device))
#     file.close()

#     train_data = {}
#     train_data['adj_lists'] = train_adjlists
#     train_data['features'] = train_features

#     test_data = {}
#     test_data['adj_lists'] = test_adjlists
#     test_data['features'] = test_features

#     # print(train_adjlists)
#     # print(train_features)
#     return train_data, train_labels, test_data, test_labels

# def load_data(dataset, device):
#     data_file = f"../original_datasets/{dataset}/{dataset}_train"
#     file = open(data_file, "r")
#     node_types = set()
#     label_types = set()
#     tr_len = 0
#     for line in file:
#         tr_len += 1
#         smiles = line.split("\t")[1]
#         s = []
#         mol = AllChem.MolFromSmiles(smiles)
#         for atom in mol.GetAtoms():
#             s.append(atom.GetAtomicNum())
#         node_types |= set(s)
#         label = line.split("\t")[2][0]
#         label_types.add(label)
#     file.close()

#     te_len = 0
#     data_file = f"../original_datasets/{dataset}/{dataset}_test"
#     file = open(data_file, "r")
#     for line in file:
#         te_len += 1
#         smiles = line.split("\t")[1]
#         s = []
#         mol = AllChem.MolFromSmiles(smiles)
#         for atom in mol.GetAtoms():
#             s.append(atom.GetAtomicNum())
#         node_types |= set(s)
#         label = line.split("\t")[2][0]
#         label_types.add(label)
#     file.close()

#     print(tr_len)
#     print(te_len)

#     node2index = {n: i for i, n in enumerate(node_types)}
#     label2index = {l: i for i, l in enumerate(label_types)}

#     print(node2index)
#     print(label2index)

#     data_file = f"../original_datasets/{dataset}/{dataset}_train"
#     file = open(data_file, "r")
#     train_adjlists = []
#     train_features = []
#     train_labels = torch.zeros(tr_len)
#     for line in file:
#         smiles = line.split("\t")[1]
#         # print(smiles)
#         label = line.split("\t")[2][0]
#         mol = AllChem.MolFromSmiles(smiles)
#         feature = torch.zeros(len(mol.GetAtoms()), len(node_types))
#         l = 0
#         for atom in mol.GetAtoms():
#             feature[l, node2index[atom.GetAtomicNum()]] = 1
#             l += 1
#         adj_list = defaultdict(list)
#         for bond in mol.GetBonds():
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             typ = bond.GetBondType()
#             adj_list[i].append(j)
#             adj_list[j].append(i)
#             if typ == Chem.rdchem.BondType.DOUBLE:
#                 adj_list[i].append(j)
#                 adj_list[j].append(i)
#             elif typ == Chem.rdchem.BondType.TRIPLE:
#                 adj_list[i].append(j)
#                 adj_list[j].append(i)
#                 adj_list[i].append(j)
#                 adj_list[j].append(i)

#         train_labels[len(train_adjlists)]= int(label2index[label])
#         train_adjlists.append(adj_list)
#         train_features.append(torch.FloatTensor(feature).to(device))
#     file.close()

#     data_file = f"../original_datasets/{dataset}/{dataset}_test"
#     file = open(data_file, "r")
#     test_adjlists = []
#     test_features = []
#     test_labels = np.zeros(te_len)
#     for line in file:
#         smiles = line.split("\t")[1]
#         # print(smiles)
#         label = line.split("\t")[2][0]
#         mol = AllChem.MolFromSmiles(smiles)
#         feature = torch.zeros(len(mol.GetAtoms()), len(node_types))
#         l = 0
#         for atom in mol.GetAtoms():
#             feature[l, node2index[atom.GetAtomicNum()]] = 1
#             l += 1
#         adj_list = defaultdict(list)
#         for bond in mol.GetBonds():
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             typ = bond.GetBondType()
#             adj_list[i].append(j)
#             adj_list[j].append(i)
#             if typ == Chem.rdchem.BondType.DOUBLE:
#                 adj_list[i].append(j)
#                 adj_list[j].append(i)
#             elif typ == Chem.rdchem.BondType.TRIPLE:
#                 adj_list[i].append(j)
#                 adj_list[j].append(i)
#                 adj_list[i].append(j)
#                 adj_list[j].append(i)

#         test_labels[len(test_adjlists)] = int(label2index[label])
#         test_adjlists.append(adj_list)
#         test_features.append(torch.FloatTensor(feature).to(device))
#     file.close()

#     train_data = {}
#     train_data['adj_lists'] = train_adjlists
#     train_data['features'] = train_features

#     test_data = {}
#     test_data['adj_lists'] = test_adjlists
#     test_data['features'] = test_features

#     return train_data, train_labels, test_data, test_labels

