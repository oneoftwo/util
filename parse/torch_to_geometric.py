import torch
from torch.nn import functional as F
from torch_geometric import nn
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric import nn
from torch_geometric.nn import SchNet
import time
import numpy as np


def _generate_LP_edge_index(l_R, p_R, knn=4, is_cuda=False):
    r"""
    generation of LP edge index (considering as single graph of \
            ligand and protein)
    args:
        l_R: torch.Tensor [n_ligand_atom dimension]
        p_R: torch.Tensor [n_protein_atom dimension]
        knn: int, k nearest
    output:
        edge_index: edge index,torch geometric format
        total_R: concatenated l_R and p_R
    """
    n_l, n_p = l_R.size(0), p_R.size(0)
    lp_distance = torch.cdist(l_R, p_R)
    p_index = lp_distance.topk(k=knn, largest=False).indices
    edge_index_l = torch.Tensor(range(n_l)).unsqueeze(1).repeat(1, knn).view(-1)
    edge_index_p = p_index.view(-1) + n_l
    edge_index = torch.stack([edge_index_l, edge_index_p], dim=0)
    total_R = torch.cat([l_R, p_R], dim=0)
    return edge_index, total_R


def inverse_generate_LP_edge_index_from_coord(data):
    pass


def _generate_LL_edge_index(l_R):
    # fully connected graph
    n_l = l_R.size(0)
    e = torch.Tensor(range(n_l))
    e = torch.stack([e, e], dim=0)
    return e


def generate_geo_data_from_LP(l_Z, l_R, p_Z, p_R, knn=4):
    edge_index, total_R = _generate_LP_edge_index(l_R, p_R, knn=knn)
    total_Z = torch.cat([l_Z, p_Z], dim=0)
    data = Data(x=total_Z, pos=total_R, edge_index=edge_index)
    return data


def generate_geo_data_from_L(l_Z, l_R):
    edge_index = _generate_LL_edge_index(l_R)
    data = Data(x=l_Z, pos=l_R, edge_index=edge_index)
    return data


def remove_node(data, index_list):
    """ 
    remove a node () from torch gemoetry graph data 
    arg: 
        data: torch geometric data 
        index_list: list (or itterable) of index desired to be removed 
    ouput:
        data: torch geometric data 
    """
    from copy import deepcopy
    x, pos, edge_index = data.x, data.pos, data.edge_index
    x, pos, edge_index = deepcopy([x, pos, edge_index])
    for idx in index_list:
        pass




if __name__ == '__main__':
    l_R = torch.rand(4, 3)
    l_Z = torch.rand(4, 16)
    p_R = torch.rand(7, 3)
    p_Z = torch.rand(7, 16)
    print(l_R)
    print(p_R)
    print()
    data = generate_geo_data_from_L(l_Z, l_R)
    print(type(data.x))
    print(type(data.pos))
    remove_node(data, [3,4,5])
    
    a = [[1], [2], [3]]
    e = [[1, 3], [3, 1]]
    data = Data(a, e)
    a_ = [[2], [1], ['c']]
    e_ = [[2, 3], [3, 2]]
    data_ = Data(a_, e_)
    print(data)
    print(data_)
    print(type(data.x))




