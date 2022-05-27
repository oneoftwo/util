import numpy as np

from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs

try:
    import mol_to_graph as MOLTOGRAPH
    import mol_to_3d as MOLTO3D
    from . import mol_to_graph as MOLTOGRAPH 
    from . import mol_to_3d as MOLTO3D
except:
    pass 


def mol_to_edge_index_and_edge_attr(mol): 
    adj = MOLTOGRAPH.mol_to_adjacency_matrix(mol, is_self_loop=True)
    e = MOLTOGRAPH.mol_to_edge_feature_matrix(mol, include_extra=False)
    N = adj.shape[0]
    ei_1 = []
    ei_2 = []
    edge_attr = []
    for i in range(N):
        for j in range(N):
            if adj[i, j] == 1:
                edge_feature = e[i, j]
                ei_1.append(i)
                ei_2.append(j)
                edge_attr.append(list(edge_feature))
    edge_index = np.array([ei_1, ei_2])
    edge_index = np.array(edge_index)
    edge_attr = np.array(edge_attr)
    return edge_index, edge_attr


def mol_to_pos(mol):
    pos = MOLTO3D.mol_to_position_matrix(mol, optimize=True)
    pos = np.array(pos)
    return pos 


def mol_to_data(mol, pos=False):
    mol = RemoveHs(mol)
    x = MOLTOGRAPH.mol_to_node_feature_matrix(mol)
    edge_index, edge_attr = mol_to_edge_index_and_edge_attr(mol)
    if pos:
        pos = mol_to_pos(mol)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    else:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def merge_graph(graph_1, graph_2):
    """ 
    merge graph (graph_1 - graph_2) index shift 
    """
    print(graph_1)
    print(graph_2)
    idx_shift = len(graph_1.x)
    print(idx_shift)

    x_1, edge_index_1 = graph_1.x, graph_1.edge_index
    x_2, edge_index_2 = graph_2.x, graph_2.edge_index

    x = np.concatenate([x_1, x_2], axis=0)
    edge_index_2 = edge_index_2 + idx_shift 
    edge_index = np.concatenate([edge_index_1, edge_index_2], axis=1)
    
    try:
        edge_attr_1 = graph_1.edge_attr
        edge_attr_2 = graph_2.edge_attr
        edge_attr = np.concatenate([edge_attr_1, edge_attr_2], axis=0)
    except:
        pass 

    try:
        pos_1 = graph_1.pos 
        pos_2 = graph_2.pos
        pos = np.concatenate([pos_1, pos_2], axis=0)
    except:
        pass

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    graph.seperate_index = [idx_shift, len(x) - idx_shift]
    return graph
 
    
if __name__ == '__main__':
    mol = Chem.MolFromSmiles('CNCC')
    graph_1 = mol_to_data(mol, pos=True)
    mol = Chem.MolFromSmiles('CC')
    graph_2 = mol_to_data(mol, pos=True)
    graph = merge_graph(graph_1, graph_2)
    print(graph)

