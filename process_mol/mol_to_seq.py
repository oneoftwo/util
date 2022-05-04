import numpy as np
from rdkit import Chem
import random


a = 'CC(=O)NCCC1=CNc2c1cc(OC)cc2'
# a = 'CN=CO'
m = Chem.MolFromSmiles(a)


def get_all_possible_smiles(m):
    """ 
    get all possible (mostly) smiles for given mol 
    input:
        m: mol object
    output: 
        smiles_list: possible smiles_list 
    """
    smiles_list = []
    while True:
        smiles = Chem.MolToSmiles(m, doRandom=True)
        if not smiles in smiles_list:
            smiles_list.append(smiles)
            c = 0
        else:
            c += 1
        if (c > len(smiles_list) * 10) and (c > 100):
            break
    return smiles_list


def get_random_smiles(m):
    """ 
    get random smiles (non cannonical) from mol
    """
    smiles = Chem.MolToSmiles(m, doRandom=True)
    return smiles


def smiles_to_seq(smiles, c_to_i, make_cannonical=False):
    if cannonical:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    seq = []
    for char in smiles:
        if char in c_to_i:
            i = c_to_i.index(char)
            seq.append(i)
        else:
            seq.append(len(c_to_i))
    seq = np.array(seq)
    return smiles, seq


def update_c_to_i(smiles_list, c_to_i=[], make_cannonical=False):
    if cannonical:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    for smiles in smiles_list:
        for char in smiles:
            if not char in c_to_i:
                c_to_i.append(char)
    return c_to_i


