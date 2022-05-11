import numpy as np
import random 

from rdkit import Chem


a = 'CC(=O)NCCC1=CNc2c1cc(OC)cc2'
m = Chem.MolFromSmiles(a)


def get_all_possible_smiles(m):
    """ 
    !!! takes time
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


def get_random_smiles(mol, kekulize=False, isomeric=False, \
        explicit_bond=False, explicit_H=False):
    """ 
    get random smiles (non cannonical) from mol
    """
    if kekulize:
        kekulize = bool(random.randint(0, 1))
    if isomeric:
        isomeric = bool(random.randint(0, 1))
    if explicit_bond:
        allBondsExplicit = bool(random.randint(0, 1))
    if explicit_H:
        allHsExplicit = bool(random.randint(0, 1))

    smiles = Chem.MolToSmiles(mol, doRandom=True, kekuleSmiles=kekulize, \
            isomericSmiles=isomeric, allBondsExplicit=allBondsExplicit, \
            allHsExplicit=allHsExplicit)
    return smiles


def smiles_to_seq(smiles, c_to_i, make_cannonical=False):
    if make_cannonical:
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



if __name__ == '__main__':
    smiles = a
    for _ in range(10):
        print(get_random_smiles(Chem.MolFromSmiles(smiles), kekulize=True, isomeric=True, explicit_bond=True, explicit_H=True))

