from collections import defaultdict
import logging
from random import Random
from typing import Dict, List, Set, Tuple, Union
import warnings

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np
import pickle

# The function was copied from chemprop (https://github.com/chemprop/chemprop)
def generate_scaffold(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol) 
    if isinstance(mol, tuple):
        mol = mol[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol = mol, includeChirality = include_chirality)

    return scaffold

# The function was copied from chemprop (https://github.com/chemprop/chemprop)
def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).

    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total = len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


# The function was copied from chemprop (https://github.com/chemprop/chemprop)
def scaffold_split(mols,
                   sizes: Tuple[float, float, float] = (0.85, 0.05, 0.1),
                   balanced: bool = True,
                   seed: int = 0,
                   ) :
    r"""
    Splits a :class:`~chemprop.data.MoleculeDataset` by scaffold so that no molecules sharing a scaffold are in different splits.

    :param data: A :class:`MoleculeDataset`.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param balanced: Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    :param seed: Random seed for shuffling when doing balanced splitting.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    assert sum(sizes) == 1


    # Split
    train_size, val_size, test_size = sizes[0] * len(mols), sizes[1] * len(mols), sizes[2] * len(mols)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(mols, use_indices=True)

    # Seed randomness
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    return train, val, test
def make_label_fwd_rev(label_list):
    new_train = []
    for label in label_list:
        new_train+=[2*label,2*label+1]
    return new_train

if __name__ =="__main__":
    
    with open('ccsdtf12_uni_fwd_rev.csv') as f:
        lines = f.readlines()

    mols = []
    for index, line in enumerate(lines[1:]):
        elements = line.strip().split(',')
        if index %2 == 0:
            r_smarts = elements[1]
            p_smarts = elements[2]
            mols.append(r_smarts)

    train, val, test =scaffold_split(mols)
    train = make_label_fwd_rev(train)
    val = make_label_fwd_rev(val)
    test = make_label_fwd_rev(test)
    print(len(train), len(val), len(test))
    print(type(train))
    print(train[0])
    data = np.array([np.array(train), np.array(val), np.array(test)])
    data = np.array([data])
    print(data[0][0])
    with open('train_seed_0.txt','w') as f:
        for label in train:
            f.write(f'{label}\n')

    with open('val_seed_0.txt','w') as f:
        for label in val:
            f.write(f'{label}\n')

    with open('test_seed_0.txt','w') as f:
        for label in test:
            f.write(f'{label}\n')

    with open('seed0.pkl','wb') as f:
        pickle.dump(data,f)
