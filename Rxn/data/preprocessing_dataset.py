from collections import namedtuple
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.undirected import to_undirected
from ase.io import read
from multiprocessing import Pool
import pickle
import itertools
import math
import os
import argparse
from rdkit import Chem
from rdkit.Chem import rdmolfiles
import pandas

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--cutoff', type=int, default=5,
              help='cutoff of radial edge index')
parser.add_argument('--origin_data_dir', type=str, default='./total_data',
              help='Dacon data directory')
parser.add_argument('--save_data_dir', type=str, default='save_graph_data',
              help='Directory of traning dataset using fully connected edge')

args = parser.parse_args()

symbol2charge = {'H':1,'C':6, 'O':8, 'N':7, 'F':9}
cutoff = args.cutoff
#data_dir = f'aromatic_cutoff_{cutoff}'
radial_edge_data_dir = f'{cutoff}_'+args.radial_edge_data_dir
save_data_dir = args.save_data_dir
origin_data_dir = args.origin_data_dir

def obtain_mean_variance(vals):
    mean = sum(vals) / len(vals)
    vsum = 0
    for val in vals:
        vsum = vsum + (val - mean)**2
    variance = vsum / len(vals)

    std = math.sqrt(variance)
    std = np.std(vals)
    return mean, std

ATOM_DICT = {
    1: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    14: 7,
    15: 8,
    16: 9,
    17: 10,
    35: 11,
    53: 12,
}  # atomic number --> index


fn= 'b97d3_fwd_rev_chemprop.csv'
data= pandas.read_csv(fn)
prop_list=data['ea']

with open('CV_data/b97d3/train_seed_0.txt') as f:
    lines = f.readlines()
    train_labels = [ int(line.strip()) for line in lines ]

with open('CV_data/b97d3/val_seed_0.txt') as f:
    lines = f.readlines()
    val_labels = [ int(line.strip()) for line in lines ]

prop_list = prop_list[train_labels+val_labels]
#prop_list_ = prop_list[train_labels]
mean, std = obtain_mean_variance(prop_list)
print('mean : ', mean)
print('std : ', std)

#sys.exit()
if not os.path.isdir(radial_edge_data_dir):
    os.system('mkdir '+radial_edge_data_dir)
if not os.path.isdir(save_data_dir):
    os.system('mkdir '+save_data_dir)


def make_database_one(params):
    label, r, ts, p, target = params
    
    symbols = r['symbols']
    r_coords = torch.FloatTensor(np.array(r['coords']))
    p_coords = torch.FloatTensor(np.array(p['coords']))
    ts_coords = torch.FloatTensor(np.array(ts['coords']))
    atomids = []
    for symbol in symbols:
        atom_charge = symbol2charge[symbol]
        atomids.append(ATOM_DICT[atom_charge])
    
    atomids = torch.LongTensor(np.array(atomids))

    
    graph_data = Data(
        atomids=atomids,
        r_coords = r_coords,
        p_coords = p_coords,
        ts_coords = ts_coords,
        num_nodes = atomids.size(0),
        target = target,
        label = label
    )
    output = radial_edge_data_dir + f'/{label}.npz'
    with open(output,'wb') as f:
        pickle.dump(graph_data,f)
    del graph_data

# The function was copied from chemprop (https://github.com/chemprop/chemprop)
def get_dicts(xyz_path):
    """Creates list of dictionaries containing the molecule coordinates and atomic numbers"""
    dicts = []
    with open(xyz_path, 'r') as f:
        xyz_lines = ''
        for line in f.readlines():
            if '$$$$$' in line:
                dicts.append(xyz_file_format_to_xyz(xyz_lines))
                xyz_lines = ''
            else:
                xyz_lines += line
    return dicts


# The function was copied from chemprop (https://github.com/chemprop/chemprop)
def xyz_file_format_to_xyz(xyz_file):
    """
    Creates a xyz dictionary from an `XYZ file format <https://en.wikipedia.org/wiki/XYZ_file_format>`_ representation.

    Args:
        xyz_file (str): The content of an XYZ file

    Returns:
        dict: An xyz dictionary.

    Raises:
        ConverterError: If cannot identify the number of atoms entry or if it is different that the actual number.
    """

    lines = xyz_file.strip().splitlines()
    if not lines[0].isdigit():
        raise ConverterError('Cannot identify the number of atoms from the XYZ file format representation. '
                             'Expected a number, got: {0} of type {1}'.format(lines[0], type(lines[0])))
    number_of_atoms = int(lines[0])
    lines = lines[2:]
    if len(lines) != number_of_atoms:
        raise ConverterError('The actual number of atoms ({0}) does not match the expected number parsed ({1}).'.format(
            len(lines), number_of_atoms))
    xyz_str = '\n'.join(lines)

    xyz_dict = {'symbols': tuple(), 'coords': tuple()}
    for line in xyz_str.strip().splitlines():
        if line.strip():
            splits = line.split()
            if len(splits) != 4:
                raise ConverterError(f'xyz_str has an incorrect format, expected 4 elements in each line, '
                                     f'got "{line}" in:\n{xyz_str}')
            symbol = splits[0]
            if '(iso=' in symbol.lower():
                symbol = symbol.split('(')[0]
            else:
                # no specific isotope is specified in str_xyz
                #isotope = NUMBER_BY_SYMBOL[symbol]
                khskhs = 0
            coord = (float(splits[1]), float(splits[2]), float(splits[3]))
            #if symbol != 'H':
            xyz_dict['symbols'] += (symbol,)
            xyz_dict['coords'] += (coord,)
            #print(symbol)
    return xyz_dict



def make_database(labels, r_dicts, ts_dicts, p_dicts, prop_list):
    params_list = []

    for label, r, ts, p, prop in zip(labels, r_dicts, ts_dicts, p_dicts, prop_list):
        print(label)    
        target = torch.FloatTensor([prop])
        params = [label, r, ts, p, target]        
        params_list.append( params )
        make_database_one(params)

if __name__ == '__main__':

    data = pandas.read_csv('b97d3_fwd_rev_chemprop.csv')
    print(data)
    prop_list = data['ea']
    r_dicts = get_dicts('b97d3_reactants.txt')  # list of reactant dictionaries
    ts_dicts = get_dicts('b97d3_ts.txt')       # list of ts dictionaries
    p_dicts = get_dicts('b97d3_products.txt')       # list of ts dictionaries

    # Generation of train set
    new_prop_list = []
    for i, prop in enumerate(prop_list):
        new_prop_list.append( (prop-mean)/std )
    prop_list = new_prop_list
    labels = [i for i in range(len(prop_list))]

    make_database(labels, r_dicts, ts_dicts, p_dicts, prop_list)
   
