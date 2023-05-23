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
from ase.units import Hartree, eV, Bohr, Ang
import pandas

symbol2charge = {'H':1,'C':6, 'O':8, 'N':7, 'F':9}

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--QM9M_data_dir', type=str, default='./qm9mmff/',
              help='QM9M data directory')
parser.add_argument('--QM9_data_dir', type=str, default='./qm9origin/',
              help='QM9 data directory')
parser.add_argument('--save_data_dir', type=str, default='geom_data',
              help='Directory of traning dataset using fully connected edge')

args = parser.parse_args()

save_data_dir = args.save_data_dir
QM9M_data_dir = args.QM9M_data_dir

def obtain_mean_variance(vals):
    mean = sum(vals) / len(vals)
    vsum = 0
    for val in vals:
        vsum = vsum + (val - mean)**2
    variance = vsum / len(vals)

    std = math.sqrt(variance)
    std = np.std(vals)
    return mean, std

def multiprocessing(function,elements, ncores):
    pool = Pool(processes = ncores)
    results = pool.map(function,elements)
    pool.terminate()
    pool.join()
    return results


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


keys = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'u0_atom', 'cv' , 'zpve' ]
converts = [1., 1., 27.2114, 27.2114, 27.2114, 1., 1.0/23.06, 1.0, 27.2114  ] 

with open('train_labels_egnn.txt') as f:
    train_lines = f.readlines()
with open('val_labels_egnn.txt') as f:
    validation_lines = f.readlines()
with open('test_labels_egnn.txt') as f:
    test_lines = f.readlines()

train_val_idx = [int(line.strip())-1 for line in train_lines] + [int(line.strip())-1 for line in validation_lines]
mean_list = []
std_list = []
data=pandas.read_csv('qm2mmff_label.txt')

label2mmff = {}
for label, mmff in zip(data['mol_id'], data['mmff']):
    label2mmff[label] = str(mmff)

for key, convert in zip(keys, converts):
    prop_list = list(data[key][train_val_idx]*convert)
    mean, std = obtain_mean_variance(prop_list)
    mean_list.append(mean)
    std_list.append(std)

print('keys : ', keys)
print('mean_list : ', mean_list)
print('std_list : ', std_list)
print('Done')

if not os.path.isdir(save_data_dir):
    os.system('mkdir '+save_data_dir)


def process_file(fn):
    coords = []
    atomids = []
    atom_charges, atom_positions = [], []
    if '.xyz' in fn:
        with open(fn) as f:
            xyz_lines = f.readlines()
        num_atoms = int(xyz_lines[0])
        mol_props = xyz_lines[1].split()
        mol_xyz = xyz_lines[2:num_atoms+2]
        for line in mol_xyz:
            atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
            atom_charges.append(symbol2charge[atom])
            atom_positions.append([float(posx), float(posy), float(posz)])

    else:
        mol = read(fn)
        mol_xyz = mol.get_positions()
        num_atoms = len(mol_xyz)
        charges = list(mol.get_atomic_numbers())

        for charge, position in zip(charges ,mol_xyz):
            posx, posy, posz = list(position)
            atom_positions.append([float(posx), float(posy), float(posz)])
            atom_charges.append(charge)
        del mol

    for ind in range(num_atoms):
        #if atom_charges[ind] >1:
        x, y, z= atom_positions[ind]
        coords.append( (x,y,z)  )
        atomids.append(ATOM_DICT[atom_charges[ind]])

    atomids = torch.LongTensor(np.array(atomids))
    coords = torch.FloatTensor(np.array(coords))

    return atomids, coords

def read_other_format_sdf(lines):
    read_line = False

    coords = []
    atomids = []
    for line in lines:
        if 'END ATOM' in line:
            break            
        if read_line:
            elements = line.strip().split()
            symbol = elements[3]
            x = float(elements[4])
            y = float(elements[5])
            z = float(elements[6])
            position = [x, y, z]
            atomids.append(ATOM_DICT[symbol2charge[symbol]])
            coords.append(position)
        if 'BEGIN ATOM' in line:
            read_line = True

    atomids = torch.LongTensor(np.array(atomids))
    coords = torch.FloatTensor(np.array(coords))
    return atomids, coords            


def make_database_one(params):
    label, g_xyz_fn, ex_xyz_fn, target =  params
    print(label)
    atomids, coords = process_file(g_xyz_fn)
    _, coords_ex = process_file(ex_xyz_fn)

   
    if len(coords) != len(coords_ex):
        if len(coords) == 0:
            with open(g_xyz_fn) as f:
                lines = f.readlines()
            atomids, coords =read_other_format_sdf(lines)              
            print('format diff MMFF') 
        if len(coords_ex) == 0:
            with open(ex_xyz_fn) as f:
                lines = f.readlines()
            _ , coords_ex =read_other_format_sdf(lines)               
            print('format diff QM9')
    
    print( len(coords) , len(coords_ex)) 
    
    
    assert len(coords) == len(coords_ex)  == len(atomids)

    graph_data = Data(
        atomids=atomids,
        coords=coords,
        coords_ex = coords_ex,
        num_nodes=atomids.size(0),
        target = target,
        label = label
    )

    output = save_data_dir + f'/{label}.npz'
    with open(output,'wb') as f:
        pickle.dump(graph_data,f)
    del graph_data


def make_database(label_list, prop_dict):
    params_list = []
    for label in label_list:
        g_xyz_fn = QM9M_data_dir+f'{label2mmff[label]}.mmff.sdf'
        new_label = label.split('_')[-1]
        num = 6-len(new_label)
        for i in range(num):
            new_label = '0'+new_label
        
        ex_xyz_fn = f'{args.QM9_data_dir}/data/dsgdb9nsd_{new_label}.xyz'
        target = torch.FloatTensor([prop_dict[label]])
        params = [label, g_xyz_fn, ex_xyz_fn,  target ]
        params_list.append( params )
        make_database_one(params)

if __name__ == '__main__':

    data= pandas.read_csv('qm2mmff_label.txt')

    labels = list(data['mol_id'])
    labels = [ str(label) for label in labels]

    key_prop_list = []
    for key in keys:
        prop_list = data[key]
        key_prop_list.append(prop_list)

    prop_dict = {}
    for i in range(len(key_prop_list[0])):
        prop_list = [(key_prop_list[key][i]*converts[key]-mean_list[key])/std_list[key] for key in range(len(keys))]
        prop_dict[str(data['mol_id'][i])] = prop_list
    
    make_database(labels, prop_dict)
