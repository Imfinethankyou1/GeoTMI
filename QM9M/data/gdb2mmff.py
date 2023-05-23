import time
import os
import argparse


parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--QM9_data_dir', type=str, default='./qm9origin/',
              help='QM9 data directory')
parser.add_argument('--QM9M_data_dir', type=str, default='./qm9mmff/',
              help='QM9M data directory')

args = parser.parse_args()



key_list = []

qm9origin_label2line = {}
labels = [f'{i+1}' for i in range(133885)]

excluded_list = []
with open('uncharacterized.txt') as f:
    lines = f.readlines()

    for line in lines:
        idx = line.strip().split()[0]
        excluded_list.append('gdb_'+idx)

qmlabel2target = {}
target2mmff = {}

for label in labels:
    new_label = label
    num = 6-len(label)
    for i in range(num):
        new_label = '0'+new_label
    with open(f'{args.QM9_data_dir}/data/dsgdb9nsd_{new_label}.xyz') as f:
        props,line = f.readlines()[1:3]
        x,y,z =line.strip().split()[1:4]
        print(f'{args.QM9_data_dir}/data/dsgdb9nsd_{new_label}.xyz')
        if '*' in x:
            x= 0.0
        if '*' in y:
            y= 0.0
        if '*' in z:
            z= 0.0
        x= str(round(float(x),4))
        if x  == '-0.0':
            x = '0.0'
        y= str(round(float(y),4))
        if y  == '-0.0':
            y = '0.0'
        z= str(round(float(z),4))
        if z  == '-0.0':
            z = '0.0'

        prop = props.strip().split()[2]
        if not '.' in prop:
            prop+='.0'
        if prop[-1] == '.':
            prop += '0'

        key = ','.join([x,y,z, prop])
        qmlabel2target[label] = key
        
   
    with open(f'{args.QM9M_data_dir}/{label}.sdf')  as f:
        line = f.readlines()[4]

    if 'BEGIN' in line:
        with open(f'{args.QM9M_data_dir}/{label}.sdf')  as f:
            line = f.readlines()[7]
        x,y,z = line.strip().split()[4:7]
    else:
        x,y,z = line.strip().split()[:3]

    x= str(round(float(x),4))
    if x  == '-0.0':
        x = '0.0'
    y= str(round(float(y),4))
    if y  == '-0.0':
        y = '0.0'
    z= str(round(float(z),4))
    if z  == '-0.0':
        z = '0.0'

    with open(f'{QM9M_data_dir}/{label}.sdf')  as f:
        line = f.readlines()[0]
    prop = line.strip().split()[0] 
    key = ','.join([x,y,z,prop])
    target2mmff[key] = label

start = time.time()
label2qm9 = {}
idx = 0
qmlabel2mmff = {}
for label in labels:
    target = qmlabel2target[label]
    qmlabel2mmff[label] = target2mmff[target]

with open('qm2mmff_label.txt','w') as f:
    with open('gdb9.sdf.csv') as g:
        lines = g.readlines()

    f.write(lines[0].strip()+',mmff\n')
    for label, line in enumerate(lines[1:]):
        label = str(label+1)
        f.write(line.strip()+','+qmlabel2mmff[label]+'\n')
