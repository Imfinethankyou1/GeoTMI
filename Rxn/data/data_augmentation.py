import os
from multiprocessing import Pool

functional = 'functional'
functional = 'b97d3'

with open(f'{functional}.csv') as f:
    lines = f.readlines()

index_to_smarts = {}
for line in lines[1:]:
    elements = line.strip().split(',')
    index = elements[0]
    len_index = len(index)
    for i in range(6-len_index):
        index = '0' +index
    r_smarts = elements[1]
    p_smarts = elements[2]
    index_to_smarts[index] = r_smarts+','+p_smarts



def read_sp_zero(fn):
    with open(fn) as f:
        lines = f.readlines()
    
    position_check = False
    for line in lines:
        if 'ccsd' in fn:
            if '!CCSD(T)-F12a total energy' in line:
                SP = 627.509608030592*float(line.strip().split()[-1])

        else:    
            if 'Zero point vibrational energy:' in line:
                zero_VE = float( line.strip().split()[4])
            if 'energy in the final basis set' in line:
                SP = 627.509608030592*float(line.strip().split()[-1])

        if 'I     Atom           X                Y' in line:
            symbols = []
            positions = []
            position_check = True
            count =0
        if position_check:
            if count >1:
                if not '--'  in line:
                    element, x, y, z= line.strip().split()[1:5]
                    symbols.append(element)
                    positions.append([x, y, z])
                else:
                    position_check = False
            count +=1
    try:
        val = SP + 0
    except:
        print(fn)
    return SP, zero_VE, symbols, positions

def multiprocessing(function,elements, ncores):
    pool = Pool(processes = ncores)
    results = pool.map(function,elements)
    pool.terminate()
    pool.join()

    return results

csv_name = f'{functional}_fwd_rev.csv'
f_csv = open(csv_name,'w')

r_geo_name = f'{functional}_reactants.txt'
f_r_geo = open(r_geo_name,'w')

p_geo_name = f'{functional}_products.txt'
f_p_geo = open(p_geo_name,'w')

ts_geo_name = f'{functional}_ts.txt'
f_ts_geo = open(ts_geo_name,'w')

def make_geo_lines(symbols, positions,label):

    assert len(symbols) == len(positions)
    lines = [f'{len(symbols)}\n{label}\n']
    for symbol, position in zip(symbols, positions):
        line = f'{symbol}\t{position[0]}\t{position[1]}\t{position[2]}\n'
        lines.append(line)
    lines += '$$$$$\n'
    return lines

def get_ts_energy(fn):
    index = fn.split('rxn')[-1]
    fn_r = f'{functional}/{fn}/r{index}.log'
    fn_p = f'{functional}/{fn}/p{index}.log'
    fn_ts = f'{functional}/{fn}/ts{index}.log'
    
    r_sp, r_zero, symbols, r_positions = read_sp_zero(fn_r)
    p_sp, p_zero, p_symbols, p_positions = read_sp_zero(fn_p)
    ts_sp, ts_zero, ts_symbols, ts_positions = read_sp_zero(fn_ts)

    ea = ts_zero+ts_sp - r_sp - r_zero
    ea_rev = ts_zero+ts_sp - (p_sp+p_zero)
    
    E_ts = ts_zero+ts_sp
    E_r = r_sp + r_zero

    assert symbols == p_symbols == ts_symbols
    if index in index_to_smarts:
        smarts= index_to_smarts[index]

        csv_lines = [ ','.join([index, smarts,str(ea)]),  ','.join([index+'_rev', smarts,str(ea_rev)])  ]
        
        r_geo_lines = make_geo_lines(symbols, r_positions, fn_r)
        p_geo_lines = make_geo_lines(p_symbols, p_positions,fn_p)
        ts_geo_lines = make_geo_lines(ts_symbols, ts_positions,fn_ts)

        return csv_lines, r_geo_lines, p_geo_lines, ts_geo_lines
    else:
        return '', '' ,'', ''

fns = os.listdir(f'{functional}/')
new_fns = []
for fn in fns:
    if not '.log' in fn and not '._' in fn:
        new_fns.append(fn)
fns = new_fns       
fns.sort()


results=multiprocessing(get_ts_energy, fns, 8)
for result in results:
    csv_lines, r_geo_lines, p_geo_lines, ts_geo_lines = result
    for line in csv_lines:
        f_csv.write(line+'\n')
    for r_line, p_line, ts_line in zip(r_geo_lines, p_geo_lines, ts_geo_lines):
        if r_line != '':
            f_r_geo.write(r_line)
            f_p_geo.write(p_line)
            f_ts_geo.write(ts_line)
        
    for r_line, p_line, ts_line in zip(r_geo_lines, p_geo_lines, ts_geo_lines):
        if r_line !='':
            f_r_geo.write(p_line)
            f_p_geo.write(r_line)
            f_ts_geo.write(ts_line)


