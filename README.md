# Geometric denoising for Three-term Mutual Information maximization (GeoTMI)
------------------------------------------------------------------------
The GeoTMI is a model-agnostic method to solve the practical infeasibility of high-cost 3D geometry in many other chemistry fields.
The aim of GeoTMI is maximazation of the mutual information between high-cost 3D geometries, correspoding quntum chemical properties, and easy-to-obtain geometries.

## Dependencies
---------------------------------------------------
* torch==1.12.1
* ase==3.21.1
* torch-geometric==2.0.4
* cudatoolkit==11.3.1
* torch_cluster==1.6.0

## How to install environment
-----------------------------
    source install.sh

```yaml
install.sh:
    conda create -n REP -y
    source activate REP
    conda install -c conda-forge mamba -y
    mamba install xtensor-r -c conda-forge -y
    mamba install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html 
    pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html 
    pip install torch-geometric==2.0.4 
    pip install ase==3.21.1 
    pip install networkx 
    pip install torch_cluster -f https://data.pyg.org/whl/torch-1.12.1+cu113.html 
    pip install sympy 
    pip install pandas 
    pip install rdkit-pypi 
```

## OC20
-------------------------------
1. Please put files and ocp directory to corresponding original equiformer directory.
ex) cp OC20/equiformer/oc20/trainer/* [EQUIFOMRER_ORIGIN_PATH]/oc20/trainer/
2. source GeoTMI.yml

## QM9
-------------------------------
1. Read QM9M/data/QM9_REAME.md
2. cd QM9M/data
3. python gdb2mmff.py 
4. python preprocessing_dataset.py
