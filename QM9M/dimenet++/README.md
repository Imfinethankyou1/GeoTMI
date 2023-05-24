# GeoTMI in DimeNet++

> The following code should be executed in current directory.

## Training

To train `DimeNet++ + GeoTMI` on QM9, run:

```train
task="ours"       # one of {dft,mmff,ours}
target="gap"      # one of {alpha, cv, gap, homo, lumo mu, r2, u0_atom, zpve}
dist_loss_ratio=0.1
EXP_NAME="${dist_loss_ratio}"

python -u train.py \
  --data_dir ${YOUR_DOWNLOADED_DATA_DIRECTORY} \
  --batch_size 64 \
  --lr 0.0001 \
  --task $task \
  --target $target \
  --save_dir results/${task}/${target}/${EXP_NAME} \
  --ngpus 1 \
  --ncpu 6 \
  --num_pos_steps 4 \
  --dist_loss_ratio ${dist_loss_ratio}
```

## Evaluation

To evaluate `DimeNet++ + GeoTMI` on QM9, run:

```eval
task="ours"       # one of {dft,mmff,ours}
target="gap"      # one of {alpha,cv,gap,homo,lumo mu,r2,u0_atom,zpve}

python -u test.py \
  --data_dir ${YOUR_DOWNLOADED_DATA_DIRECTORY} \
  --batch_size 128 \
  --task ${task} \
  --target ${target} \
  --ngpus 1 \
  --ncpu 6 \
  --checkpoint_file ${YOUR_CHECKPOINT_FILE}
```

## Pre-trained Models

Pre-trained models are available in this directory.

```console
results/{dft,mmff,ours}/{alpha,cv,gap,homo,lumo mu,r2,u0_atom,zpve}/best.pt
```
