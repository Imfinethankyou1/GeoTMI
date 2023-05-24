import os
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

import arguments
import utils
from data import ReorgDataset
from models.dimenet_pp import DimeNetPlusPlus
from train import eval_loop


def main_worker(args: ArgumentParser, stds: float):
    # Directory
    args.save_dir = args.checkpoint_file.parent / "results"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    filename = args.checkpoint_file.name.split(".")[0]

    # Logger
    logger = utils.initialize_logger(args.save_dir / "test.log")
    print_log = partial(utils.write_log, logger=logger)
    for arg, value in sorted(vars(args).items()):
        print_log(f"Argument {arg}: {value}")

    # Load model
    model = DimeNetPlusPlus(
        hidden_channels=args.hidden_channels,
        out_channels=args.n_outputs,
        num_blocks=args.num_blocks,
        int_emb_size=args.int_emb_size,
        basis_emb_size=args.basis_emb_size,
        out_emb_channels=args.out_emb_channels,
        num_spherical=args.num_spherical,
        num_radial=args.num_radial,
        cutoff=args.cutoff,
        envelope_exponent=args.envelope_exponent,
        num_before_skip=args.num_before_skip,
        num_after_skip=args.num_after_skip,
        num_output_layers=args.num_output_layers,
    )

    n_params = sum([param.numel() for param in model.parameters()])
    print_log(f"Number of parameters: {n_params}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.checkpoint_file.exists():
        checkpoint = torch.load(args.checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print_log(f"Load checkpoint from {args.checkpoint_file}")

    test_labels = utils.read_labels(args.label_dir / "test_labels_egnn.txt")
    test_data = ReorgDataset(test_labels, args.data_dir, args.target)
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.ncpu,
    )
    print_log(f"Test dataset length : {len(test_data)}")

    st = time.time()
    test_losses, test_results, test_dmaes = eval_loop(model, test_loader, device, args.task)
    elapsed = time.time() - st

    try:
        if args.task == "mmff":
            mmff_pred = np.concatenate(test_results.get("mmff_pred")) * stds
            target = np.concatenate(test_results.get("target")) * stds
            mmff_mae = np.mean(np.abs(mmff_pred - target))
            print_log("Test MMFF MAE\tTime")
            print_log(f"{mmff_mae:.5f}\t{elapsed:.5f}")
        elif args.task == "dft":
            pred = np.concatenate(test_results.get("pred")) * stds
            target = np.concatenate(test_results.get("target")) * stds
            mae = np.mean(np.abs(pred - target))
            print_log("Test DFT MAE\tTime")
            print_log(f"{mae:.5f}\t{elapsed:.5f}")
        elif args.task == "ours":
            mmff_pred = np.concatenate(test_results.get("mmff_pred")) * stds
            pred = np.concatenate(test_results.get("pred")) * stds
            target = np.concatenate(test_results.get("target")) * stds
            mmff_mae = np.mean(np.abs(mmff_pred - target))
            mae = np.mean(np.abs(pred - target))
            print_log("Test MMFF MAE\tTest DFT MAE\tTime")
            print_log(f"{mmff_mae:.5f}\t{mae:.5f}\t{elapsed:.5f}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args = arguments.get_args()
    seed_everything(args.seed)

    label_dir = Path("/home/share/DATA/khs_ICML/qm9_data/CV_data")
    train_label_file = label_dir / "train_labels_egnn.txt"
    val_label_file = label_dir / "val_labels_egnn.txt"
    total_data_path = "/home/share/DATA/khs_ICML/qm9_data/qm2mmff_label.txt"

    _, stds = utils.get_target_std(
        total_data_path, [train_label_file, val_label_file], args.target
    )

    main_worker(args, stds)
