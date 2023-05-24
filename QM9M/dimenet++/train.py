import os
import time
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_ema import ExponentialMovingAverage
from torch_geometric import seed_everything
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import radius_graph

import arguments
import utils
from data import ReorgDataset
from models.dimenet_pp import DimeNetPlusPlus


def step(
    model: nn.Module,
    batch: Batch,
    task: str,
    save_dist_results: bool = False,
    num_pos_steps: int = 4,
    dist_loss_ratio: float = 0.1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    target = batch.target
    pred = None
    mmff_pred = None
    loss = 0.0
    y_mae = 0.0
    mmff_y_mae = 0.0
    mmff_dist_losses = []

    mmff_target_dists = []
    mmff_pred_dists = []

    if task == "dft":
        edge_index = radius_graph(batch.coords_ex, r=5.0, batch=batch.batch)
        pred, _ = model(batch.atomids, batch.coords_ex, edge_index, batch.batch)
        pred = pred.squeeze(-1)
        y_mae = F.l1_loss(pred, target)
        loss = loss + y_mae

        pred = pred.cpu().detach().numpy()

    elif task == "mmff":
        edge_index = radius_graph(batch.coords, r=5.0, batch=batch.batch)
        mmff_pred, _ = model(batch.atomids, batch.coords, edge_index, batch.batch)
        mmff_pred = mmff_pred.squeeze(-1)
        mmff_y_mae = F.l1_loss(mmff_pred, target)
        loss = loss + mmff_y_mae

        mmff_pred = mmff_pred.cpu().detach().numpy()

    elif task == "ours":
        mmff_pos = batch.coords
        mmff_edge_index = radius_graph(mmff_pos, r=5.0, batch=batch.batch)
        mmff_vecs = mmff_pos[mmff_edge_index[0]] - mmff_pos[mmff_edge_index[1]]
        mmff_dist = (mmff_vecs**2).sum(dim=-1).sqrt()
        mmff_pred, mmff_dist_history = model(
            batch.atomids, mmff_pos, mmff_edge_index, batch.batch, True
        )
        mmff_pred = mmff_pred.squeeze(-1)
        mmff_y_mae = F.l1_loss(mmff_pred, target)
        loss = loss + mmff_y_mae

        pos = batch.coords_ex
        edge_index = radius_graph(pos, r=5.0, batch=batch.batch)
        vecs = pos[mmff_edge_index[0]] - pos[mmff_edge_index[1]]
        dist = (vecs**2).sum(dim=-1).sqrt()
        pred, _ = model(batch.atomids, pos, edge_index, batch.batch, False)
        pred = pred.squeeze(-1)
        y_mae = F.l1_loss(pred, target)
        loss = loss + y_mae

        for idx, mmff_dist_ in enumerate(mmff_dist_history):
            if idx < num_pos_steps:
                mmff_target_dist = (
                    mmff_dist + (dist - mmff_dist) * (idx + 1) / num_pos_steps
                )
                mmff_dist_loss = F.l1_loss(mmff_dist_, mmff_target_dist)
            else:
                mmff_target_dist = dist
                mmff_dist_loss = F.l1_loss(mmff_dist_, mmff_target_dist)
            mmff_dist_losses.append(mmff_dist_loss)

            loss = loss + mmff_dist_loss * dist_loss_ratio

            mmff_target_dists.append(mmff_target_dist.cpu().detach().numpy())
            mmff_pred_dists.append(mmff_dist_.cpu().detach().numpy())

        pred = pred.cpu().detach().numpy()
        mmff_pred = mmff_pred.cpu().detach().numpy()

    target = target.cpu().detach().numpy()

    losses = dict(
        total_loss=loss,
        y_mae=y_mae,
        mmff_y_mae=mmff_y_mae,
    )
    dmaes = defaultdict(int)
    for idx, mmff_dist_loss in enumerate(mmff_dist_losses):
        losses[f"mmff_dist_loss_{idx}"] = mmff_dist_loss
        if idx == num_pos_steps - 1:
            dmaes["total_dist_mae"] += np.sum(
                np.abs(mmff_pred_dists[idx] - mmff_target_dists[idx])
            )
            dmaes["dist_count"] += len(mmff_pred_dists[idx])

    results = dict(
        pred=pred,
        mmff_pred=mmff_pred,
        target=target,
    )
    if save_dist_results:
        idx = num_pos_steps - 1
        results[f"mmff_target_dist_{idx}"] = mmff_target_dists[idx]
        results[f"mmff_pred_dist_{idx}"] = mmff_pred_dists[idx]

    return losses, results, dmaes


def train_loop(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    optimizer: optim.Optimizer,
    ema: ExponentialMovingAverage,
    save_dist_results: bool = False,
    num_pos_steps: int = 0,
    dist_loss_ratio: float = 0.1,
):
    model.train()

    losses = defaultdict(list)
    dmaes = defaultdict(list)
    results = defaultdict(list)

    for g_batch in loader:
        optimizer.zero_grad()
        g_batch = g_batch.to(device)
        step_losses, step_results, step_dmaes = step(
            model, g_batch, task, save_dist_results, num_pos_steps, dist_loss_ratio
        )

        step_losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        ema.update()

        for key, value in step_losses.items():
            if isinstance(value, int | float) and value == 0:
                continue
            losses[key].append(value.item())
        for key, value in step_dmaes.items():
            dmaes[key].append(value)
        for key, value in step_results.items():
            if value is None:
                continue
            results[key].append(value)

    return OrderedDict(losses), OrderedDict(results), OrderedDict(dmaes)


@torch.no_grad()
def eval_loop(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    save_dist_results: bool = False,
    num_pos_steps: int = 0,
    dist_loss_ratio: float = 0.1,
):
    model.eval()

    losses = defaultdict(list)
    dmaes = defaultdict(list)
    results = defaultdict(list)

    for g_batch in loader:
        g_batch = g_batch.to(device)
        step_losses, step_results, step_dmaes = step(
            model, g_batch, task, save_dist_results, num_pos_steps, dist_loss_ratio
        )

        for key, value in step_losses.items():
            if isinstance(value, int | float) and value == 0:
                continue
            losses[key].append(value.item())
        for key, value in step_dmaes.items():
            dmaes[key].append(value)
        for key, value in step_results.items():
            if value is None:
                continue
            results[key].append(value)

    return OrderedDict(losses), OrderedDict(results), OrderedDict(dmaes)


def main_worker(args: ArgumentParser, stds: float):
    # Directory
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Logger
    logger = utils.initialize_logger(args.save_dir / "train.log")
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

    initial_epoch = 0
    if args.checkpoint_file is not None:
        checkpoint = torch.load(args.checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print_log(f"Load checkpoint from {args.checkpoint_file}")
        initial_epoch = checkpoint["epoch"]

    train_labels = utils.read_labels(args.label_dir / "train_labels_egnn.txt")
    val_labels = utils.read_labels(args.label_dir / "val_labels_egnn.txt")
    test_labels = utils.read_labels(args.label_dir / "test_labels_egnn.txt")

    train_data = ReorgDataset(train_labels, args.data_dir, args.target)
    val_data = ReorgDataset(val_labels, args.data_dir, args.target)
    test_data = ReorgDataset(test_labels, args.data_dir, args.target)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.ncpu,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.ncpu,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.ncpu,
    )

    print_log(f"Train dataset length : {len(train_data)}")
    print_log(f"Val dataset length : {len(val_data)}")
    print_log(f"Test dataset length : {len(test_data)}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    ema = ExponentialMovingAverage(
        model.parameters(), decay=0.999, use_num_updates=False
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=20, verbose=True
    )

    min_mae = 1e10

    for epoch in range(initial_epoch, args.epochs):
        st = time.time()
        train_losses, train_results, train_dmaes = train_loop(
            model,
            train_loader,
            device,
            args.task,
            optimizer,
            ema,
            save_dist_results=args.save_dist_results,
            num_pos_steps=args.num_pos_steps,
            dist_loss_ratio=args.dist_loss_ratio,
        )
        val_losses, val_results, val_dmaes = eval_loop(
            model,
            val_loader,
            device,
            args.task,
            save_dist_results=args.save_dist_results,
            num_pos_steps=args.num_pos_steps,
            dist_loss_ratio=args.dist_loss_ratio,
        )
        test_losses, test_results, test_dmaes = eval_loop(
            model,
            test_loader,
            device,
            args.task,
            save_dist_results=args.save_dist_results,
            num_pos_steps=args.num_pos_steps,
            dist_loss_ratio=args.dist_loss_ratio,
        )
        elapsed = time.time() - st

        if epoch == 0:
            line = utils.get_log_line([train_losses, val_losses, test_losses])
            print_log(line)

        line = utils.get_log_line([train_losses, val_losses, test_losses], epoch)
        train_mae = utils.parse_results(train_results, stds)
        val_mae = utils.parse_results(val_results, stds)
        test_mae = utils.parse_results(test_results, stds)
        scheduler.step(val_mae)

        line += f"\t{train_mae:.5f}\t{val_mae:.5f}\t{test_mae:.5f}"
        try:
            train_dmae = sum(train_dmaes["total_dist_mae"]) / sum(train_dmaes["dist_count"])
            val_dmae = sum(val_dmaes["total_dist_mae"]) / sum(val_dmaes["dist_count"])
            test_dmae = sum(test_dmaes["total_dist_mae"]) / sum(test_dmaes["dist_count"])
            line += f"\t{train_dmae:.5f}\t{val_dmae:.5f}\t{test_dmae:.5f}\t{elapsed:.5f}"
        except:
            line += f"\t{elapsed:.5f}"
        print_log(line)

        if val_mae < min_mae:
            min_mae = val_mae
            utils.save_model(
                model,
                optimizer,
                epoch,
                train_losses,
                val_losses,
                test_losses,
                args.save_dir / "checkpoints" / f"save_{epoch}.pt",
            )


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
