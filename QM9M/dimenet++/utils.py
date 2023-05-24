import logging
import os
import sys
from collections import defaultdict, OrderedDict
from functools import wraps
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

UNIT_CONVERTS = OrderedDict(
    {
        "mu": 1.0,
        "alpha": 1.0,
        "homo": 27.2114,
        "lumo": 27.2114,
        "gap": 27.2114,
        "r2": 1,
        "u0_atom": 1 / 23.06,
        "cv": 1.0,
        "zpve": 27.2114,
    }
)


def get_indice(path: Path) -> List[int]:
    with path.open("r") as f:
        indice = [int(line.strip()) - 1 for line in f]
    return indice


def get_target_std(data_path: Path, idx_paths: List[Path], target: str):
    indice = []
    for idx_path in idx_paths:
        indice.extend(get_indice(idx_path))

    df = pd.read_csv(data_path)
    props = df[target][indice] * UNIT_CONVERTS[target]
    mean_ = mean(props)
    std_ = stdev(props)
    return mean_, std_


def _get_rank() -> int:
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def rank_zero_only(fn: Callable) -> Callable:
    """\
    Function that can be used as a decorator to enable a function/method
    being called only on rank 0. (from pytorch-lightning)
    """

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if getattr(rank_zero_only, "rank", _get_rank()) == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


@rank_zero_only
def initialize_logger(
    log_file: Optional[str] = None,
    log_file_level: int = logging.NOTSET,
    rotate: bool = False,
    mode: str = "w",
) -> logging.Logger:
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Set logging to stdout.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file:
        if rotate:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10
            )
        else:
            file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def read_labels(path: Path) -> List[str]:
    with path.open("r") as f:
        labels = [line.strip() for line in f]
    return labels


@rank_zero_only
def write_log(line: str, logger: logging.Logger, *args):
    if args:
        line += "\t".join([f"{arg:.5f}" for arg in args])
    logger.info(line)
    return


@rank_zero_only
def get_log_line(
    losses: List[Dict[str, List[float]]], epoch: Optional[int] = None
) -> str:
    labels = ["train", "val", "test"]
    if epoch is None:  # Initial logging
        line = "Epoch"
        for idx, loss in enumerate(losses):
            line += "\t"
            line += "\t".join([f"{labels[idx]}_{key}" for key in loss.keys()])
        line += "\tTrain MAE\tVal MAE\tTest MAE"
        line += "\tTrain Dist MAE\tVal Dist MAE\tTest Dist MAE\tTime"
    else:
        line = f"{epoch}"
        for loss in losses:
            line += "\t"
            line += "\t".join([f"{mean(loss[k]):.5f}" for k in loss.keys()])
    return line


@rank_zero_only
def parse_results(results: Dict[str, List[float]], stds: float):
    if not (pred := results.get("mmff_pred", None)):
        pred = results["pred"]
    pred = np.concatenate(pred) * stds
    target = np.concatenate(results["target"]) * stds
    mae = np.mean(np.abs(pred - target))
    return mae


@rank_zero_only
def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_losses: Dict[str, List[float]],
    val_losses: Dict[str, List[float]],
    test_losses: Dict[str, List[float]],
    path: Path,
) -> None:
    consume_prefix_in_state_dict_if_present(model.state_dict(), "module.")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_losses": test_losses,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return


@rank_zero_only
def gather(data: Any) -> Dict[str, List[Any]]:
    gather_t = [None for _ in range(dist.get_world_size())]
    dist.gather_object(data, gather_t, dst=0)

    dic = defaultdict(list)
    for scatter in gather_t:
        for k, v in scatter.items():
            if isinstance(v, list):
                dic[k].extend(v)
            elif isinstance(v, np.ndarray):
                dic[k].extend(v.tolist())

    return dic
