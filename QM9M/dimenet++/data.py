import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from utils import UNIT_CONVERTS


class ReorgDataset(Dataset):
    def __init__(self, labels: List[str], data_dir: Path, target: str):
        """Base Dataset class for delfta"""
        self.labels = labels
        target_idx = list(UNIT_CONVERTS.keys()).index(target)

        graph_data_dict = {}
        for label in self.labels:
            try:
                name = f"{data_dir}/{label}.npz"
                graph_data = np.load(name, allow_pickle=True)
            except FileNotFoundError:
                name = f"{data_dir}/gdb_{label}.npz"
                graph_data = np.load(name, allow_pickle=True)
            graph_data.target = graph_data.target[:, target_idx]
            graph_data_dict[label] = graph_data
        self.graph_data_dict = graph_data_dict

    def __getitem__(self, idx: int) -> Data:
        label = self.labels[idx]
        graph_data = self.graph_data_dict[label]
        return graph_data

    def __len__(self) -> int:
        return len(self.labels)
