from pathlib import Path
from threading import Thread
from queue import Queue

from h5py import File

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, h5_file_path: Path):
        self.h5_file_path = h5_file_path
        with File(self.h5_file_path, "r") as f:
            self.labels = f["labels"][:]

    def init_worker(self, worker_id):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


import torch
from torch.utils.data import Dataset
from torch.multiprocessing import Manager, Process
