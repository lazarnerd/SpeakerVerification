from pathlib import Path
from threading import Thread
from queue import Queue

from h5py import File

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, dataset_path: Path, dataset_config: Path):
        self.h5_file_path = h5_file_path
        with File(self.h5_file_path, "r") as f:
            labels = f["labels"][:]
        

    def init_worker(self, worker_id):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

