import yaml
from pathlib import Path

from speaker_verification.dataset.downloader import DatasetDownloader


class DatasetHandler:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lists_path = path / "lists"
        self.zip_path = path / "zip"
        self.h5_path = path / "h5"

        self.lists_path.mkdir(parents=True, exist_ok=True)
        self.zip_path.mkdir(parents=True, exist_ok=True)
        self.h5_path.mkdir(parents=True, exist_ok=True)

        dataset_config = self.lists_path / "dataset.yml"
        if not dataset_config.exists():
            raise FileNotFoundError(f"{dataset_config} not found")
        with open(dataset_config, "r") as f:
            self.dataset_config = yaml.safe_load(f)

    def download(self, username: str = None, password: str = None) -> None:
        DatasetDownloader(
            self.zip_path, self.dataset_config, username, password
        ).download()
