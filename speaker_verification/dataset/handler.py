import yaml
from pathlib import Path

from speaker_verification.dataset.downloader import DatasetDownloader
from speaker_verification.dataset.generator import DatasetGenerator


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

    def generate(
        self,
        transformation_config_file: Path,
        sample_rate: int = 16000,
        sample_duration: float = 0.0,
    ) -> None:
        generator = DatasetGenerator(
            zip_path=self.zip_path,
            output_path=self.h5_path,
            dataset_config=self.dataset_config,
            transformation_config_file=transformation_config_file,
            sample_rate=sample_rate,
            sample_duration=sample_duration,
        )
        generator.generate()
