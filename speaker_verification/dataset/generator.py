import yaml

from pathlib import Path
from zipfile import ZipFile
import numpy as np


class DatasetGenerator:
    def __init__(self, zip_path: Path, output_path: Path, dataset_config: dict, transformation_config_file: Path, chunk_size: int=None):
        self.file_list = {}
        self.file_list["tar"] = {}
        self.file_list["zip"] = {}

        if "zip_files" in dataset_config:
            for dic in dataset_config["zip_files"]:
                path = zip_path / dic["name"]
                extension = dic["extension"]
                with ZipFile(path) as z:
                    files = z.namelist()
                    files = [Path(file) for file in files]
                    files = [file for file in files if file.suffix == extension]

                    self.file_list["zip"][path] = files

        with open(transformation_config_file, "r") as f:
            self.transformation_config = yaml.safe_load(f)

        self.target_file = output_path / f"{transformation_config_file.stem}.h5"



    def transform_sample(self, sample: np.ndarray, transformation_config: dict):
        ...

    def generate(self):
        for zip_path in self.file_list["zip"]:
            with ZipFile(zip_path) as z:
                for 
            for file_path in files:
                sample = self.load_zip_sample(zip_path, file_path)
                sample = self.transform_sample(sample, self.transformation_config)
                ...