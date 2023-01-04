import h5py
import requests

from pathlib import Path
from h5py import File as H5File

from base import BaseDatasetGenerator


class VoxCeleb1DatasetGenerator(BaseDatasetGenerator):
    def init_metadata_dataset(self, h5_file: H5File) -> None:
        # TODO
        pass

    def store_metadata(self, h5_file: H5File, index: int, path: Path) -> None:
        # TODO
        pass

    def get_speaker_name(self, path: Path) -> str:
        # TODO
        pass
