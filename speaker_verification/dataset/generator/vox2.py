import numpy as np
from typing import Tuple, List

from pathlib import Path

from speaker_verification.dataset.config import Config
from speaker_verification.dataset.generator.utils import (
    download_fileparts,
    concat_files,
    unzip_files,
)
from speaker_verification.dataset.generator.base import BaseDatasetGenerator


class VoxCeleb2DatasetGenerator(BaseDatasetGenerator):
    """
    VoxCeleb2 Dataset Generator
    ===========================
    Downloads, extracts and stores the VoxCeleb2 Dataset in a H5 File.

    Dataset Details:
    ----------------

    |                               |           |
    |-------------------------------|:----------|
    | # Speakers:                   | 6,112     |
    | # male Speakers:              | 3,761     |
    | # female Speakers:            | 2,351     |
    | # videos:                     | 150,480   |
    | # hours:                      | 2,442     |
    | # utterance:                  | 1,128,246 |
    | Avg # videos per Speaker:     | 25        |
    | Avg # utterances per Speaker: | 185       |
    | Avg length of utterances [s]: | 7.8       |

    Dataset Sizes (Dev & Test):
    ---------------------------

    |         Stored as | Size      |
    |-------------------|:----------|
    |              Zip: | 74.6 GiB  |
    | Extracted audios: | 77.7 GiB  |
    |      H5 Waveform: | GiB       |
    |   H5 Spectrogram: | GiB       |
    """

    @classmethod
    def download(cls) -> None:
        """
        Downloads the VoxCeleb2 Dataset
        """
        config = Config()
        download_fileparts(
            fileparts_list=config.VOXCELEB2_FILEPARTS_LIST,
            target_dir=config.VOXCELEB2_ZIP_PATH,
            use_credentials=True,
        )

    @classmethod
    def concat(cls) -> None:
        """
        Concatenates the VoxCeleb2 Dataset
        """
        config = Config()
        concat_files(
            file_list=config.VOXCELEB2_CONCAT_LIST,
            zip_path=config.VOXCELEB2_ZIP_PATH,
        )

    @classmethod
    def unzip(cls) -> None:
        """
        Unzips the VoxCeleb2 Dataset
        """
        config = Config()
        unzip_files(
            zip_path=config.VOXCELEB2_ZIP_PATH,
            files_path=config.VOXCELEB2_FILES_PATH,
        )

    def load_sample(self, path: Path) -> Tuple[np.ndarray, List]:
        # return super().load_sample(path)
        pass


if __name__ == "__main__":
    VoxCeleb2DatasetGenerator.download()
    VoxCeleb2DatasetGenerator.concat()
    VoxCeleb2DatasetGenerator.unzip()
