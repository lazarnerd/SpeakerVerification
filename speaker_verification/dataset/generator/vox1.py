import numpy as np
from typing import Tuple, List

from pathlib import Path

from speaker_verification.dataset.config import Config
from speaker_verification.dataset.generator.utils import (
    process_fileparts,
    concat_files,
    unzip_files,
)
from speaker_verification.dataset.generator.base import BaseDatasetGenerator


class VoxCeleb1DatasetGenerator(BaseDatasetGenerator):
    """
    VoxCeleb1 Dataset Generator
    ===========================
    Downloads, extracts and stores the VoxCeleb1 Dataset in a H5 File.

    Dataset Details:
    ----------------

    |                               |         |
    |-------------------------------|:--------|
    | # Speakers:                   | 1,251   |
    | # male Speakers:              | 690     |
    | # female Speakers:            | 561     |
    | # videos:                     | 22,496  |
    | # hours:                      | 352     |
    | # utterance:                  | 153,516 |
    | Avg # videos per Speaker:     | 18      |
    | Avg # utterances per Speaker: | 116     |
    | Avg length of utterances [s]: | 8.2     |


    Dataset Sizes (Dev & Test):
    ---------------------------

    |         Stored as | Size      |
    |-------------------|:----------|
    |              Zip: | 31.4 GiB  |
    | Extracted audios: | 38.1 GiB  |
    |      H5 Waveform: | 75.5 GiB  |
    |   H5 Spectrogram: | 121.4 GiB |
    """

    @classmethod
    def download(cls) -> None:
        """
        Downloads the VoxCeleb1 Dataset
        """
        config = Config()
        process_fileparts(
            fileparts_list=config.VOXCELEB1_FILEPARTS_LIST,
            target_dir=config.VOXCELEB1_ZIP_PATH,
            use_credentials=True,
        )

    @classmethod
    def concat(cls) -> None:
        """
        Concatenates the VoxCeleb1 Dataset
        """
        config = Config()
        concat_files(
            file_list=config.VOXCELEB1_CONCAT_LIST,
            zip_path=config.VOXCELEB1_ZIP_PATH,
        )

    @classmethod
    def unzip(cls) -> None:
        """
        Unzips the VoxCeleb1 Dataset
        """
        config = Config()
        unzip_files(
            zip_path=config.VOXCELEB1_ZIP_PATH,
            files_path=config.VOXCELEB1_FILES_PATH,
        )

    def load_sample(self, path: Path) -> Tuple[np.ndarray, List]:
        # return super().load_sample(path)
        pass
