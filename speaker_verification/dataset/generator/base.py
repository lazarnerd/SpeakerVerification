from typing import Tuple, List
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torchaudio
import numpy as np
import os
import h5py

from h5py import File as H5File, Dataset as H5Dataset


class BaseDatasetGenerator(ABC):
    file_list: List[Path]
    n_samples: int

    h5_file: H5File
    h5_x_dataset: H5Dataset
    h5_x_ref_dataset: H5Dataset
    h5_y_dataset: H5Dataset

    def __init__(
        self,
        zip_path: Path,
        destination_path: Path,
        dataset_config: dict,
        chunk_size: int,
        extension: str,
    ) -> None:
        """
        Base class for dataset generators

        Args:
            source_path (Path):
                Path to source directory
            destination_path (Path):
                Path to destination dataset
            chunk_size (int):
                Chunk size for the x dataset
            extension (str):
                Extension of the files to load.

        Raises:
            FileNotFoundError:
                If the source path does not exist
            NotADirectoryError:
                If the source path is not a directory
            FileExistsError:
                If the destination path already exists
            TypeError:
                If the sample loading function does not return a numpy array for the sample or a list for the label
        """

        # Validate source path
        if not source_path.exists():
            raise FileNotFoundError(f"Source path {source_path} does not exist")
        elif source_path.is_file():
            raise NotADirectoryError(f"Source path {source_path} is not a directory")

        # Validate destination path
        if destination_path.exists():
            raise FileExistsError(f"Target dataset {destination_path} already exists")

        # Create destination path
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        # Get all files in source path with given extension
        self.file_list = sorted([path for path in source_path.rglob(f"*.{extension}")])

        # Validate sample loading
        sample, label = self.load_sample(self.file_list[0])
        if not isinstance(sample, np.ndarray):
            raise TypeError(
                f"Sample loading function must return a numpy array for the sample, not {type(sample)}"
            )
        if not isinstance(label, list):
            raise TypeError(
                f"Sample loading function must return a list for the label, not {type(label)}"
            )

        # Calculate the shapes of the datasets
        x_shape = (1,) + sample.shape[1:]
        x_max_shape = (None,) + sample.shape[1:]
        x_chunk_shape = (chunk_size,) + sample.shape[1:]

        y_shape = (len(self.file_list), len(label) + 2)

        # Create HDF5 file
        self.h5_file = H5File(destination_path, "w")
        self.h5_x_dataset = self.h5_file.create_dataset(
            "x",
            shape=x_shape,
            maxshape=x_max_shape,
            chunks=x_chunk_shape,
            dtype=sample.dtype,
        )

        self.h5_y_dataset = self.h5_file.create_dataset(
            "y",
            shape=y_shape,
            dtype=h5py.string_dtype(),
        )

    @abstractmethod
    def transform_sample(
        self, raw_audio: np.ndarray, path: Path
    ) -> Tuple[np.ndarray, List]:
        """
        Load a sample from a given path

        Args:
            path (Path): Path to sample file

        Returns:
            Tuple[np.ndarray, List]:
                Sample and label.
                Make sure that if your samples vary in length, you put that axis first.
                If not, make sure that the first axis shape is 1.
        """
        ...

    def generate(self) -> None:
        """
        Generate the dataset
        """
        start = 0
        # Generate the dataset
        for i, path in enumerate(self.file_list):
            # Load sample
            sample, label = self.load_sample(path)
            end = start + sample.shape[0]

            # Add sample to dataset
            self.h5_x_dataset.resize((end,) + sample.shape[1:])
            self.h5_x_dataset[start:end] = sample

            # Add sample reference to dataset
            self.h5_x_ref_dataset[i] = [start, end]

            # Add label to dataset
            label = [str(l) for l in label]
            self.h5_y_dataset[i] = label

            # Update start
            start = end

        # Flush dataset
        self.h5_file.flush()

        # Close dataset
        self.h5_file.close()
