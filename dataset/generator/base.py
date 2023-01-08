from typing import Tuple, List, Union
from abc import ABC, abstractmethod
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

import glob
import librosa
import torch
import torchaudio
import numpy as np
import os

from tqdm import tqdm
from h5py import File as H5File
from h5py import Dataset as H5Dataset


class BaseDatasetGenerator(ABC):
    sample_rate: int
    store_raw: bool

    file_list: List[Path]
    n_samples: int

    speakers: List[str]
    speaker_map: dict
    n_speakers: int

    n_freq_bins: int

    h5_raw_file: H5File
    h5_raw_x: H5Dataset
    h5_raw_y: H5Dataset

    h5_spectrogram_file: H5File
    h5_spectrogram_x: H5Dataset
    h5_spectrogram_y: H5Dataset

    def testit(self, test: Union[str, List[str]]):
        """_summary_

        Args:
            test (Union[str, List[str]]): _description_
        """

    def __init__(
        self,
        source_path: Path,
        file_list_path: Path,
        raw_destination_path: Path,
        spectrogram_destination_path: Path,
        store_raw: bool = True,
        extension: str = "wav",
        sample_rate: int = 16000,
        chunk_length: float = 4.0,
        n_fft: int = 512,
        window_length: float = 0.025,
        hop_length: float = 0.01,
    ) -> None:
        """Abstract class for generating h5 datasets of audio files.

        Args:
            source_path (Path):
                Source path containing the audio files.

            file_list_path (Path):
                File list path containing the file names of audio files.

            raw_destination_path (Path):
                Path of target h5 dataset file for raw audio waveforms.

            spectrogram_destination_path (Path):
                Path of target h5 dataset file for spectrograms.

            store_raw (bool, optional):
                Whether to store the raw audio waveforms in a h5 as well or not.
                Defaults to true.

            extension (str, optional):
                Audio file extension to look for.
                Defaults to "wav".

            sample_rate (int, optional):
                Sample rate to use for reading & conversion.
                Defaults to 16000.

            chunk_length (float, optional):
                Chunk size to use for the h5 dataset in seconds.
                Best to use 2x the duration of the sample you will use for training/testing.
                Defaults to 4.0.

            n_fft (int, optional):
                Size of the FFT to use for the STFT.
                Defaults to 512.

            window_length (float, optional):
                Duration of a window in seconds for the STFT.
                Defaults to 0.025.

            hop_length (float, optional):
                Length of hop in seconds between STFT windows.
                Defaults to 0.01.

        Raises:
            FileNotFoundError:
                If source_path does not exist.
            NotADirectoryError:
                If source_path is not a directory.
        """
        self.sample_rate = sample_rate
        self.store_raw = store_raw

        # Validate source path
        if not source_path.exists():
            raise FileNotFoundError(f"Source path {source_path} does not exist")
        elif source_path.is_file():
            raise NotADirectoryError(f"Source path {source_path} is not a directory")
        
        # Validate destination path
        if store_raw and raw_destination_path.exists():
            raise FileExistsError(
                f"Target dataset for raw audio waveforms {raw_destination_path} already exists"
            )
        if spectrogram_destination_path.exists():
            raise FileExistsError(
                f"Target dataset for spectrograms {spectrogram_destination_path} already exists"
            )

        f = open(file_list_path, "w")
        for file in glob.iglob(str(source_path)+"/*/*/*.wav", recursive=True):
            data = file.split("/")
            speaker_id = data[-3]
            filename = data[-3]+"/"+data[-2]+"/"+data[-1]
            f.write(speaker_id+" "+filename+"\n")
        f.close()
        os.system(f"sort {file_list_path} -o {file_list_path}")

        # Get all files in source path
        f = open(file_list_path,"r")
        l = []
        for line in f.readlines():
            l.append(str(source_path)+"/"+line.split()[1].replace("\n",""))
        self.file_list = l
        f.close()
         
        self.n_samples = len(self.file_list)
        # Get all speakers and map them to an index
        self.speakers = list(
            set(list([self.get_speaker_name(path) for path in self.file_list]))
        )
        self.speakers.sort()
        self.n_speakers = len(self.speakers)
        self.speaker_map = {speaker: i for i, speaker in enumerate(self.speakers)}

        # Initialize torch function to convert audio to spectrogram
        window_length = int(window_length * sample_rate)
        hop_length = int(hop_length * sample_rate)
        self.convert_to_stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=window_length,
            hop_length=hop_length,
            window_fn=torch.hamming_window,
        )

        # Validate sample loading
        #duration, raw_audio, spectrogram = self.read_sample(self.file_list[0])
        duration, raw_audio = self.read_sample(self.file_list[0])
        raw_audio_chunk_size = (int(self.sample_rate * chunk_length),)
        """ self.n_freq_bins = spectrogram.shape[1]
        raw_audio_chunk_size = (int(self.sample_rate * chunk_length),)
        spectrogram_chunk_size = (
            int(spectrogram.shape[0] * chunk_length / duration),
            self.n_freq_bins,
        ) """

        # Initialize h5 datasets
        if self.store_raw:
            self.h5_raw_file = H5File(raw_destination_path, "w-")
            self.h5_raw_x = self.h5_raw_file.create_dataset(
                "x",
                shape=(1,),
                maxshape=(None,),
                dtype=raw_audio.dtype,
                chunks=raw_audio_chunk_size,
            )
            self.h5_raw_y = self.h5_raw_file.create_dataset(
                "y", shape=(self.n_samples, 3), dtype=np.int64, chunks=True
            )
            self.init_metadata_dataset(self.h5_raw_file)

        """ self.h5_spectrogram_file = H5File(spectrogram_destination_path, "w-")
        self.h5_spectrogram_x = self.h5_spectrogram_file.create_dataset(
            "x",
            shape=(1, self.n_freq_bins),
            maxshape=(None, self.n_freq_bins),
            dtype=spectrogram.dtype,
            chunks=spectrogram_chunk_size,
        )
        self.h5_spectrogram_y = self.h5_spectrogram_file.create_dataset(
            "y", shape=(self.n_samples, 3), dtype=np.int64, chunks=True
        ) 
        self.init_metadata_dataset(self.h5_spectrogram_file)
        """

    def read_sample(self, path: Path) -> Tuple[float, np.ndarray, np.ndarray]:
        """Reads a sample from the given path.
        Returns the duration in seconds, the waveform and the spectrogram of the audio file.

        Args:
            path (Path): Audio file path

        Returns:
            Tuple[float, np.ndarray, np.ndarray]:
                [Duration, waveform, spectrogram]
                Duration in seconds
                Waveform has shape (time,)
                Spectrogram has shape (time, n_freq)
        """
        raw_audio, _ = librosa.load(
            path,
            sr=self.sample_rate,
            mono=True,
        )
        #spectrogram = self.convert_to_stft(torch.from_numpy(raw_audio)).numpy()
        return raw_audio.shape[0] / self.sample_rate, raw_audio#, spectrogram.T

    @abstractmethod
    def get_speaker_name(self, path: Path) -> str:
        """Returns the speaker name of the utterance at the given path.

        Args:
            path (Path): Audio file path

        Returns:
            str: Speaker name
        """
        ...

    @abstractmethod
    def init_metadata_dataset(self, h5_file: H5File) -> None:
        """Initializes the metadata dataset for the given h5 file.
        IMPORTANT: Do not store the dataset as a property!

        Args:
            h5_file (H5File):
                h5 file to initialize the metadata dataset for
        """
        ...

    @abstractmethod
    def store_metadata(self, h5_file: H5File, index: int, path: Path) -> None:
        """Stores the metadata for the given file in the given h5 file.

        Args:
            h5_file (H5File):
                h5 file to store the metadata in
            index (int):
                Index of the metadata in the dataset
            path (Path):
                Path of the file to store the metadata for
        """
        ...

    def generate_dataset(self) -> None:
        """
        Generates the h5 datasets.
        """
        raw_index = 0
        spec_index = 0

        # Store data
        for i, path in tqdm(
            enumerate(self.file_list),
            ncols=100,
            total=self.n_samples,
            desc="Generating dataset",
            unit_scale=True,
            ascii=True,
        ):
            speaker = self.get_speaker_name(path)
            speaker_id = self.speaker_map[speaker]
            #duration, raw_audio, spectrogram = self.read_sample(path)
            duration, raw_audio = self.read_sample(path)
            if self.store_raw:
                start = raw_index
                end = raw_index + raw_audio.shape[0]
                self.h5_raw_y[i, :] = np.array([start, end, speaker_id])
                self.h5_raw_x.resize((end,))
                self.h5_raw_x[start:end] = raw_audio
                self.store_metadata(self.h5_raw_file, i, path)
                raw_index = end
            """ start = spec_index
            end = spec_index + spectrogram.shape[0]
            self.h5_spectrogram_y[i, :] = np.array([speaker_id, start, end])
            self.h5_spectrogram_x.resize((end, self.n_freq_bins))
            self.h5_spectrogram_x[start:end, :] = spectrogram
            self.store_metadata(self.h5_spectrogram_file, i, path)
            spec_index = end """

        # Close h5 files
        if self.store_raw:
            self.h5_raw_file.close()
        #self.h5_spectrogram_file.close()
