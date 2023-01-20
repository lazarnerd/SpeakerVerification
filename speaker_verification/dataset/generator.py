import yaml
import tarfile

from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torchaudio
import torch
import ffmpeg
import soundfile
from typing import IO, List, Tuple, Generator
from torch import Tensor
from rich.progress import Progress

import h5py


class DatasetGenerator:
    def __init__(
        self,
        zip_path: Path,
        output_path: Path,
        dataset_config: dict,
        transformation_config_file: Path,
        sample_rate: int = 16000,
        sample_duration: float = 0,
    ):
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.min_length = int(self.sample_duration * self.sample_rate)

        self.file_list = {}
        self.file_list["tar"] = {}
        self.file_list["zip"] = {}

        self.n_samples = 0

        if "zip_files" in dataset_config:
            for dic in dataset_config["zip_files"]:
                path = zip_path / dic["name"]
                extension = dic["extension"]
                extension = extension if extension[0] == "." else "." + extension
                with ZipFile(path) as z:
                    files = z.namelist()
                    files = [Path(file) for file in files]
                    files = [file for file in files if file.suffix == extension]
                    self.n_samples += len(files)
                    self.file_list["zip"][path] = files

        if "tar_files" in dataset_config:
            for dic in dataset_config["tar_files"]:
                path = zip_path / dic["name"]
                extension = dic["extension"]
                extension = extension if extension[0] == "." else "." + extension
                with tarfile.open(path) as t:
                    files = t.getnames()
                    files = [Path(file) for file in files]
                    files = [file for file in files if file.suffix == extension]
                    self.n_samples += len(files)
                    self.file_list["tar"][path] = files

        with open(transformation_config_file, "r") as f:
            self.transformation_config = yaml.safe_load(f)

        self.tmp_in_file = output_path / "tmp_in"
        self.tmp_out_file = output_path / "tmp_out"

        self.target_file = (
            output_path
            / f"{transformation_config_file.stem}__D_{self.sample_duration:.2f}s.hdf5"
        )

    def get_sample_name(self, sample_path: Path) -> str:
        audio_name = sample_path.stem + ".wav"
        video_name = sample_path.parent.name
        speaker_name = sample_path.parent.parent.name
        return f"{speaker_name}/{video_name}/{audio_name}"

    def transform_sample(self, sample: Tensor) -> Tensor:
        if self.transformation_config["transform"] == "RAW":
            return sample
        elif self.transformation_config["transform"] == "SPEC":
            options = self.transformation_config["options"]
            n_fft = int(options["n_fft"])
            win_length = int(options["window_length"] * self.sample_rate)
            hop_length = int(options["hop_length"] * self.sample_rate)
            sample = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=torch.hamming_window,
            )(sample)
            return sample.T
        raise ValueError(
            f"Unknown transformation: {self.transformation_config['transform']}"
        )

    def process_sample(
        self, sample: IO[bytes], sample_path: Path
    ) -> Tuple[Tensor, List]:
        # Load sample
        with open(self.tmp_in_file, "wb") as f:
            f.write(sample.read())
        audio_stream = ffmpeg.input(str(self.tmp_in_file)).audio
        output = ffmpeg.output(
            audio_stream,
            str(self.tmp_out_file),
            format="wav",
            acodec="pcm_s16le",
            ar=self.sample_rate,
            ac=1,
        ).overwrite_output()
        ffmpeg.run(output, quiet=True)

        sample, sample_rate = soundfile.read(str(self.tmp_out_file))

        # Pad if necessary
        if sample.shape[0] < self.min_length:
            sample = np.pad(sample, self.min_length - sample.shape[0], "wrap")
        # Transform sample
        sample = self.transform_sample(torch.from_numpy(sample))
        # Get sample name
        sample_name = self.get_sample_name(sample_path)
        return sample, sample_name

    def load_samples(self) -> Generator[Tuple[Tensor, str], None, None]:
        try:
            for zip_path in self.file_list["zip"]:
                with ZipFile(zip_path) as zip_file:
                    for sample_path in self.file_list["zip"][zip_path]:
                        with zip_file.open(str(sample_path)) as sample_file:
                            yield self.process_sample(sample_file, sample_path)
            for tar_path in self.file_list["tar"]:
                with tarfile.open(tar_path) as tar_file:
                    for sample_path in self.file_list["tar"][tar_path]:
                        with tar_file.extractfile(str(sample_path)) as sample_file:
                            yield self.process_sample(sample_file, sample_path)
        except GeneratorExit:
            pass

    def generate(self):
        if self.target_file.exists():
            print(f"Dataset {self.target_file} already exists. Skipping.\n")
            return
        try:
            # generate test sample
            raw = np.random.rand(self.min_length)
            sample = self.transform_sample(torch.from_numpy(raw))
            sample = sample.numpy()

            with Progress() as progress:
                task = progress.add_task(
                    f"Generating: {self.target_file.name}", total=self.n_samples
                )

                # Generate HDF5 file
                with h5py.File(self.target_file, "w") as h5_file:
                    x_shape = (1,) + sample.shape[1:]
                    x_max_shape = (None,) + sample.shape[1:]

                    x_dataset = h5_file.create_dataset(
                        "x",
                        x_shape,
                        maxshape=x_max_shape,
                        chunks=sample.shape,
                        dtype=sample.dtype,
                    )

                    start = 0
                    y = []
                    for sample, sample_name in self.load_samples():
                        end = start + sample.shape[0]
                        x_dataset.resize(end, axis=0)
                        x_dataset[start:end] = sample.numpy()

                        y.append((sample_name, start, end))
                        start = end
                        progress.advance(task)

                    dtype = [
                        ("sample_name", h5py.string_dtype()),
                        ("start", "uint64"),
                        ("end", "uint64"),
                    ]

                    y = np.rec.array(y, dtype=dtype)
                    h5_file.create_dataset("y", data=y, dtype=dtype)
        except:
            if self.target_file.exists():
                self.target_file.unlink()
            raise
        finally:
            if self.tmp_in_file.exists():
                self.tmp_in_file.unlink()
            if self.tmp_out_file.exists():
                self.tmp_out_file.unlink()
