import h5py
import requests

from pathlib import Path
from h5py import File as H5File

from base import BaseDatasetGenerator


class VoxCeleb2DatasetGenerator(BaseDatasetGenerator):
    """
    VoxCeleb2 Dataset Generator
    ===========================
    Stores the VoxCeleb2 Dataset in a H5 File.
    Can be configured to store the dataset as Spectrograms
    or Raw Audio Waveforms.
    The Speaker Metadata is downloaded from the official
    website.
    https://mm.kaist.ac.kr/datasets/voxceleb/meta/vox2_meta.csv

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

    SPEAKER_METADATA_URI: str = (
        "https://mm.kaist.ac.kr/datasets/voxceleb/meta/vox2_meta.csv"
    )

    def init_metadata_dataset(self, h5_file: H5File) -> None:
        # Try to download Speaker Metadata
        try:
            response = requests.get(self.SPEAKER_METADATA_URI)
            content = response.content.decode("utf-8")
            metadata = []
            for speaker_metadata in content.splitlines()[1:]:
                speaker_metadata = [
                    meta.strip() for meta in speaker_metadata.split(",")
                ]
                if speaker_metadata[0] in self.speaker_map:
                    speaker_metadata = [
                        str(self.speaker_map[speaker_metadata[0]])
                    ] + speaker_metadata
                    metadata.append(speaker_metadata)
            h5_file.create_dataset(
                "speaker_metadata", data=metadata, dtype=h5py.string_dtype()
            )

        except:
            raise
            pass

        # Initialize Sample Metadata Dataset
        h5_file.create_dataset(
            "sample_metadata", shape=(self.n_samples, 3), dtype=h5py.string_dtype()
        )

    def store_metadata(self, h5_file: H5File, index: int, path: Path) -> None:
        sample_metadata = [
            self.get_speaker_name(path),
            self.get_video_name(path),
            self.get_utterance_name(path),
        ]
        h5_file["sample_metadata"][index] = sample_metadata

    def get_speaker_name(self, path: Path) -> str:
        return str(path).split("/")[-3]

    def get_video_name(self, path: Path) -> str:
        return str(path).split("/")[-2]

    def get_utterance_name(self, path: Path) -> str:
        return Path(str(path)).stem


if __name__ == "__main__":
    # exmaple usage
    # only works if you're using the devcontainer
    source_path = Path("/home/ubuntu/Documents/vt1code/voxceleb_trainer/data/voxceleb2")
    file_list_path = Path(
        "/home/ubuntu/Documents/vt1code/voxceleb_trainer/SpeakerVerification/lists/voxceleb2.txt"
    )
    dest_path_raw = Path(
        "/home/ubuntu/Documents/vt1code/voxceleb_trainer/data/h5_files/vox2.raw.h5"
    )
    dest_path_spec = Path(
        "/home/ubuntu/Documents/vt1code/voxceleb_trainer/data/h5_files/vox2.spec.h5"
    )
    if dest_path_raw.exists():
        dest_path_raw.unlink()
    if dest_path_spec.exists():
        dest_path_spec.unlink()

    generator = VoxCeleb2DatasetGenerator(
        source_path,
        file_list_path,
        dest_path_raw,
        dest_path_spec,
        extension="wav",
    )
    generator.generate_dataset()
