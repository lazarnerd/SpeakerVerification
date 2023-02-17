from pathlib import Path

ROOT = Path("/workspaces/SpeakerVerification")
DATA = ROOT / "data"

VOXCELEB1 = DATA / "voxceleb1"
VOXCELEB2 = DATA / "voxceleb2"

VOXCELEB1_DEEPLAKE = VOXCELEB1 / "deeplake" / "VoxCeleb1"
VOXCELEB2_DEEPLAKE = VOXCELEB2 / "deeplake" / "VoxCeleb2"

VOXCELEB1_ZIP_TEST = VOXCELEB1 / "zip" / "vox1_test.zip"
VOXCELEB1_ZIP_TRAIN = VOXCELEB1 / "zip" / "vox1_dev.zip"
VOXCELEB2_ZIP_TEST = VOXCELEB2 / "zip" / "vox2_test.zip"
VOXCELEB2_ZIP_TRAIN = VOXCELEB2 / "zip" / "vox2_dev.zip"

VOXCELEB1_EXTRACTED_TEST = VOXCELEB1 / "extracted" / "test"
VOXCELEB1_EXTRACTED_TRAIN = VOXCELEB1 / "extracted" / "train"
VOXCELEB2_EXTRACTED_TEST = VOXCELEB2 / "extracted" / "test"
VOXCELEB2_EXTRACTED_TRAIN = VOXCELEB2 / "extracted" / "train"


import deeplake
from pathlib import Path


def generate(train: Path, test: Path, dest: Path, header: list, meta: dict):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dataset = deeplake.empty(dest, overwrite=True)
    with dataset:
        dataset.create_tensor(
            "Audio",
            htype="audio",
            sample_compression="wav",
        )

        dataset.create_tensor(
            "Speaker ID",
            htype="class_label",
        )

        dataset.create_tensor(
            "Video",
            htype="class_label",
        )

        dataset.create_tensor("Sample Name", htype="text")

        for key in header:
            dataset.create_tensor(
                key,
                htype="class_label",
            )
    test_audios = [audio for audio in test.glob("**/*.wav")]
    train_audios = [audio for audio in train.glob("**/*.wav")]
    audios = [*test_audios, *train_audios]
    for audio in audios:
        video = audio.parent.name
        speaker = audio.parent.parent.name
        sample_name = f"{speaker}/{video}/{audio.name}"
        with dataset:
            sample = {
                "Audio": deeplake.read(audio),
                "Speaker ID": speaker,
                "Video": video,
                "Sample Name": sample_name,
            }
            for key in meta[speaker]:
                sample[key] = meta[speaker][key]
            dataset.append(sample)


with open(VOXCELEB2 / "lists" / "meta.txt", "r") as f:
    lines = f.readlines()
header = lines[0].split(" ,")
header = header[1:]
header = [h.strip() for h in header]
meta = {}
for line in lines[1:]:
    try:
        line = line.split(" ,")
        spkr = line[0]
        line = line[1:]
        meta[spkr] = {header[i]: line[i].strip() for i in range(len(header))}
    except:
        pass
generate(
    VOXCELEB2_EXTRACTED_TRAIN,
    VOXCELEB2_EXTRACTED_TEST,
    VOXCELEB2_DEEPLAKE,
    header,
    meta,
)
