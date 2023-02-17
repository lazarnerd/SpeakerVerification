from pathlib import Path

from speaker_verification.dataset.config import Secrets
from speaker_verification.dataset.handler import DatasetHandler


if __name__ == "__main__":

    print("\033[H\033[J", end="")
    secrets = Secrets()
    handler = DatasetHandler(Path("data/voxceleb1"))
    # handler.download(secrets.USERNAME, secrets.PASSWORD)
    handler.generate(
        transformation_config_file=Path(
            "configs/dataset/MEL__F_512__M_40__W_0.025s__H_0.01s.yml"
        ),
        sample_duration=4.0,
        sample_rate=16000,
    )

    handler = DatasetHandler(Path("data/voxceleb2"))
    # handler.download(secrets.USERNAME, secrets.PASSWORD)
    handler.generate(
        transformation_config_file=Path(
            "configs/dataset/MEL__F_512__M_40__W_0.025s__H_0.01s.yml"
        ),
        sample_duration=2.0,
        sample_rate=16000,
    )
