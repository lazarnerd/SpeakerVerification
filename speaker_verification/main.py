from pathlib import Path

from speaker_verification.dataset.config import Secrets
from speaker_verification.dataset.handler import DatasetHandler


if __name__ == "__main__":

    print("\033[H\033[J", end="")
    secrets = Secrets()
    handler = DatasetHandler(Path("data/voxceleb1"))
    handler.download(secrets.USERNAME, secrets.PASSWORD)

    handler = DatasetHandler(Path("data/voxceleb2"))
    handler.download(secrets.USERNAME, secrets.PASSWORD)
