from pathlib import Path

from pydantic import BaseSettings


class Secrets(BaseSettings):
    USERNAME: str
    PASSWORD: str

    class Config:
        case_sensitive = True
        env_file = "secrets.env"


class Config(BaseSettings):
    VOXCELEB1_FILEPARTS_LIST: Path
    VOXCELEB1_CONCAT_LIST: Path
    VOXCELEB1_METADATA_LIST: Path
    VOXCELEB1_ZIP_PATH: Path
    VOXCELEB1_FILES_PATH: Path
    VOXCELEB1_H5_PATH: Path

    VOXCELEB2_FILEPARTS_LIST: Path
    VOXCELEB2_CONCAT_LIST: Path
    VOXCELEB2_METADATA_LIST: Path
    VOXCELEB2_ZIP_PATH: Path
    VOXCELEB2_FILES_PATH: Path
    VOXCELEB2_H5_PATH: Path

    class Config:
        case_sensitive = True
        env_file = ".env"
