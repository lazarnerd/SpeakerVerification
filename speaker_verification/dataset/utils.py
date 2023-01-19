import hashlib
import subprocess
import requests
import sys

from rich.progress import Progress

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock

from speaker_verification.dataset.config import Secrets


ASCII_CLOCK = "\u23F0"
ASCII_CHECKMARK = "\u2713"
ASCII_XMARK = "\u2717"


def reduce_string(string: str, max_length: int = 20, reduce_at: str = "CENTER") -> str:
    if len(string) > max_length:
        if reduce_at.upper() == "CENTER":
            left = (max_length // 2) - 1
            right = -((max_length // 2) - 2)
            string = string[:left] + "..." + string[right:]
        elif reduce_at.upper() == "LEFT":
            string = "..." + string[-(max_length - 3) :]
        else:
            string = string[: max_length - 3] + "..."
    return string


def fit_string(string: str, length: int = 20) -> str:
    if len(string) > length:
        string = string[: length - 3] + "..."
    elif len(string) < length:
        string += " " * (length - len(string))
    return string


def match_checksum(
    progress: Progress,
    file: Path,
    checksum: str,
    buffer_size: int = 4096,
    remove: bool = False,
) -> bool:
    display_filename = reduce_string(file.stem, max_length=20, reduce_at="CENTER")
    hash_md5 = hashlib.md5()

    task = progress.add_task(
        description=fit_string(f"md5 hash - {display_filename} [{ASCII_CLOCK}]", 40),
        total=file.stat().st_size,
    )
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(buffer_size), b""):
            hash_md5.update(chunk)
            progress.update(task, advance=len(chunk))
    matches = hash_md5.hexdigest() == checksum
    if matches:
        progress.update(
            task, fit_string(f"md5 hash - {display_filename} [{ASCII_CHECKMARK}]", 40)
        )
    else:
        progress.update(
            task, fit_string(f"md5 hash - {display_filename} [{ASCII_XMARK}]", 40)
        )
        if remove:
            file.unlink()

    return matches


def download_file(
    progress: Progress,
    url: str,
    target: Path,
    checksum: str,
    buffer_size: int = 4096,
    use_credentials: bool = False,
) -> None:
    display_filename = reduce_string(target.stem, max_length=20, reduce_at="CENTER")

    if not use_credentials:
        r = requests.get(url, stream=True)
    else:
        secrets = Secrets()
        r = requests.get(url, stream=True, auth=(secrets.USERNAME, secrets.PASSWORD))

    if r.status_code == 200:
        task = progress.create_task(
            fit_string(f"download: {display_filename} [{ASCII_CLOCK}]", 40),
            total=int(r.headers["Content-length"]),
        )
        with open(target, "wb") as f:
            for chunk in r.iter_content(buffer_size):
                f.write(chunk)
                progress.update(task, advance=len(chunk))
        progress.update(
            task, fit_string(f"download - {display_filename} [{ASCII_CHECKMARK}]", 40)
        )
    else:
        raise Exception(f"Error downloading file: {url}")


def process_fileparts(
    fileparts_list: Path, target_dir: Path, use_credentials=False
) -> None:
    print("====== Downloading VoxCeleb1 files ======")
    print("Validating target directory...")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Reading fileparts list...")
    with open(fileparts_list, "r") as f:
        lines = f.read().splitlines()
    fileparts = []
    for line in lines:
        parts = line.split(" ")
        if len(parts) == 2:
            url, checksum = parts
            target = target_dir / url.split("/")[-1]
            fileparts.append((url, target, checksum))

    existing_fileparts = [part for part in fileparts if part[1].exists()]
    if len(existing_fileparts) > 0:
        print("\nChecking md5 hashes of existing fileparts...")
        with Progress() as progress:
            with ThreadPoolExecutor() as executor:
                for i, (url, target, checksum) in enumerate(existing_fileparts):
                    executor.submit(
                        match_checksum,
                        progress,
                        target,
                        checksum,
                        remove=True,
                    )
        print("")

    new_fileparts = [part for part in fileparts if not part[1].exists()]
    if len(new_fileparts) > 0:
        print("\nDownloading new fileparts...")
        with Progress() as progress:
            with ThreadPoolExecutor() as executor:
                for i, (url, target, checksum) in enumerate(new_fileparts):
                    executor.submit(
                        download_file,
                        progress,
                        url,
                        target,
                        checksum,
                        use_credentials=use_credentials,
                    )
        print("")

    if len(new_fileparts) > 0:
        print("\nChecking md5 hashes of new fileparts...")
        matches = []
        with Progress() as progress:
            with ThreadPoolExecutor() as executor:
                futures = []
                for i, (url, target, checksum) in enumerate(new_fileparts):
                    futures.append(
                        executor.submit(
                            match_checksum,
                            progress,
                            target,
                            checksum,
                            remove=False,
                        )
                    )
                for future in as_completed(futures):
                    matches.append(future.result())
        print("")
        if not all(matches):
            raise Exception("Some fileparts failed md5 checksum validation.")


def concat_files(file_list: Path, zip_path: Path) -> None:
    zip_path.mkdir(parents=True, exist_ok=True)
    with open(file_list, "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        parts = line.split(" ")
        if len(parts) == 3:
            infile = zip_path / parts[0]
            outfile = zip_path / parts[1]
            checksum = parts[2]
            if outfile.exists():
                if match_checksum(outfile, checksum):
                    continue
                else:
                    outfile.unlink()

            subprocess.call(
                f"cat {str(infile)} > {str(outfile)}",
                shell=True,
            )

            if not match_checksum(outfile, checksum):
                raise Exception(f"Checksum does not match for file: {outfile}")


def unzip_files(zip_path: Path, files_path: Path) -> None:
    files_path.mkdir(parents=True, exist_ok=True)
    zip_files = zip_path.rglob("*.zip")
    for zip_file in zip_files:
        subprocess.call(
            f"unzip -o {str(zip_file)} -d {str(files_path)} -q",
            shell=True,
        )
