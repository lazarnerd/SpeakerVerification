from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rich.progress import Progress, TaskID
import requests
import yaml
import time
import hashlib


ASCII_CHECKMARK = "\u2713"

ASCII_XMARK = "\u2717"


class FileVerificator:
    def __init__(
        self,
        file: Path,
        checksum: str,
        progress: Progress,
        chunk_size: int = 4096,
        prefix: str = "",
    ) -> None:
        self.file = file
        self.checksum = checksum
        self.progress = progress
        self.name = file.name
        self.chunk_size = chunk_size
        self.prefix = prefix

    def register_task(self) -> None:
        self.task = self.progress.add_task(self.name)

    def verify(self) -> bool:
        self.progress.update(
            self.task,
            total=self.file.stat().st_size,
            completed=0,
            refresh=True,
            description=f"{self.prefix}Verifying: {self.name}",
        )
        hash_md5 = hashlib.md5()
        with open(self.file, "rb") as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b""):
                hash_md5.update(chunk)
                self.progress.update(self.task, advance=len(chunk))
        matches = hash_md5.hexdigest() == self.checksum
        symbol = ASCII_CHECKMARK if matches else ASCII_XMARK
        self.progress.update(
            self.task,
            refresh=True,
            description=f"{self.prefix}Verifying: {self.name} {symbol}",
        )
        return matches


class FileDownloader(FileVerificator):
    def __init__(
        self,
        url: str,
        file: Path,
        checksum: str,
        progress: Progress,
        username: str = None,
        password: str = None,
        chunk_size: int = 4096,
        prefix: str = "",
        resume: bool = False,
    ) -> None:
        self.url = url

        self.username = username
        self.password = password
        self.resume = resume
        super().__init__(file, checksum, progress, chunk_size=chunk_size, prefix=prefix)

    def open_request(self, headers: dict = None):
        if self.username is not None and self.password is not None:
            r = requests.get(
                self.url,
                headers=headers,
                stream=True,
                auth=(self.username, self.password),
            )
        else:
            r = requests.get(self.url, headers=headers, stream=True)
        return r

    def download(self, resume: bool = False) -> None:
        open_mode = "ab" if resume else "wb"
        start = 0
        headers = None
        if resume:
            start = self.file.stat().st_size
            headers = {"Range": f"bytes={start}-"}
        r = self.open_request(headers=headers)
        total_size = start + int(r.headers.get("content-length", 0))

        with open("tmp", "w") as f:
            f.write(str(r.headers))
            f.write(f"{total_size}")

        self.progress.update(
            self.task,
            description=f"{self.prefix}Downloading: {self.name}",
            completed=start,
            refresh=True,
            total=total_size,
        )
        with open(self.file, open_mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    self.progress.update(self.task, advance=len(chunk))

    def process(self) -> None:
        self.register_task()
        resume = False
        if self.file.exists():
            if self.verify():
                return
            else:
                if self.resume:
                    resume = True
                else:
                    self.file.unlink()
        self.download(resume)
        if not self.verify():
            raise RuntimeError(f"MD5 Checksum mismatch for {self.file}")


class FileConcatenater(FileVerificator):
    def __init__(
        self,
        file: Path,
        file_parts: list[FileDownloader],
        checksum: str,
        progress: Progress,
        chunk_size: int = 4096,
        prefix: str = "",
    ) -> None:
        self.file_parts = file_parts
        super().__init__(file, checksum, progress, chunk_size=chunk_size, prefix=prefix)

    def register_task(self) -> None:
        self.task = self.progress.add_task(f"{self.prefix}Concatenating: {self.name}")

    def concatenate(self) -> None:
        self.register_task()
        if self.file.exists():
            if self.verify():
                return
            else:
                self.file.unlink()

        with ThreadPoolExecutor() as executor:
            for file in self.file_parts:
                executor.submit(file.process)

        size = sum([file.file.stat().st_size for file in self.file_parts])
        self.progress.update(
            self.task,
            description=f"{self.prefix}Concatenating: {self.name}",
            completed=0,
            refresh=True,
            total=size,
        )
        with open(self.file, "wb") as f:
            for file in self.file_parts:
                with open(file.file, "rb") as f_part:
                    for chunk in iter(lambda: f_part.read(self.chunk_size), b""):
                        f.write(chunk)
                        self.progress.update(self.task, advance=len(chunk))


class DatasetDownloader:
    def __init__(
        self,
        path: Path,
        dataset_config: dict,
        username: str = None,
        password: str = None,
        chunk_size: int = 1024 * 1024,
        resume: bool = True,
    ):
        self.path = path
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

        self.dataset_config = dataset_config
        self.chunk_size = chunk_size
        self.username = username
        self.password = password
        self.resume = resume

    def download(self):
        with Progress() as progress:
            with ThreadPoolExecutor() as executor:
                for file_name in self.dataset_config["download"]:
                    file = self.path / file_name
                    url = self.dataset_config["download"][file_name]["url"]
                    checksum = self.dataset_config["download"][file_name]["md5"]
                    file = FileDownloader(
                        url=url,
                        file=file,
                        checksum=checksum,
                        username=self.username,
                        password=self.password,
                        chunk_size=self.chunk_size,
                        progress=progress,
                        resume=self.resume,
                    )
                    executor.submit(file.process)
                for file_name in self.dataset_config["concatenate"]:
                    time.sleep(0.1)
                    file_parts = []
                    file_configs = self.dataset_config["concatenate"][file_name][
                        "files"
                    ]
                    for config in file_configs:
                        part_name = config["part_name"]
                        file = self.path / part_name
                        url = config["url"]
                        checksum = config["md5"]
                        file = FileDownloader(
                            url=url,
                            file=file,
                            checksum=checksum,
                            username=self.username,
                            password=self.password,
                            chunk_size=self.chunk_size,
                            progress=progress,
                            prefix=" |- ",
                            resume=self.resume,
                        )
                        file_parts.append(file)

                    file = self.path / file_name
                    checksum = self.dataset_config["concatenate"][file_name]["md5"]
                    file = FileConcatenater(
                        file=file,
                        file_parts=file_parts,
                        checksum=checksum,
                        chunk_size=self.chunk_size,
                        progress=progress,
                    )
                    executor.submit(file.concatenate)
