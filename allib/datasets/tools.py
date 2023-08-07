import hashlib
import logging
import os
import zipfile
from typing import Optional

# from tqdm import tqdm
import requests
import tempfile
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
import tarfile
from allib.constants import *
from allib.exceptions import ChecksumNotMatchError

__BUF_SIZE = 1024 * 1024 * 10  # 10 MB buffer for file reading
__CHUNK_SIZE = 1024 * 8  # 8 MB chunk for download stream

logger = logging.Logger(__name__)

progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeRemainingColumn(),
)


def get_cache_path():
    return os.path.join(os.getcwd(), CACHE_DIR)


def _test_checksum(path: str, checksum: str):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        buffer = f.read(__BUF_SIZE)
        while buffer:
            md5.update(buffer)
    if checksum != md5.hexdigest():
        raise ChecksumNotMatchError()


def _download_file(url: str, path: str, desc: str | None = None):
    """Download file with progress visualization

    Args:
        url: file url
        fd: file descriptor of the local destination file
        desc: file description
    """
    with open(path, "wb") as f:
        with requests.get(url, stream=True) as req:
            req.raise_for_status()
            total = int(req.headers.get("content-length", 0))
            # print(total)
            with progress:
                tid = progress.add_task("Download", filename=desc)
                progress.update(tid, total=total)
                progress.start_task(tid)
                for chunk in req.iter_content(chunk_size=__CHUNK_SIZE):
                    f.write(chunk)
                    progress.update(tid, advance=len(chunk))


def _extract_file(src_path: str, dst_path: str = ".", ext: str = "zip"):
    print(f"Extracting {src_path} to {dst_path}...")
    extractor = (
        (lambda path: zipfile.ZipFile(path, "r"))
        if (ext == "zip")
        else (lambda path: tarfile.open(path, "r:*"))
    )
    with extractor(src_path) as f:
        os.chdir(dst_path)
        f.extractall()


def _download_dataset(url: str, checksum: str, name: str):
    cache_path = os.path.join(get_cache_path(), name)
    # make dir
    try:
        os.makedirs(cache_path)
    except OSError:
        if os.path.isfile(cache_path):
            raise

    fd, download_path = tempfile.mkstemp()
    os.close(fd)
    _download_file(url, download_path, name)
    # _test_checksum(download_path, checksum)
    _extract_file(download_path, cache_path)


def _test_download(url: str, desc: Optional[str]):
    """ Test url availability """
    desc = desc if desc else url.split("/")[-1]
    with requests.get(url, stream=True, headers={"Accept-Encoding": None}) as req:
        req.raise_for_status()
        logger.debug(f"""testing {desc} ... {req.headers.get("Content-Length", 0)}""")
