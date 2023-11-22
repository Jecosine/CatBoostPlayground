import hashlib
import logging
import os
import warnings
import zipfile
from typing import Optional, List
import numpy as np
import pandas as pd
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
            for chunk in req.iter_content(chunk_size=__CHUNK_SIZE):
                f.write(chunk)
            # with progress:
            #     tid = progress.add_task("Download", filename=desc)
            #     progress.update(tid, total=total)
            #     progress.start_task(tid)
    print(f"Temp file downloaded: {path}")


def _extract_file(src_path: str, dst_path: str = ".", ext: str = "zip"):
    cur_path = os.getcwd()
    print(f"Extracting {src_path} to {dst_path}...")
    extractor = (
        (lambda path: zipfile.ZipFile(path, "r"))
        if (ext == "zip")
        else (lambda path: tarfile.open(path, "r:*"))
    )
    with extractor(src_path) as f:
        os.chdir(dst_path)
        try:
            f.extractall()
        finally:
            os.chdir(cur_path)


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


def _get_feature_info(info: dict, dataset_name: str):
    columns = []
    cat_idx = []
    for idx, attr in enumerate(info[dataset_name]["attributes"]):
        a = attr["name"].strip().lower().replace("-", "_").replace(" ", "_")
        columns.append(a)
        if attr["type"] in ["Categorical", "Binary"] and attr["role"] != "Target":
            cat_idx.append(idx)
    return columns, cat_idx


def apply_cat_dtypes(data: pd.DataFrame, cat_idx: list):
    columns = list(data.columns)
    if len(cat_idx) == 0:
        return data
    for idx in cat_idx:
        if idx >= len(columns):
            warnings.warn("Cat idx out of range, make sure the idx is correct")
            continue
        data[columns[idx]] = data[columns[idx]].astype("category")
    return data


def get_cat_idx(data: pd.DataFrame) -> List[int]:
    columns = data.columns
    cat_idx = []
    for idx, col in enumerate(columns):
        # if pd.api.types.is_categorical_dtype(data[col]):
        if isinstance(data[col].dtype, pd.CategoricalDtype):
            cat_idx.append(idx)
    return cat_idx


def remove_inf(data: pd.DataFrame):
    return data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
