import os


def ensure_path(path: str, is_dir: bool = True):
    try:
        if is_dir:
            os.makedirs(path)
        else:
            return os.path.isfile(path)
    except OSError:
        if is_dir and os.path.isfile(path):
            raise


def validate_dataset(path: str):
    ensure_path(path)
    file_list = os.listdir(path)
    return ".meta" in file_list
