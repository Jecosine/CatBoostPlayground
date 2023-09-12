import os
import re


def newest_sample(template: str, ext: str = ".jpg", cur_dir=".", zf: int = 4, c: int = 0):
    flist = [i for i in os.listdir(cur_dir) if i.endswith(ext)]
    c = 0
    for i in flist:
        res = re.match(template, i)
        if res:
            res = res.groups()[0]
            zf = len(res)
            c = max(c, int(res))
    return c, zf

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
    file_list = [fn.split(".")[-1] for fn in os.listdir(path)]
    return "names" in file_list
