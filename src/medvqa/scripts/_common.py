#!/usr/bin/env python3
from pathlib import Path
import shutil, zipfile, time, json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def project_root(file_path):
    return Path(file_path).resolve().parents[3]


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def safe_move(src, dst):
    dst = Path(dst)
    if not dst.exists():
        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
    return dst


def extract_zip_once(z, out_dir, marker=".extracted"):
    out_dir = ensure_dir(out_dir)
    m = out_dir / marker
    if m.exists():
        return out_dir
    with zipfile.ZipFile(z) as zf:
        zf.extractall(out_dir)
    m.touch()
    return out_dir


def verify_json(path):
    json.loads(Path(path).read_text(encoding="utf-8"))
    return True


def osf_download_many(jobs, max_workers=16, retries=3, backoff=1.5):
    def _dl(args):
        fobj, dest = args
        dest = Path(dest)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        for i in range(retries):
            try:
                with dest.open("wb") as fp:
                    fobj.write_to(fp)
                return dest
            except Exception:
                time.sleep(backoff**i)
        return dest

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for _ in tqdm(
            ex.map(_dl, jobs), total=len(jobs), unit="file", desc="Downloading VQA-RAD"
        ):
            pass
