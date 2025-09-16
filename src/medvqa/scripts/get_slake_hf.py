#!/usr/bin/env python3
from pathlib import Path
from huggingface_hub import snapshot_download
from ._common import project_root, ensure_dir, safe_move, extract_zip_once, verify_json

ROOT = project_root(__file__)
RAW = ensure_dir(ROOT / "data" / "raw" / "slake_all")
TMP = ensure_dir(RAW / "_tmp")

NEEDED = {"train.json", "validation.json", "test.json", "imgs.zip"}


def _summarize():
    images = RAW / "images"
    ann = RAW / "annotations"
    n_img = sum(1 for _ in images.glob("*")) if images.exists() else 0
    n_ann = sum(1 for _ in ann.glob("*.json")) if ann.exists() else 0
    print("\n[Listing]")
    print(f"{images}: {n_img} files")
    print(f"{ann}: {n_ann} files")


def main():
    repo = Path(snapshot_download(repo_id="BoKelvin/SLAKE", repo_type="dataset"))
    for p in repo.iterdir():
        if p.is_file() and p.name in NEEDED:
            dst = TMP / p.name
            if not dst.exists():
                dst.write_bytes(p.read_bytes())

    imgs = TMP / "imgs.zip"
    if not imgs.exists():
        raise SystemExit("imgs.zip not found in HF snapshot")
    extract_zip_once(imgs, RAW / "images")

    ann = ensure_dir(RAW / "annotations")
    for name in ("train.json", "validation.json", "test.json"):
        src = TMP / name
        if not src.exists():
            raise SystemExit(f"missing {name}")
        verify_json(src)
        safe_move(src, ann / name)

    import shutil

    shutil.rmtree(TMP, ignore_errors=True)

    print(f"[OK] slake_all -> {RAW / 'images'} , {ann}")
    _summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
