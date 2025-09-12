#!/usr/bin/env python3
from pathlib import Path
from osfclient.api import OSF
from ._common import project_root, ensure_dir, osf_download_many, safe_move
import shutil

ROOT = project_root(__file__)
RAW = ensure_dir(ROOT / "data" / "raw" / "vqa-rad")
TMP = ensure_dir(RAW / "_tmp")

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ANN_EXT = (".json", ".csv", ".tsv", ".xlsx", ".xls", ".xml", ".docx", ".txt")
KEEP_AT_ROOT = {
    "VQA_RAD Dataset Public.json",
    "VQA_RAD Dataset Public.xlsx",
    "VQA_RAD Dataset Public.xml",
    "Readme.docx",
}


def _walk_files():
    osf = OSF()
    store = osf.project("89kps").storage("osfstorage")
    img_folder = next(
        (f for f in store.folders if f.name.lower() == "vqa_rad image folder".lower()),
        None,
    )
    if not img_folder:
        raise SystemExit("VQA_RAD Image Folder not found")

    def rec(folder):
        for f in folder.files:
            yield f
        for sub in folder.folders:
            yield from rec(sub)

    img_files = list(rec(img_folder))
    root_files = [
        f
        for f in store.files
        if f.name in KEEP_AT_ROOT or any(f.name.lower().endswith(e) for e in ANN_EXT)
    ]
    return img_files, root_files


def _normalize():
    images = ensure_dir(RAW / "images")
    ann = ensure_dir(RAW / "annotations")
    for p in TMP.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        target = images / p.name if name.endswith(IMG_EXT) else ann / p.name
        safe_move(p, target)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"[OK] vqa-rad -> {images} , {ann}")


def main():
    img_files, ann_files = _walk_files()
    jobs = [(f, TMP / "images_raw" / f.name) for f in img_files]
    jobs += [(f, TMP / "ann_raw" / f.name) for f in ann_files]
    osf_download_many(jobs)
    _normalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
