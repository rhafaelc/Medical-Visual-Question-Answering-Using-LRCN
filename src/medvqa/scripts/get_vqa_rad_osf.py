#!/usr/bin/env python3
from pathlib import Path
import requests, zipfile, io, shutil
from ._common import project_root, ensure_dir, safe_move

ROOT = project_root(__file__)
RAW = ensure_dir(ROOT / "data" / "raw" / "vqa-rad")
TMP = ensure_dir(RAW / "_tmp")

ZIP_URL = "https://files.osf.io/v1/resources/89kps/providers/osfstorage/?zip="

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ANN_EXT = (".json", ".csv", ".tsv", ".xlsx", ".xls", ".xml", ".docx", ".txt")
KEEP_AT_ROOT = {
    "VQA_RAD Dataset Public.json",
    "VQA_RAD Dataset Public.xlsx",
    "VQA_RAD Dataset Public.xml",
    "Readme.docx",
}


def _extract_all(buf):
    images = ensure_dir(TMP / "images_raw")
    ann = ensure_dir(TMP / "ann_raw")
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            lo = name.lower()
            if lo.endswith(IMG_EXT):
                out = images / name
            elif lo.endswith(ANN_EXT) or name in KEEP_AT_ROOT:
                out = ann / name
            else:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _summarize():
    images = RAW / "images"
    ann = RAW / "annotations"
    n_img = sum(1 for _ in images.glob("*")) if images.exists() else 0
    n_ann = sum(1 for _ in ann.glob("*")) if ann.exists() else 0
    print("\n[Listing]")
    print(f"{images}: {n_img} files")
    print(f"{ann}: {n_ann} files")


def _normalize():
    images = ensure_dir(RAW / "images")
    ann = ensure_dir(RAW / "annotations")
    for p in TMP.rglob("*"):
        if not p.is_file():
            continue
        lo = p.name.lower()
        target = images / p.name if lo.endswith(IMG_EXT) else ann / p.name
        safe_move(p, target)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"[OK] vqa-rad -> {images} , {ann}")
    _summarize()


def main():
    print("[OSF] downloading full project zip …")
    resp = requests.get(ZIP_URL, stream=True, timeout=300)
    resp.raise_for_status()
    buf = resp.content
    print(f"[OSF] zip size = {len(buf)/1e6:.1f} MB")
    _extract_all(buf)
    _normalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
