#!/usr/bin/env python3
from pathlib import Path
import argparse, json
from medvqa.datamodules.slake import load_slake
from medvqa.datamodules.vqa_rad import load_vqa_rad


def _head(xs, n):
    return xs[: max(0, n)]


def main():
    ap = argparse.ArgumentParser(description="Preview standardized MedVQA datasets")
    ap.add_argument("--dataset", choices=["slake", "vqa-rad", "all"], default="all")
    ap.add_argument("--head", type=int, default=3, help="show first N examples")
    ap.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    args = ap.parse_args()

    sets = []
    if args.dataset in ("slake", "all"):
        slake = load_slake()
        print(
            f"[SLAKE] total={len(slake)}  splits={{"
            f"train={sum(x['split']=='train' for x in slake)}, "
            f"validation={sum(x['split']=='validation' for x in slake)}, "
            f"test={sum(x['split']=='test' for x in slake)}}}"
        )
        sets.append(("SLAKE", slake))

    if args.dataset in ("vqa-rad", "all"):
        try:
            vqa = load_vqa_rad()
            print(
                f"[VQA-RAD] total={len(vqa)}  splits={{"
                f"train={sum(x['split']=='train' for x in vqa)}, "
                f"test={sum(x['split']=='test' for x in vqa)}}}"
            )
            sets.append(("VQA-RAD", vqa))
        except Exception as e:
            print(f"[VQA-RAD] Error loading dataset: {e}")

    for name, data in sets:
        print(f"\n[{name}] head {args.head}")
        for ex in _head(data, args.head):
            s = json.dumps(ex, ensure_ascii=False, indent=2 if args.pretty else None)
            print(s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
