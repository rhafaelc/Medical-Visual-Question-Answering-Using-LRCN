from pathlib import Path
import json


def _norm_answer_type(s, answer):
    s = (s or "").strip().lower()
    if s in {"open", "closed"}:
        return s
    a = (answer or "").strip().lower()
    if a in {"yes", "no"} or a.replace(".", "", 1).isdigit():
        return "closed"
    return "open"


def load_slake(root="data/raw/slake_all"):
    root_p = Path(root)
    ann_dir = root_p / "annotations"
    out = []
    for split in ("train", "validation", "test"):
        p = ann_dir / f"{split}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        idx = 0
        for ex in data:
            if (ex.get("q_lang") or "").strip().lower() != "en":
                continue
            img_rel = ex.get("img_name") or ex.get("img_id")
            if not img_rel:
                continue
            q = (ex.get("question") or "").strip()
            a = (ex.get("answer") or "").strip().lower()
            at = _norm_answer_type(ex.get("answer_type"), a)
            out.append(
                {
                    "id": f"slake_{split}_{idx:05d}",
                    "dataset": "slake_all",
                    "split": split,
                    "image": str(root_p / "images" / img_rel),
                    "question": q,
                    "answer": a,
                    "answer_type": at,
                }
            )
            idx += 1
    return out
