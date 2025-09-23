from pathlib import Path
import json

_PUBLIC = "VQA_RAD Dataset Public.json"


def _norm_answer_type(s, answer):
    s = (s or "").strip().lower()
    if s in {"open", "closed"}:
        return s
    a = str(answer or "").strip().lower()
    if a in {"yes", "no"} or a.replace(".", "", 1).isdigit():
        return "closed"
    return "open"


def load_vqa_rad(root="data/raw/vqa-rad"):
    root_p = Path(root)
    ann = root_p / "annotations" / _PUBLIC
    out = []
    if not ann.exists():
        return out

    data = json.loads(ann.read_text(encoding="utf-8"))
    counters = {"train": 0, "test": 0}
    for ex in data:
        img = ex.get("image_name")
        if not img:
            continue
        q = (ex.get("q_lang") or ex.get("question") or "").strip()
        a = str(ex.get("answer") or "").strip().lower()
        at = _norm_answer_type(ex.get("answer_type"), a)

        pt = (ex.get("phrase_type") or "").strip().lower()
        split = "test" if pt.startswith("test_") else "train"

        i = counters[split]
        out.append(
            {
                "id": f"vqarad_{split}_{i:05d}",
                "dataset": "vqa-rad",
                "split": split,
                "image": str(root_p / "images" / img),
                "question": q,
                "answer": a,
                "answer_type": at,
            }
        )
        counters[split] += 1

    return out
