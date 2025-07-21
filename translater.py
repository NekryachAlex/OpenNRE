import re
import json
import os
from pathlib import Path

def convert_brat_to_openNRE(txt_file, ann_file, out_jsonl, rel2id_out):
    text = Path(txt_file).read_text(encoding="utf-8")
    ann_lines = Path(ann_file).read_text(encoding="utf-8").splitlines()

    # Простая токенизация
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    # Создание отображения символов → токены
    char_to_token_idx = {}
    pos = 0
    for i, token in enumerate(tokens):
        start = text.find(token, pos)
        for c in range(start, start + len(token)):
            char_to_token_idx[c] = i
        pos = start + len(token)

    # Чтение аннотаций
    entities = {}
    relations = []

    for line in ann_lines:
        if line.startswith("T"):
            parts = line.strip().split('\t')
            eid = parts[0]
            etype_span = parts[1]
            etext = parts[2]
            etype, start, end = re.match(r"(\S+) (\d+) (\d+)", etype_span).groups()
            start, end = int(start), int(end) - 1
            if start in char_to_token_idx and end in char_to_token_idx:
                entities[eid] = {
                    "name": etext,
                    "type": etype,
                    "char_start": start,
                    "char_end": end,
                    "token_start": char_to_token_idx[start],
                    "token_end": char_to_token_idx[end],
                }
        elif line.startswith("R"):
            parts = line.strip().split('\t')
            rid = parts[0]
            rel_info = parts[1]
            match = re.match(r"(\S+) Arg1:(T\d+) Arg2:(T\d+)", rel_info)
            if match:
                rel_type, arg1, arg2 = match.groups()
                if arg1 in entities and arg2 in entities:
                    relations.append({
                        "label": rel_type,
                        "h": entities[arg1],
                        "t": entities[arg2]
                    })

    # Генерация jsonl
    result = []
    for rel in relations:
        result.append({
            "token": tokens,
            "h": {
                "name": rel["h"]["name"],
                "pos": [rel["h"]["token_start"], rel["h"]["token_end"]]
            },
            "t": {
                "name": rel["t"]["name"],
                "pos": [rel["t"]["token_start"], rel["t"]["token_end"]]
            },
            "label": rel["label"]
        })

    # Сохранение
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    labels = sorted(set(r["label"] for r in relations))
    rel2id = {"no_relation": 0}
    rel2id.update({label: idx + 1 for idx, label in enumerate(labels)})

    with open(rel2id_out, "w", encoding="utf-8") as f:
        json.dump(rel2id, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {out_jsonl} and {rel2id_out}")


# ====== Пример использования ======
# Положи рядом файлы 598012_text.txt и 598012_text.ann, затем:

if __name__ == "__main__":
    convert_brat_to_openNRE(
        txt_file="1130.txt",
        ann_file="1130.ann",
        out_jsonl="train.txt",
        rel2id_out="rel2id.json"
    )