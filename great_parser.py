import re
import json
from pathlib import Path

def brat_to_openNRE(text, ann):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    char_to_token_idx = {}
    pos = 0
    for i, token in enumerate(tokens):
        start = text.find(token, pos)
        for c in range(start, start + len(token)):
            char_to_token_idx[c] = i
        pos = start + len(token)

    entities = {}
    relations = []

    for line in ann.splitlines():
        if line.startswith("T"):
            parts = line.strip().split('\t')
            if len(parts) != 3: continue
            eid = parts[0]
            etype_span = parts[1]
            etext = parts[2]
            m = re.match(r"(\S+) (\d+) (\d+)", etype_span)
            if not m: continue
            etype, start, end = m.groups()
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
            if len(parts) != 2: continue
            rel_info = parts[1]
            m = re.match(r"(\S+) Arg1:(T\d+) Arg2:(T\d+)", rel_info)
            if m:
                rel_type, arg1, arg2 = m.groups()
                if arg1 in entities and arg2 in entities:
                    relations.append({
                        "label": rel_type,
                        "h": entities[arg1],
                        "t": entities[arg2]
                    })

    samples = []
    for rel in relations:
        samples.append({
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

    return samples, [rel["label"] for rel in relations]


def process_folder(input_folder, out_train, out_rel2id):
    folder = Path(input_folder)
    all_data = []
    all_labels = set()

    for txt_file in folder.glob("*.txt"):
        ann_file = txt_file.with_suffix(".ann")
        if not ann_file.exists():
            continue
        text = txt_file.read_text(encoding="utf-8")
        ann = ann_file.read_text(encoding="utf-8")
        samples, labels = brat_to_openNRE(text, ann)
        all_data.extend(samples)
        all_labels.update(labels)

    # Save train.txt
    with open(out_train, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save rel2id.json
    all_labels = sorted(all_labels)
    rel2id = {"no_relation": 0}
    rel2id.update({label: idx + 1 for idx, label in enumerate(all_labels)})

    with open(out_rel2id, "w", encoding="utf-8") as f:
        json.dump(rel2id, f, ensure_ascii=False, indent=2)

    print(f"✅ Обработано файлов: {len(list(folder.glob('*.ann')))}")
    print(f"✅ Сохранено: {out_train} и {out_rel2id}")


# ======= Использование =======
# Пример: process_folder("data", "train.txt", "rel2id.json")

if __name__ == "__main__":
    process_folder("test", "test.txt", "rel2id.json")
