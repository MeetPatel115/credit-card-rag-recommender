from pathlib import Path
import json

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/all_card_snippets.jsonl")


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_chunk_text(text: str) -> str:
    if not text:
        return ""

    noise_terms = [
        "change country",
        "english français",
        "submit a claim",
        "information button icon",
        "axp-icon-pluscircle",
        "axp-icon-right",
        "support",
        "view all",
    ]

    t = str(text)
    for term in noise_terms:
        t = t.replace(term, " ")

    t = " ".join(t.split())
    return t.strip()


def main():
    snippet_files = sorted(RAW_DIR.glob("card_snippets_*.jsonl"))
    if not snippet_files:
        raise FileNotFoundError("No files found like data/raw/card_snippets_*.jsonl")

    seen = set()
    total = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for file in snippet_files:
            print(f"Reading {file.name}")

            for obj in load_jsonl(file):
                chunk_id = obj.get("chunk_id")
                if not chunk_id or chunk_id in seen:
                    continue

                obj["section"] = str(obj.get("section", "")).strip().lower()
                obj["text"] = clean_chunk_text(obj.get("text", ""))

                if not obj["text"]:
                    continue

                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                seen.add(chunk_id)
                total += 1

    print(f"\nSaved: {OUT_PATH}")
    print(f"Total chunks: {total}")


if __name__ == "__main__":
    main()