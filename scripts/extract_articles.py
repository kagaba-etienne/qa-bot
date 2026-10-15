import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "edited"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


txt_files = [
    {
        "file": 'LAW N° 027_2023 OF 18_05_2023 AMENDING THE LAW N° 66_2018 OF 30_08_2018 REGULATING LABOUR IN RWANDA.txt',
        "id": "law_027_2023",
        "law_name": "Regulating Labour in Rwanda",
        "law_id": "Law N° 027/2023",
        "amending_law_id": "Law N° 66/2018",
        # (former article, ammended article, repealed?, inserted article?)
        "ammended_articles": [
            (24, 1, 0, 1),
            (43, 2, 0, 0),
            (44, 3, 1, 0),
            (45, 4, 0, 0),
            (56, 5, 0, 0),
            (57, 6, 1, 0)
        ],
    },
    {
        "file": 'LAW N° 27_2021 OF 10_06_2021 GOVERNING LAND.txt',
        "id": "law_27_2021",
        "law_name": "Governing Land",
        "law_id": "Law N° 27/2021",
    },
    {
        "file": 'LAW N° 66_2018 OF 30_08_2018 REGULATING LABOUR IN RWANDA.txt',
        "id": "law_66_2018",
        "law_name": "Regulating Labour in Rwanda",
        "law_id": "Law N° 66/2018",
    },
    {
        "file": 'LAW Nº 059_2023 OF 04_12_2023 AMENDING LAW Nº 68_2018 OF 30_08_2018 DETERMINING OFFENCES AND PENALTIES IN GENERAL.txt',
        "id": "law_059_2023",
        "law_name": "Determining Offences and Penalties in General",
        "law_id": "Law N° 059/2023",
        "amending_law_id": "Law N° 68/2018",
        # (former article, ammended article, repealed?, inserted article?)
        "ammended_articles": [
            (35, 1, 0, 0),
            (60, 2, 0, 0),
            (64, 3, 0, 0),
            (84, 4, 0, 0),
            (92, 5, 0, 0),
            (99, 6, 0, 0),
            (100, 7, 0, 0),
            (108, 8, 0, 0),
            (117, 9, 0, 0),
            (120, 10, 0, 0),
            (121, 11, 0, 0),
            (123, 12, 0, 0),
            (132, 13, 0, 1),
            (133, 14, 0, 0),
            (152, 15, 0, 0),
            (172, 16, 0, 0),
            (177, 17, 0, 0),
            (218, 18, 1, 0),
            (262, 19, 0, 1),
            (271, 20, 0, 0),
            (291, 21, 0, 0)
        ],
    },
    {
        "file": 'LAW Nº 69_2019 OF 08_11_2019 AMENDING LAW Nº 68_2018 OF 30_08_2018 DETERMINING OFFENCES AND PENALTIES IN GENERAL.txt',
        "id": "law_69_2019",
        "law_name": "Determining Offences and Penalties in General",
        "law_id": "Law N° 69/2019",
        "amending_law_id": "Law N° 68/2018",
        # (former article, ammended article, repealed?, inserted article?)
        "ammended_articles": [
            (60, 1, 0, 0),
            (68, 2, 0, 0),
            (92, 3, 0, 0),
            (133, 4, 0, 0),
            (136, 5, 0, 0),
            (140, 6, 0, 0),
            (154, 7, 1, 0),
            (190, 8, 0, 0),
            (233, 9, 1, 0),
            (236, 10, 1, 0),
            (236, 10, 1, 0),
            (263, 11, 0, 0),
        ],
    },
    {
        "file": 'LAW Nº 71_2024 OF 26_06_2024 GOVERNING PERSONS AND FAMILY.txt',
        "id": "law_71_2024",
        "law_name": "Governing Persons and Family",
        "law_id": "Law N° 71/2024",

    },
    {
        "file": 'LAW Nº68_2018 OF 30_08_2018 DETERMINING OFFENCES AND PENALTIES IN GENERAL.txt',
        "id": "law_68_2018",
        "law_name": "Determining Offences and Penalties in General",
        "law_id": "Law N° 68/2018",
    },
]

extraction_strategy = {
    "content": re.compile(r"^Article\s+\d+\s*:[\s\S]*?(?=(?:^\s*Article\s+\d+\s*:)|$(?![\r\n]))", re.MULTILINE),
    "cleaning": re.compile(r"^\s*(?:CHAPTER|Section|Sub-?section)\s+[^:\n]*:[\s\S]*", re.MULTILINE | re.IGNORECASE),
    "is_definitions": re.compile(r"^Article\s([\d]+):\s(Definitions|Interpretation)$", re.MULTILINE),
    "definition_items": re.compile(r"^(\([a-z]+\)|\d+\s*[°º])[^\n]*(?:\n(?!\([a-z]+\)|\d+\s*[°º])[^\n]*)*", re.MULTILINE)
}


def clean_article_content(raw_art: str) -> str:
    cleaned = extraction_strategy["cleaning"].sub("", raw_art)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*(?=[a-z])", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def extract_article_number(raw_art: str) -> int:
    match = re.search(r"^Article\s+(\d+)", raw_art, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def process_file_articles(file_meta: dict) -> dict:
    file_path = INTERIM_DIR / file_meta["file"]
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return {}

    text = file_path.read_text(encoding="utf-8")
    raw_articles = extraction_strategy["content"].findall(text)

    articles_dict = {}
    for raw_art in raw_articles:
        cleaned = clean_article_content(raw_art)
        if not cleaned:
            continue
        art_num = extract_article_number(cleaned)
        if art_num:
            articles_dict[art_num] = cleaned

    return articles_dict


def parse_article_into_chunks(law_id: str, law_name: str, art_num: int, art_text: str) -> list:
    chunks = []
    def_match = extraction_strategy["is_definitions"].search(art_text)

    if def_match:
        items = extraction_strategy["definition_items"].finditer(art_text)
        if items:
            header_line = art_text.splitlines()[0].strip()
            for idx, item in enumerate(items, start=1):
                item_content = item.group(0).strip()
                term_match = re.match(
                    r"^(?:\([a-z]+\)|\d+\s*[°º])\s*(?:[\"“‘](.+?)[\"”’]\s+(?:means|includes)|(.+?)\s*:)",
                    item_content
                )
                defined_term = (term_match.group(1) or term_match.group(
                    2) or "").strip() if term_match else ""
                chunks.append({
                    "id": f"{law_id}_art_{art_num}_def_{idx}",
                    "law_id": law_id,
                    "law_name": law_name,
                    "article_number": art_num,
                    "chunk_type": "definition_item",
                    "content": f"{header_line}\n{item_content}",
                    "metadata": {
                        "citation": f"Article {art_num} of {law_id}",
                        "is_definition": True,
                        "defined_term": defined_term
                    },
                })
            return chunks

    # Standard article chunk
    chunks.append({
        "id": f"{law_id}_art_{art_num}",
        "law_id": law_id,
        "law_name": law_name,
        "article_number": art_num,
        "chunk_type": "full_article",
        "content": art_text,
        "metadata": {
            "citation": f"Article {art_num} of {law_id}",
            "is_definition": False,
        },
    })
    return chunks


def build_dataset():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    # Map base laws and amendments
    base_laws = [f for f in txt_files if "amending_law_id" not in f]
    amendment_laws = [f for f in txt_files if "amending_law_id" in f]

    # Process independent base laws first
    for base in base_laws:
        print(f"Processing: {base['law_id']} ({base['law_name']})")
        articles = process_file_articles(base)

        # Apply amendments tracking (former_art, amending_art, repealed, inserted)
        matching_amendments = [
            a for a in amendment_laws if a["amending_law_id"] == base["law_id"]
        ]

        for amend in matching_amendments:
            amend_articles = process_file_articles(amend)
            insertion_counts = {}
            for former_art, amend_art_num, repealed, inserted in amend["ammended_articles"]:
                if repealed == 1:
                    articles.pop(former_art, None)
                elif inserted == 1:
                    if amend_art_num in amend_articles:
                        insertion_counts[former_art] = insertion_counts.get(
                            former_art, 0) + 1
                        insert_key = former_art + \
                            (insertion_counts[former_art] * 0.1)
                        articles[insert_key] = amend_articles[amend_art_num]
                else:
                    if amend_art_num in amend_articles:
                        articles[former_art] = amend_articles[amend_art_num]

        # Convert remaining articles into output chunks
        definitions = []
        for art_num in sorted(articles.keys()):
            clean_art_num_str = f"{art_num:g}"
            art_chunks = parse_article_into_chunks(
                law_id=base["law_id"],
                law_name=base["law_name"],
                art_num=clean_art_num_str,
                art_text=articles[art_num],
            )
            for chunk in art_chunks:
                if chunk["metadata"].get("is_definition"):
                    # Add to tracker if a term was successfully extracted
                    term = chunk["metadata"].get("defined_term")
                    if term:
                        definitions.append({"id": chunk["id"], "term": term})

                    chunk["metadata"]["related_definitions"] = []
                else:
                    related_ids = []
                    chunk_content = chunk["content"]

                    for def_obj in definitions:
                        term = def_obj["term"]
                        # Use regex with word boundaries (\b) and ignore case
                        pattern = r'\b' + re.escape(term) + r'\b'
                        if re.search(pattern, chunk_content, re.IGNORECASE):
                            related_ids.append(def_obj["id"])

                    chunk["metadata"]["related_definitions"] = related_ids

            all_chunks.extend(art_chunks)

    # Save to JSONL
    output_path = PROCESSED_DIR / "rwanda_law_corpus.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(
        f"\nPipeline complete. {len(all_chunks)} chunks written to {output_path}")


if __name__ == "__main__":
    build_dataset()
