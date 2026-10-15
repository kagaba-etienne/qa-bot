import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"/ "edited"
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
    "articles": {
        "content": r"/^Article\s+\d+\s*:[\s\S]*?(?=(?:^\s*Article\s+\d+\s*:)|$(?![\r\n]))/gm",
        "cleaning": r"/^\s*(?:CHAPTER|Section|Sub-?section)\s+[^:\n]*:[\s\S]*/gmi",
        "is_definitions": r"/^Article\s([\d]+):\s(Definitions|Interpretation)$/gm",
        "definition_items": r"/^(\([a-z]+\)|\d+\s*[°º])[^\n]*(?:\n(?!\([a-z]+\)|\d+\s*[°º])[^\n]*)*/gm"
    }
}
