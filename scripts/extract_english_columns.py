from pathlib import Path
import pdfplumber

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "unedited"


def extract_english_column(
    input_pdf_path: Path,
    output_txt_path: Path,
    top_margin: int = 70,
    bottom_margin: int = 72,
    left_margin_ratio: float = 0.33,
    right_margin_ratio: float = 0.66,
    pages: tuple[int, int] = (1, 100)  # inclusive
):
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_pages = []

    with pdfplumber.open(input_pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages[pages[0]-1:pages[1]], start=1):
            width = float(page.width)
            height = float(page.height)

            # Middle column boundaries (English)
            x0 = width * left_margin_ratio
            x1 = width * right_margin_ratio

            # Margins to strip recurring headers and running footers
            top = top_margin
            bottom = height - bottom_margin

            cropped_page = page.crop((x0, top, x1, bottom))
            text = cropped_page.extract_text()

            if text and text.strip():
                extracted_pages.append(
                    f"--- Page {page_num} ---\n{text.strip()}\n")

    output_txt_path.write_text("\n".join(extracted_pages), encoding="utf-8")
    print(
        f"Extracted: {output_txt_path.name} ({len(extracted_pages)} pages processed)")


if __name__ == "__main__":
    # Raw pdf files
    pdf_files = [
        {
            "file": 'LAW N° 027_2023 OF 18_05_2023 AMENDING THE LAW N° 66_2018 OF 30_08_2018 REGULATING LABOUR IN RWANDA.pdf',
            "left_ratio": 0.357,
            "right_ratio": 0.638,
            "top_margin": 70,
            "bottom_margin": 72,
            "pages": (3, 7)  # inclusive
        },
        {
            "file": 'LAW N° 27_2021 OF 10_06_2021 GOVERNING LAND.pdf',
            "left_ratio": 0.357,
            "right_ratio": 0.638,
            "top_margin": 70,
            "bottom_margin": 72,
            "pages": (14, 78)  # inclusive
        },
        {
            "file": 'LAW N° 66_2018 OF 30_08_2018 REGULATING LABOUR IN RWANDA.pdf',
            "left_ratio": 0.365,
            "right_ratio": 0.645,
            "top_margin": 65,
            "bottom_margin": 72,
            "pages": (18, 103)  # inclusive
        },
        {
            "file": 'LAW Nº 059_2023 OF 04_12_2023 AMENDING LAW Nº 68_2018 OF 30_08_2018 DETERMINING OFFENCES AND PENALTIES IN GENERAL.pdf',
            "left_ratio": 0.35,
            "right_ratio": 0.635,
            "top_margin": 70,
            "bottom_margin": 80,
            "pages": (4, 27)  # inclusive
        },
        {
            "file": 'LAW Nº 69_2019 OF 08_11_2019 AMENDING LAW Nº 68_2018 OF 30_08_2018 DETERMINING OFFENCES AND PENALTIES IN GENERAL.pdf',
            "left_ratio": 0.36,
            "right_ratio": 0.645,
            "top_margin": 69,
            "bottom_margin": 77,
            "pages": (3, 13)  # inclusive
        },
        {
            "file": 'LAW Nº 71_2024 OF 26_06_2024 GOVERNING PERSONS AND FAMILY.pdf',
            "left_ratio": 0.34,
            "right_ratio": 0.645,
            "top_margin": 69,
            "bottom_margin": 77,
            "pages": (54, 301)  # inclusive
        },
        {
            "file": 'LAW Nº68_2018 OF 30_08_2018 DETERMINING OFFENCES AND PENALTIES IN GENERAL.pdf',
            "left_ratio": 0.37,
            "right_ratio": 0.631,
            "top_margin": 69,
            "bottom_margin": 77,
            "pages": (44, 275)  # inclusive
        },
    ]

    for pdf_file in pdf_files:
        raw_pdf = DATA_RAW_DIR / pdf_file["file"]
        cleaned_txt = DATA_INTERIM_DIR / \
            pdf_file["file"].replace('.pdf', '.txt')
        if raw_pdf.exists():
            extract_english_column(
                raw_pdf,
                cleaned_txt,
                left_margin_ratio=pdf_file["left_ratio"],
                right_margin_ratio=pdf_file["right_ratio"],
                top_margin=pdf_file["top_margin"],
                bottom_margin=pdf_file["bottom_margin"],
                pages=pdf_file["pages"]
            )
        else:
            print(f"File not found: {raw_pdf}")
