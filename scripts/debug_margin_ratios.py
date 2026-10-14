import pdfplumber


def debug_crop_margins(pdf_path, page_number=2):
    """
    Debug the crop margins for a given PDF file and page number.
    Saves a debug image to the current directory.

    Args:
        pdf_path: The path to the PDF file.
        page_number: The page number to debug.
    """
    with pdfplumber.open(pdf_path) as pdf:
        # Get a representative page (page 2 usually has standard text)
        page = pdf.pages[page_number]
        width = page.width
        height = page.height

        # 1. Tweak these ratios to test different margins
        left_ratio = 0.37
        right_ratio = 0.631
        top_margin = 69
        bottom_margin = 77

        x0 = width * left_ratio
        x1 = width * right_ratio
        top = top_margin
        bottom = height - bottom_margin

        bounding_box = (x0, top, x1, bottom)

        # 2. Convert the page to an image
        im = page.to_image(resolution=150)

        # 3. Draw a red rectangle showing exactly where the script will crop
        im.draw_rect(bounding_box, stroke="red", stroke_width=3)

        # 4. Save the image to your computer to look at it
        debug_image_path = "debug_crop_test.png"
        im.save(debug_image_path, format="PNG")
        print(f"Debug image saved to {debug_image_path}")
