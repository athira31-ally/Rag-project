"""Real OCR test -- renders text into a synthetic image with Pillow and
verifies Tesseract actually reads it back. No fixtures, no mocking.
"""
import io

from PIL import Image, ImageDraw, ImageFont

from app.ocr import OCRPipeline, TesseractOCRProvider


def _render_text_image(text: str) -> bytes:
    image = Image.new("RGB", (600, 100), color="white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 30), text, fill="black", font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_tesseract_reads_rendered_text():
    image_bytes = _render_text_image("HELLO TALKUMENT")
    pipeline = OCRPipeline(primary=TesseractOCRProvider())

    extracted = pipeline.extract_text(image_bytes)

    assert "HELLO" in extracted.upper()
    assert "TALKUMENT" in extracted.upper()


def test_falls_back_when_primary_returns_too_little_text():
    class EmptyProvider:
        def extract_text(self, image_bytes: bytes) -> str:
            return ""

    class FallbackProvider:
        def extract_text(self, image_bytes: bytes) -> str:
            return "recovered by fallback"

    pipeline = OCRPipeline(primary=EmptyProvider(), fallback=FallbackProvider(), min_chars=5)

    assert pipeline.extract_text(b"irrelevant") == "recovered by fallback"


def test_no_fallback_when_primary_is_sufficient():
    class GoodProvider:
        def extract_text(self, image_bytes: bytes) -> str:
            return "this is plenty of text already"

    class ShouldNotBeCalledProvider:
        def extract_text(self, image_bytes: bytes) -> str:
            raise AssertionError("fallback should not have been called")

    pipeline = OCRPipeline(primary=GoodProvider(), fallback=ShouldNotBeCalledProvider(), min_chars=5)

    assert pipeline.extract_text(b"irrelevant") == "this is plenty of text already"
