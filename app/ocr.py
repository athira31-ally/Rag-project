"""OCR pipeline: extract text from an uploaded image before it's chunked
and ingested like any other document.

Provider chain mirrors the rest of the repo's pluggable-provider pattern:
a real, fully local default (Tesseract, via pytesseract -- no API key, no
network) with an optional cloud fallback for handwriting/low-quality scans
that a local OCR engine struggles with.
"""
from __future__ import annotations

import io
from typing import Protocol

from app.config import settings


class OCRProvider(Protocol):
    def extract_text(self, image_bytes: bytes) -> str: ...


class TesseractOCRProvider:
    """Local, offline OCR -- the default. Requires the `tesseract` binary
    (apt install tesseract-ocr) which the Dockerfile installs."""

    def extract_text(self, image_bytes: bytes) -> str:
        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image).strip()


class OpenAIVisionOCRProvider:
    """Cloud fallback for scans Tesseract can't read well (skewed pages,
    handwriting, low-resolution phone photos)."""

    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OCR cloud fallback")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def extract_text(self, image_bytes: bytes) -> str:
        import base64

        b64 = base64.b64encode(image_bytes).decode("ascii")
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe all text in this image exactly, with no commentary."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        )
        return (response.choices[0].message.content or "").strip()


class OCRPipeline:
    """Runs the primary provider and falls back to the secondary one when
    the primary returns suspiciously little text (a cheap, real-world proxy
    for "the scan was too poor to read locally")."""

    def __init__(self, primary: OCRProvider, fallback: OCRProvider | None = None, min_chars: int = 20):
        self.primary = primary
        self.fallback = fallback
        self.min_chars = min_chars

    def extract_text(self, image_bytes: bytes) -> str:
        text = self.primary.extract_text(image_bytes)
        if len(text) < self.min_chars and self.fallback is not None:
            fallback_text = self.fallback.extract_text(image_bytes)
            if len(fallback_text) > len(text):
                return fallback_text
        return text


def get_ocr_pipeline() -> OCRPipeline:
    primary: OCRProvider = TesseractOCRProvider()
    fallback: OCRProvider | None = None
    if settings.ocr_fallback_provider == "openai":
        fallback = OpenAIVisionOCRProvider(api_key=settings.openai_api_key, model=settings.openai_vision_model)
    return OCRPipeline(primary=primary, fallback=fallback)
