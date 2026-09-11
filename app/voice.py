"""Speech-to-text and text-to-speech providers for voice Q&A.

Default providers are fully offline and dependency-free (a deterministic
passthrough for STT driven by an explicit `known_text` in tests/demo mode,
and a real, self-contained WAV tone synthesizer for TTS using only the
standard library) so the voice route is genuinely testable without a
microphone, an audio corpus, or an API key. Real cloud providers plug into
the same interface.
"""
from __future__ import annotations

import io
import math
import struct
import wave
from typing import Protocol

from app.config import settings


class SpeechToTextProvider(Protocol):
    def transcribe(self, audio_bytes: bytes) -> str: ...


class StubSpeechToTextProvider:
    """Offline default. Real audio decoding needs a model (Whisper) that's
    too heavy to require for local dev/tests; this provider makes the voice
    *pipeline* (upload -> transcript -> RAG query -> synthesized answer)
    fully exercisable by reading a plain-text transcript that was uploaded
    alongside/instead of real audio bytes."""

    def transcribe(self, audio_bytes: bytes) -> str:
        return audio_bytes.decode("utf-8", errors="ignore").strip()


class OpenAIWhisperSTTProvider:
    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the 'openai' STT provider")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def transcribe(self, audio_bytes: bytes) -> str:
        buffer = io.BytesIO(audio_bytes)
        buffer.name = "audio.wav"
        result = self._client.audio.transcriptions.create(model=self._model, file=buffer)
        return result.text


class TextToSpeechProvider(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class ToneTextToSpeechProvider:
    """Offline default: encodes a real, valid WAV file (a short tone whose
    length scales with the answer length) rather than a dummy blob, so the
    output can genuinely be played/inspected. Swap TTS_PROVIDER=openai for
    natural speech synthesis once an API key is configured."""

    def __init__(self, sample_rate: int = 16000, freq_hz: float = 440.0):
        self.sample_rate = sample_rate
        self.freq_hz = freq_hz

    def synthesize(self, text: str) -> bytes:
        duration_seconds = max(0.3, min(3.0, len(text) / 40))
        n_samples = int(self.sample_rate * duration_seconds)
        frames = bytearray()
        for i in range(n_samples):
            t = i / self.sample_rate
            amplitude = 0.3 * math.sin(2 * math.pi * self.freq_hz * t)
            frames += struct.pack("<h", int(amplitude * 32767))

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(bytes(frames))
        return buffer.getvalue()


class OpenAITextToSpeechProvider:
    def __init__(self, api_key: str | None, model: str, voice: str):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the 'openai' TTS provider")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._voice = voice

    def synthesize(self, text: str) -> bytes:
        response = self._client.audio.speech.create(model=self._model, voice=self._voice, input=text)
        return response.read()


def get_stt_provider() -> SpeechToTextProvider:
    if settings.stt_provider == "stub":
        return StubSpeechToTextProvider()
    if settings.stt_provider == "openai":
        return OpenAIWhisperSTTProvider(api_key=settings.openai_api_key, model=settings.openai_stt_model)
    raise ValueError(f"Unknown STT_PROVIDER: {settings.stt_provider!r}")


def get_tts_provider() -> TextToSpeechProvider:
    if settings.tts_provider == "stub":
        return ToneTextToSpeechProvider()
    if settings.tts_provider == "openai":
        return OpenAITextToSpeechProvider(
            api_key=settings.openai_api_key, model=settings.openai_tts_model, voice=settings.openai_tts_voice
        )
    raise ValueError(f"Unknown TTS_PROVIDER: {settings.tts_provider!r}")
