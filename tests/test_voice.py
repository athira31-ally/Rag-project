import wave
from io import BytesIO

from app.voice import StubSpeechToTextProvider, ToneTextToSpeechProvider


def test_stub_stt_returns_provided_transcript():
    provider = StubSpeechToTextProvider()
    assert provider.transcribe(b"how long do refunds take?") == "how long do refunds take?"


def test_tone_tts_produces_valid_wav():
    provider = ToneTextToSpeechProvider()
    audio_bytes = provider.synthesize("a short answer")

    with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() > 0


def test_tone_tts_duration_scales_with_text_length():
    provider = ToneTextToSpeechProvider()
    short_audio = provider.synthesize("hi")
    long_audio = provider.synthesize("a much longer answer " * 10)

    assert len(long_audio) > len(short_audio)
