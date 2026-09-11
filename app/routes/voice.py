from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, Form, UploadFile
from pydantic import BaseModel

from app.deps import get_pipeline
from app.models import Citation
from app.rag_pipeline import RagPipeline
from app.voice import get_stt_provider, get_tts_provider

router = APIRouter(prefix="/voice", tags=["voice"])


class VoiceQueryResponse(BaseModel):
    transcript: str
    answer: str
    citations: list[Citation]
    answer_audio_base64: str
    used_llm: bool


@router.post("/query", response_model=VoiceQueryResponse)
async def voice_query(
    audio: UploadFile,
    workspace_id: str = Form(default="default"),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> VoiceQueryResponse:
    audio_bytes = await audio.read()

    transcript = get_stt_provider().transcribe(audio_bytes)
    result = pipeline.query(question=transcript, workspace_id=workspace_id)
    answer_audio = get_tts_provider().synthesize(result.answer)

    return VoiceQueryResponse(
        transcript=transcript,
        answer=result.answer,
        citations=result.citations,
        answer_audio_base64=base64.b64encode(answer_audio).decode("ascii"),
        used_llm=result.used_llm,
    )
