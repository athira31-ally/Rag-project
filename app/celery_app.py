"""Celery app for async ingestion, so a large document upload doesn't block
the request/response cycle. Broker/backend both point at Redis by default.
"""
from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "talkument_rag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.task_default_queue = "ingestion"
celery_app.autodiscover_tasks(["app"])
