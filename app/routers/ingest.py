from celery.result import AsyncResult
from fastapi import APIRouter
from pydantic import BaseModel

from app.celery_app import celery_app

router = APIRouter()

class IngestRequest(BaseModel):
    tenant_id: str
    source_uri: str
    title: str
    raw_text: str
    metadata: dict | None = None

@router.post("/")
async def ingest_endpoint(req: IngestRequest):
    task = celery_app.send_task(
        "app.tasks.ingest_document_task",
        args=[req.tenant_id, req.source_uri, req.title, req.raw_text, req.metadata],
    )
    return {"task_id": task.id, "status": "queued"}




@router.get("/status/{task_id}")
async def ingest_status(task_id: str):
    result = AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result if result.ready() else None}