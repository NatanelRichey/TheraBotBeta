from fastapi import APIRouter, Depends
from fastapi.responses import Response, StreamingResponse

from app.core.dependencies import get_chat_service
from app.models.chat import ChatRequest, ChatResponse, CompareResponse, VoteRequest
from app.services.chat import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await service.chat(request)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> Response:
    async def generate() -> object:
        async for chunk in service.stream(request):
            yield chunk.model_dump_json() + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/compare", response_model=CompareResponse)
async def chat_compare(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> CompareResponse:
    return await service.compare(request)


@router.post("/compare/vote")
async def chat_compare_vote(
    vote: VoteRequest,
    service: ChatService = Depends(get_chat_service),
) -> dict[str, str]:
    await service.record_vote(vote)
    return {"status": "ok"}
