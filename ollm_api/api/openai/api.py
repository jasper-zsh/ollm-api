from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from .chat import ChatCompletionRequest, ChatCompletion, Usage, ChatCompletionChunkChoice, ChatCompletionChunk
from ...engine.transformers import TransformersEngine
from ...model.qwen import QwenModel
from typing import Generator
import uuid

router = APIRouter()

engine = TransformersEngine()
adapter = QwenModel(engine)

loaded = False

def warpStream(model: str, gen: Generator[ChatCompletionChunkChoice, None, None]) -> Generator:
    id = 'chatcmpl-' + str(uuid.uuid4()).replace('-', '')
    for choice in gen:
        yield ChatCompletionChunk(id=id, choices=[choice], model=model).json(exclude_none=True, ensure_ascii=False)

@router.post('/v1/chat/completions', response_model_exclude_none=True)
def createChatCompletion(req: ChatCompletionRequest) -> ChatCompletion:
    global loaded
    if not loaded:
        engine.load_model('Qwen/Qwen-1_8B-Chat', adapter)
        loaded = True
    if req.stream:
        gen = engine.chat_stream(req)
        gen = warpStream(req.model, gen)
        return EventSourceResponse(gen, media_type='text/event-stream')
    else:
        choice, usage = engine.chat(req)
        return ChatCompletion(
            model=req.model,
            choices=[choice],
            usage=usage,
        )