from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
import time
import uuid

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str
    function: FunctionCall

class Message(BaseModel):
    role: str
    content: Optional[str]
    name: Optional[str]
    tool_call_id: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    function_call: Optional[FunctionCall]

class Function(BaseModel):
    name: str
    description: Optional[str]
    parameters: dict

class Tool(BaseModel):
    type: str
    function: Function

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]]
    logprobs: Optional[bool]
    top_logprobs: Optional[int]
    max_tokens: Optional[int]
    n: Optional[int]
    presence_penalty: Optional[float]
    response_format: Optional[dict]
    seed: Optional[int]
    stop: Optional[Union[str, List[str]]]
    stream: Optional[bool]
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1
    tools: Optional[List[Tool]]
    tool_choice: Optional[str]
    user: Optional[str]
    function_call: Optional[Union[str, dict]]
    functions: Optional[List[Function]]

class LogProbContent(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: Optional[List['LogProbContent']]

class LogProbs(BaseModel):
    content: List[LogProbContent]

class ChatCompletionChoice(BaseModel):
    finish_reason: str
    index: int
    message: Message
    logprobs: Optional[LogProbs]

class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class ChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: 'chatcmpl-' + str(uuid.uuid4()).replace('-', ''))
    choices: List[ChatCompletionChoice]
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: Optional[str]
    object: str = 'chat.completion'
    usage: Usage

class ChatCompletionChunkChoice(BaseModel):
    delta: Message
    logprobs: Optional[LogProbs]
    finish_reason: Optional[str]
    index: int

class ChatCompletionChunk(BaseModel):
    id: str
    choices: List[ChatCompletionChunkChoice]
    created: int
    model: str
    system_fingerprint: str
    object: str

router = APIRouter()

@router.post('/v1/chat/completions')
def createChatCompletion(req: ChatCompletionRequest) -> ChatCompletion:
    return ChatCompletion(
        model=req.model,
        choices=[],
        usage=Usage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0
        )
    )