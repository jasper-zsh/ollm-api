from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Union

class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    encoding_format: Optional[str] = 'float'
    user: Optional[str]

class Embedding(BaseModel):
    index: int
    embedding: List[float]
    object: str = 'embedding'

router = APIRouter()

@router.post('/v1/embeddings')
def createEmbeddings(req: EmbeddingsRequest) -> List[Embedding]:
    return [
        Embedding(
            index=0,
            embedding=[]
        )
    ]