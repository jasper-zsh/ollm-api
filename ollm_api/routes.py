from fastapi import FastAPI
from .api.openai.api import router as OpenAIRouter
from .api.openai.embeddings import router as OpenAIEmbeddingsRouter

def init_routers(app: FastAPI):
    app.include_router(OpenAIRouter)
    app.include_router(OpenAIEmbeddingsRouter)