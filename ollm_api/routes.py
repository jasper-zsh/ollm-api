from fastapi import FastAPI
from .api.openai.chat import router as OpenAIChatRouter
from .api.openai.embeddings import router as OpenAIEmbeddingsRouter

def init_routers(app: FastAPI):
    app.include_router(OpenAIChatRouter)
    app.include_router(OpenAIEmbeddingsRouter)