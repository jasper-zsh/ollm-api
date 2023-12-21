from fastapi import FastAPI
from .routes import init_routers

app = FastAPI()

init_routers(app)
