from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.config.env import PROJECT_ROOT
from src.routes.chat import router

FRONTEND_BUILD = PROJECT_ROOT / "frontend" / "build" / "client"

app = FastAPI(title="Rwandan legal assistant")
app.include_router(router, prefix="/api")

if FRONTEND_BUILD.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_BUILD, html=True),
        name="frontend",
    )
