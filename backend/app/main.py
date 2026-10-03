from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import ALLOWED_ORIGINS
from backend.app.utils.logging import configure_logging
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.prediction import router as prediction_router
from backend.app.api.routes.model import router as model_router


configure_logging()

app = FastAPI(
    title="Enterprise Credit Risk Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(model_router)


frontend = Path(__file__).resolve().parents[2] / "frontend"

if frontend.exists():
    app.mount(
        "/",
        StaticFiles(directory=frontend, html=True),
        name="frontend",
    )
