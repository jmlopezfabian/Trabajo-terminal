"""FastAPI application entry point for the Black Marble processing API."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(
    title="Black Marble NTL API",
    version="1.0.0",
    description="Async API for nighttime lights processing (VNP46A2) with background jobs.",
)
app.include_router(router)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def root():
        return FileResponse(static_dir / "index.html")
