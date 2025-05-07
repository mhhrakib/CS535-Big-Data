# File: src/api/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.api.services.model_manager import get_model_manager
from src.api.routers.models import router as models_router
from src.api.routers.summarise import router as summarise_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-News Summarizer API",
        description="Abstractive multi-document summarization with PEGASUS, BART, LED",
    )

    # Enable CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files and templates
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")

    # Include API routers
    app.include_router(models_router, prefix="/api")
    app.include_router(summarise_router, prefix="/api")

    # Home page
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        mgr = get_model_manager()
        models = mgr.get_available_models()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models
        })

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
