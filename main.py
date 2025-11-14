from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from app.api import routes_cv
from app.core.config import settings
from pathlib import Path
import asyncio
import sys
import warnings
import logging

# Configurer le logging pour supprimer les erreurs asyncio sur Windows
if sys.platform == "win32":
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    
    # Handler pour capturer les exceptions dans les callbacks asyncio
    def handle_exception(loop, context):
        exception = context.get('exception')
        if isinstance(exception, ConnectionResetError):
            # Ignorer silencieusement les erreurs de connexion reset sur Windows
            return
        # Pour les autres exceptions, utiliser le handler par défaut
        loop.default_exception_handler(context)
    
    # Configurer le handler d'exception pour la boucle d'événements
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(handle_exception)
    except RuntimeError:
        # Si aucune boucle n'existe encore, on la configurera au démarrage
        pass

app = FastAPI(
    title="CV Recommendation System",
    description="Experience-focused CV recommendation system using FAISS and LLM analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_cv.router, prefix="/api/cv", tags=["CV Processing"])

# Resolve static directory paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup_event():
    # Supprimer les warnings asyncio sur Windows pour éviter les erreurs de callback
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        # Filtrer les warnings de connexion perdue (survient souvent sur Windows)
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*")
        warnings.filterwarnings("ignore", message=".*coroutine.*")
        # Configurer le handler d'exception pour la boucle d'événements (si pas déjà fait)
        try:
            loop = asyncio.get_event_loop()
            if loop.get_exception_handler() is None:
                loop.set_exception_handler(handle_exception)
        except Exception:
            pass  # Ignorer si la boucle n'est pas encore créée
    
    settings.warmup_models()


@app.get("/", response_class=HTMLResponse)
async def root():
    # Serve the HTML interface
    if INDEX_FILE.exists():
        return FileResponse(str(INDEX_FILE))
    return JSONResponse({
        "message": "CV Recommendation System API",
        "endpoints": {
            "upload_batch": "/api/cv/upload-cv-batch",
            "recommend": "/api/cv/recommend-candidates",
            "stats": "/api/cv/stats",
            "view_all": "/api/cv/",
            "clear_data": "/api/cv/clear-all"
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
