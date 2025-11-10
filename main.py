from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from app.api import routes_cv
from pathlib import Path

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
