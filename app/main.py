from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .router import router
from .settings import settings
from .model_store import get_model

app = FastAPI(title="Dental Detection API", version=settings.APP_VERSION)

origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",")] if settings.CORS_ALLOW_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    _ = get_model()

app.include_router(router, prefix="")  # raíz
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
)