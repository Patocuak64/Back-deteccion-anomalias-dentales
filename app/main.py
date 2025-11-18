# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import router
from .settings import settings
from .model_store import get_model

# BASE DE DATOS Y AUTENTICACIÓN
from . import models
from .database import engine
from .auth import router as auth_router

# CREAR TABLAS DE BASE DE DATOS AL INICIAR
print("Creando tablas de base de datos...")
models.Base.metadata.create_all(bind=engine)
print("Tablas creadas: users, analyses")

# APLICACIÓN FASTAPI
app = FastAPI(
    title="Dental Detection API",
    version=settings.APP_VERSION,
    description=(
        "API de detección dental con autenticación y almacenamiento "
        "de historial por usuario"
    ),
)

# CONFIGURACIÓN DE CORS
# Lista base de orígenes permitidos
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "https://front-deteccion-dental.vercel.app",
]

# Orígenes adicionales desde variable de entorno (si existe)
env_origins = []
if settings.CORS_ALLOW_ORIGINS:
    env_origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()]

# Combinar ambas listas y eliminar duplicados
all_origins = list(set(default_origins + env_origins))

print(f"CORS allow_origins = {all_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=all_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EVENTO DE INICIO
@app.on_event("startup")
def startup_event():
    """Se ejecuta al iniciar la aplicación"""
    _ = get_model()
    print("Modelo YOLO cargado")
    print("Base de datos SQLite lista (dental.db)")
    print("API corriendo")
    print("Documentación en /docs")

# INCLUSIÓN DE ROUTERS
app.include_router(auth_router)  # /auth/register, /auth/login, etc.
app.include_router(router)       # /analyze, /analyze-public, /analyses, ...

# ENDPOINT RAÍZ
@app.get("/")
def root():
    """Información general de la API"""
    return {
        "message": "Dental Detection API",
        "version": settings.APP_VERSION,
        "features": [
            "Detección de caries, dientes retenidos, pérdida ósea",
            "Autenticación con JWT",
            "Historial de análisis por usuario",
            "Resumen FDI por diente y patología",
        ],
        "endpoints": {
            "docs": "/docs",
            "register": "/auth/register",
            "login": "/auth/login",
            "analyze": "/analyze",
            "history": "/analyses",
        },
    }

# PUNTO DE ENTRADA LOCAL
if __name__ == "__main__":
    import uvicorn, os

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=True,
    )