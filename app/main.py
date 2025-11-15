# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import router                    # rutas de análisis e historial
from .settings import settings
from .model_store import get_model

# ════════════════════════════════════════════════════════════════════
# NUEVAS IMPORTACIONES PARA BASE DE DATOS AND AUTENTICACIÓN
# ════════════════════════════════════════════════════════════════════
from . import models
from .database import engine
from .auth import router as auth_router


# ════════════════════════════════════════════════════════════════════
# CREAR TABLAS DE BASE DE DATOS AL INICIAR
#   Nota: si ya tenías dental.db sin los campos nuevos
#   (per_user_index, teeth_fdi_json, etc.), borra el archivo
#   dental.db para que se regenere con el nuevo esquema.
# ════════════════════════════════════════════════════════════════════
print("🗄️  Creando tablas de base de datos...")
models.Base.metadata.create_all(bind=engine)
print("✅ Tablas creadas: users, analyses")


# Crear aplicación FastAPI
app = FastAPI(
    title="Dental Detection API",
    version=settings.APP_VERSION,
    description=(
        "API de detección dental con autenticación y almacenamiento "
        "de historial por usuario"
    ),
)


# Configurar CORS
origins = (
    [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",")]
    if settings.CORS_ALLOW_ORIGINS
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # necesario para JWT
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════
# EVENTO DE INICIO
# ════════════════════════════════════════════════════════════════════
@app.on_event("startup")
def startup_event():
    """Se ejecuta al iniciar la aplicación"""
    _ = get_model()
    print("✅ Modelo YOLO cargado")
    print("✅ Base de datos SQLite lista (dental.db)")
    print("✅ API corriendo en http://localhost:8080")
    print("📚 Documentación en http://localhost:8080/docs")


# ════════════════════════════════════════════════════════════════════
# INCLUSIÓN DE ROUTERS
# ════════════════════════════════════════════════════════════════════
app.include_router(auth_router)  # /auth/register, /auth/login, etc.
app.include_router(router)       # /analyze, /analyze-public, /analyses, ...


# ════════════════════════════════════════════════════════════════════
# ENDPOINT RAÍZ CON INFORMACIÓN
# ════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA (para ejecución directa)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn, os

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=True,  # modo desarrollo
    )
