# app/settings.py (OPTIMIZADO)
import os
from typing import Optional
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DE MODELO
    # ═══════════════════════════════════════════════════════════════════
    MODEL_URL: Optional[AnyHttpUrl] = None
    MODEL_LOCAL_PATH: str = os.getenv("MODEL_LOCAL_PATH", "/tmp/model.pt")
    DEFAULT_CONFIDENCE: float = float(os.getenv("DEFAULT_CONFIDENCE", "0.25"))
    
    # ⚡ OPTIMIZACIÓN: Cachear modelo en memoria (no recargar)
    MODEL_CACHE_ENABLED: bool = True

    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DE CORS
    # ═══════════════════════════════════════════════════════════════════
    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "*")

    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DE OUTPUTS
    # ═══════════════════════════════════════════════════════════════════
    SAVE_OUTPUTS: bool = os.getenv("SAVE_OUTPUTS", "false").lower() == "true"
    OUTPUT_BUCKET: Optional[str] = os.getenv("OUTPUT_BUCKET")

    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DE LA APP
    # ═══════════════════════════════════════════════════════════════════
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")

    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN JWT (OPTIMIZADA)
    # ═══════════════════════════════════════════════════════════════════
    JWT_SECRET: str = os.getenv("JWT_SECRET", "mandarina")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 horas

    # ═══════════════════════════════════════════════════════════════════
    # ⚡ OPTIMIZACIÓN: BCRYPT ROUNDS (menos rounds = más rápido)
    # ═══════════════════════════════════════════════════════════════════
    # PRODUCCIÓN: 12-14 rounds (más seguro, más lento)
    # DESARROLLO: 4-6 rounds (menos seguro, MÁS RÁPIDO)
    BCRYPT_ROUNDS: int = int(os.getenv("BCRYPT_ROUNDS", "8"))  # Balance seguridad/velocidad
    
    # ═══════════════════════════════════════════════════════════════════
    # ⚡ OPTIMIZACIÓN: DATABASE
    # ═══════════════════════════════════════════════════════════════════
    # Pool de conexiones para mejor rendimiento
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    
    # ═══════════════════════════════════════════════════════════════════
    # ⚡ OPTIMIZACIÓN: CACHING
    # ═══════════════════════════════════════════════════════════════════
    # Cachear resultados de análisis recientes (en memoria)
    ENABLE_RESULT_CACHE: bool = os.getenv("ENABLE_RESULT_CACHE", "true").lower() == "true"
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutos
    
    # ═══════════════════════════════════════════════════════════════════
    # ⚡ OPTIMIZACIÓN: COMPRESIÓN DE IMÁGENES
    # ═══════════════════════════════════════════════════════════════════
    # Reducir tamaño de imágenes antes de análisis
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "2048"))  # píxeles
    IMAGE_QUALITY: int = int(os.getenv("IMAGE_QUALITY", "85"))  # 0-100
    
    # ═══════════════════════════════════════════════════════════════════
    # ⚡ OPTIMIZACIÓN: TIMEOUTS
    # ═══════════════════════════════════════════════════════════════════
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))  # segundos
    INFERENCE_TIMEOUT: int = int(os.getenv("INFERENCE_TIMEOUT", "10"))  # segundos

    class Config:
        env_file = ".env"

settings = Settings()


# ═══════════════════════════════════════════════════════════════════════
# HELPER: Información de optimización
# ═══════════════════════════════════════════════════════════════════════
def print_optimization_settings():
    """Muestra configuración de optimización al iniciar"""
    print("=" * 70)
    print("⚡ CONFIGURACIÓN DE OPTIMIZACIÓN")
    print("=" * 70)
    print(f"✓ Model cache enabled: {settings.MODEL_CACHE_ENABLED}")
    print(f"✓ Bcrypt rounds: {settings.BCRYPT_ROUNDS} (menor = más rápido)")
    print(f"✓ Result cache: {settings.ENABLE_RESULT_CACHE}")
    print(f"✓ Cache TTL: {settings.CACHE_TTL_SECONDS}s")
    print(f"✓ Max image size: {settings.MAX_IMAGE_SIZE}px")
    print(f"✓ Image quality: {settings.IMAGE_QUALITY}%")
    print(f"✓ DB pool size: {settings.DB_POOL_SIZE}")
    print(f"✓ Request timeout: {settings.REQUEST_TIMEOUT}s")
    print("=" * 70)