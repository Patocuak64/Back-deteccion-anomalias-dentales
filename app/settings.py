import os
from typing import Optional
from pydantic import AnyHttpUrl  # sigue en pydantic v2
from pydantic_settings import BaseSettings  # <- viene de pydantic-settings

class Settings(BaseSettings):
    # Modelo
    MODEL_URL: Optional[AnyHttpUrl] = None
    MODEL_LOCAL_PATH: str = os.getenv("MODEL_LOCAL_PATH", "/tmp/model.pt")
    DEFAULT_CONFIDENCE: float = float(os.getenv("DEFAULT_CONFIDENCE", "0.25"))

    # CORS
    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "*")

    # Outputs opcionales
    SAVE_OUTPUTS: bool = os.getenv("SAVE_OUTPUTS", "false").lower() == "true"
    OUTPUT_BUCKET: Optional[str] = os.getenv("OUTPUT_BUCKET")

    # App
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")

    # JWT configuración
    JWT_SECRET: str = os.getenv("JWT_SECRET", "tu_clave_secreta_aqui")  # Define una clave secreta para JWT
    JWT_ALGORITHM: str = "HS256"  # El algoritmo de encriptación del token JWT
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # Tiempo de expiración del token en minutos

    class Config:
        env_file = ".env"  # opcional, si quieres cargar .env automáticamente

settings = Settings()
