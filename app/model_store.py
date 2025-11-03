import os, requests
from ultralytics import YOLO
from .settings import settings

_model = None
_model_path = None

def _download_model_if_needed():
    global _model_path
    if settings.MODEL_URL:
        dest = settings.MODEL_LOCAL_PATH
        if not os.path.exists(dest):
            r = requests.get(str(settings.MODEL_URL), timeout=60)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
        _model_path = dest
    else:
        # Si prefieres montar el modelo por volumen o imagen
        _model_path = settings.MODEL_LOCAL_PATH
        if not os.path.exists(_model_path):
            raise FileNotFoundError(f"Modelo no encontrado en {_model_path}")

def get_model() -> YOLO:
    global _model
    if _model is None:
        _download_model_if_needed()
        _model = YOLO(_model_path)
    return _model

def get_model_path() -> str:
    return _model_path or settings.MODEL_LOCAL_PATH
