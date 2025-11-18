# app/model_store.py
# ---------------------------------------------------------
# Parche para PyTorch 2.6+ en Render
# Se aplica ANTES de importar Ultralytics/YOLO
# ---------------------------------------------------------
import os

def _apply_torch_patches():
    """
    Parche para cargar modelos YOLO en PyTorch 2.6+.
    - Amplía la lista de clases permitidas.
    - Fuerza torch.load(weights_only=False) para compatibilidad.
    """
    try:
        import torch
        from torch.serialization import add_safe_globals
        from collections import OrderedDict

        # Contenedores básicos
        try:
            from torch.nn import Sequential, ModuleList, ModuleDict
            base_allow = [Sequential, ModuleList, ModuleDict, OrderedDict]
        except Exception:
            base_allow = [OrderedDict]

        allow = list(base_allow)

        # ---- Clases estándar de PyTorch usadas en YOLO ----
        try:
            from torch.nn.modules.activation import SiLU
            allow.append(SiLU)
        except Exception:
            pass

        try:
            from torch.nn.modules.batchnorm import BatchNorm2d
            allow.append(BatchNorm2d)
        except Exception:
            pass

        try:
            from torch.nn.modules.conv import Conv2d
            allow.append(Conv2d)
        except Exception:
            pass

        # ---- Clases típicas de Ultralytics/YOLO ----
        try:
            from ultralytics.nn.tasks import DetectionModel
            allow.append(DetectionModel)
        except Exception:
            pass

        try:
            from ultralytics.nn.modules.conv import Conv, DWConv, RepConv
            allow += [Conv, DWConv, RepConv]
        except Exception:
            pass

        try:
            from ultralytics.nn.modules.block import C2f, SPPF
            allow += [C2f, SPPF]
        except Exception:
            pass

        try:
            from ultralytics.nn.modules.head import Detect
            allow.append(Detect)
        except Exception:
            pass

        # Registrar clases en safe_globals (por si se usa el modo seguro)
        try:
            add_safe_globals(allow)
        except Exception as e:
            print(f"[torch-allowlist] Advertencia al registrar safe_globals: {e}")

        #  CRÍTICO: forzar weights_only=False en torch.load
        _orig_load = torch.load

        def _patched_load(*args, **kwargs):
            # Si el usuario no pasa weights_only, lo forzamos a False
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _patched_load

        print("[torch-allowlist] Parche activo: torch.load con weights_only=False")

    except Exception as e:
        print(f"[torch-allowlist] ERROR al aplicar parche: {e}")
        # En producción es mejor fallar aquí que arrancar sin poder cargar el modelo
        raise

# Ejecutar el parche ANTES de importar YOLO
_apply_torch_patches()

# ---------------------------------------------------------
# Resto del código original
# ---------------------------------------------------------
import requests
from ultralytics import YOLO
from .settings import settings

_model = None
_model_path = None

def _download_model_if_needed():
    """Descarga el .pt si MODEL_URL está definido, caso contrario usa almacenamiento local."""
    global _model_path
    if settings.MODEL_URL:
        dest = settings.MODEL_LOCAL_PATH
        if not os.path.exists(dest):
            print(f"[model_store] Descargando modelo desde {settings.MODEL_URL}...")
            r = requests.get(str(settings.MODEL_URL), timeout=60)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            print(f"[model_store] Modelo descargado en {dest}")
        _model_path = dest
    else:
        _model_path = settings.MODEL_LOCAL_PATH
        if not os.path.exists(_model_path):
            raise FileNotFoundError(f"Modelo no encontrado en {_model_path}")

def get_model() -> YOLO:
    """Devuelve instancia singleton del modelo YOLO."""
    global _model
    if _model is None:
        _download_model_if_needed()
        print(f"[model_store] Cargando modelo desde: {_model_path}")
        _model = YOLO(_model_path)
        print("[model_store] Modelo cargado exitosamente")
    return _model

def get_model_path() -> str:
    return _model_path or settings.MODEL_LOCAL_PATH
