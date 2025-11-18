# app/model_store.py
# ───────────────────────────────────────────────────────────────────
# Parche robusto para PyTorch 2.6+: allow-list + parche de torch.load
# Debe ejecutarse ANTES de importar Ultralytics/YOLO.
# ───────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "1")  # mantener modo seguro


def _apply_torch_patches():
    """Configura allow-list de clases seguras y envuelve torch.load con safe_globals."""
    try:
        import torch
        from torch.serialization import add_safe_globals, safe_globals

        # Contenedores estándar
        from torch.nn.modules.container import Sequential, ModuleList, ModuleDict
        from collections import OrderedDict

        # --- NUEVO: capas base de PyTorch que aparecen en el checkpoint ---
        from torch.nn.modules.conv import Conv2d
        from torch.nn.modules.batchnorm import BatchNorm2d
        # -------------------------------------------------------------------

        allow = [
            Sequential,
            ModuleList,
            ModuleDict,
            OrderedDict,
            Conv2d,         # añadida
            BatchNorm2d,    # añadida
        ]

        # ===== Clases de Ultralytics que aparecen en los .pt =====
        try:
            from ultralytics.nn.tasks import DetectionModel
            allow.append(DetectionModel)
        except Exception:
            pass

        try:
            from ultralytics.nn.modules.conv import Conv, DWConv, Focus, GhostConv, RepConv
            allow += [Conv, DWConv, Focus, GhostConv, RepConv]
        except Exception:
            pass

        try:
            from ultralytics.nn.modules.block import (
                C2f, Bottleneck, BottleneckCSP, SPPF, SPP, C3
            )
            allow += [C2f, Bottleneck, BottleneckCSP, SPPF, SPP, C3]
        except Exception:
            pass

        try:
            from ultralytics.nn.modules.head import Detect, Pose, Classify
            allow += [Detect, Pose, Classify]
        except Exception:
            pass
        # =========================================================

        # Registrar allow-list global
        try:
            add_safe_globals(allow)
        except Exception as e:
            print(f"[torch-allowlist] aviso: add_safe_globals parcial: {e}")

        # Guardar referencia original y envolver torch.load
        _orig_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            # Dejamos que PyTorch decida weights_only,
            # pero siempre con nuestras clases en safe_globals.
            with safe_globals(allow):
                return _orig_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load
        print("[torch-allowlist] parche activo (safe_globals + torch.load)")
    except Exception as e:
        print(f"[torch-allowlist] no se pudo aplicar completamente: {e}")


# Ejecutar parches ANTES de importar YOLO/Ultralytics
_apply_torch_patches()
# ───────────────────────────────────────────────────────────────────

import requests
from ultralytics import YOLO
from .settings import settings

_model = None
_model_path = None


def _download_model_if_needed():
    """Descarga el .pt si MODEL_URL está definido; si no, usa MODEL_LOCAL_PATH."""
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
        _model_path = settings.MODEL_LOCAL_PATH
        if not os.path.exists(_model_path):
            raise FileNotFoundError(f"Modelo no encontrado en {_model_path}")


def get_model() -> YOLO:
    """Singleton del modelo YOLO (la carga ya queda protegida por el parche)."""
    global _model
    if _model is None:
        _download_model_if_needed()
        _model = YOLO(_model_path)
    return _model


def get_model_path() -> str:
    return _model_path or settings.MODEL_LOCAL_PATH
