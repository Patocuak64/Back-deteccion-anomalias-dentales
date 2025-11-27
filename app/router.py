# app/router.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
from sqlalchemy.orm import Session
import json
from datetime import datetime, timezone, timedelta

from .settings import settings
from .image_io import pil_from_upload, pil_from_url, img_to_base64_png
from .inference import run_inference, CLASS_NAMES, CLASS_COLORS
from .model_store import get_model_path
from .schemas import AnalyzeResponse, AnalyzeUrlRequest
from .image_validator import validate_dental_xray  # ⚡ NUEVO

# auth + BD + modelos
from .dependencies import get_db
from .auth import get_current_user
from . import models

router = APIRouter()

# Zona horaria Perú (UTC-5)
PERU_TZ = timezone(timedelta(hours=-5))


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def build_teeth_fdi_from_detections(detections):
    """
    Intenta construir un dict { clase: [FDI...] } a partir de detections.
    Busca keys típicas: 'tooth_fdi', 'tooth', 'fdi'.
    """
    result = {
        "Caries": [],
        "Diente_Retenido": [],
        "Perdida_Osea": [],
    }
    if not detections:
        return result

    for d in detections:
        cls = d.get("class_name") or d.get("cls_name")
        if not cls:
            continue

        tooth = d.get("tooth_fdi") or d.get("tooth") or d.get("fdi")
        if tooth is None:
            continue

        # intenta convertir a int si es string
        try:
            tooth_int = int(tooth)
        except Exception:
            tooth_int = tooth

        result.setdefault(cls, [])
        if tooth_int not in result[cls]:
            result[cls].append(tooth_int)

    return result


# -------------------------------------------------------------------
# INFO BÁSICA
# -------------------------------------------------------------------
@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": get_model_path(),
        "model_classes": CLASS_NAMES,
        "version": settings.APP_VERSION,
    }


@router.get("/metadata")
def metadata():
    classes = [
        {"id": k, "name": v, "color_rgb": CLASS_COLORS[k]} for k, v in CLASS_NAMES.items()
    ]
    return {"classes": classes, "default_conf_threshold": settings.DEFAULT_CONFIDENCE}


# -------------------------------------------------------------------
# ANALYZE (requiere login) + guarda en BD si save=true
# -------------------------------------------------------------------
@router.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
async def analyze(
    file: UploadFile = File(...),
    confidence: float = Form(settings.DEFAULT_CONFIDENCE),
    return_image: bool = Form(False),
    save: bool = Form(False),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    # ⚡ VALIDACIÓN 1: Tipo de contenido básico
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Se requiere un archivo de imagen (JPG, PNG, etc.)"
        )
    
    # Leer archivo
    file_bytes = await file.read()
    
    # ⚡ VALIDACIÓN 2: Validación completa de imagen radiográfica
    print(f"[IMAGE] Validando archivo: {file.filename}")
    is_valid, error_msg, validation_details = validate_dental_xray(
        file_bytes, 
        file.filename
    )
    
    if not is_valid:
        print(f"[IMAGE] ❌ Imagen rechazada: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=error_msg  # Mensaje claro para el usuario
        )
    
    print(f"[IMAGE] ✅ Imagen válida (X-ray: {validation_details['xray_confidence']:.1f}%, Panoramic: {validation_details['panoramic_confidence']:.1f}%)")
    
    # ✅ Si llega aquí, la imagen es válida
    # Continuar con análisis YOLO normal
    img = pil_from_upload(file_bytes)
    annotated, payload = run_inference(img, confidence)

    detections = payload.get("detections", []) or []

    if return_image:
        payload["image_base64"] = img_to_base64_png(annotated)

    # ----------------------------------------------------------------
    # Guardado opcional del análisis
    # ----------------------------------------------------------------
    if save:

        def _count(cls_name: str) -> int:
            return sum(1 for d in detections if d.get("class_name") == cls_name)

        # índice por usuario (1,2,3...) solo dentro de esa cuenta
        last_idx_row = (
            db.query(models.Analysis.per_user_index)
            .filter(models.Analysis.user_id == user.id)
            .order_by(models.Analysis.per_user_index.desc())
            .first()
        )
        next_idx = (last_idx_row[0] if last_idx_row and last_idx_row[0] else 0) + 1

        # mapa de dientes FDI
        teeth_map = (
            payload.get("teeth_fdi")
            or payload.get("teeth_fdi_map")
            or build_teeth_fdi_from_detections(detections)
        )

        row = models.Analysis(
            user_id=user.id,
            per_user_index=next_idx,
            image_filename=file.filename,
            image_base64=payload.get("image_base64"),
            model_used=str(payload.get("model", "best.pt")),
            confidence=confidence,
            total_detections=len(detections),
            caries_count=_count("Caries"),
            diente_retenido_count=_count("Diente_Retenido"),
            perdida_osea_count=_count("Perdida_Osea"),
            results_json=AnalyzeResponse(**payload).model_dump_json(),
            teeth_fdi_json=json.dumps(teeth_map, ensure_ascii=False),
            report_text=(
                (payload.get("summary") or {}).get("text")
                if isinstance(payload.get("summary"), dict)
                else None
            ),
        )
        db.add(row)
        db.commit()

    return JSONResponse(content=AnalyzeResponse(**payload).model_dump())


# -------------------------------------------------------------------
# ANALYZE PÚBLICO (no requiere login, no guarda)
# -------------------------------------------------------------------
@router.post("/analyze-public", response_model=AnalyzeResponse, tags=["analyze"])
async def analyze_public(
    file: UploadFile = File(...),
    confidence: float = Form(settings.DEFAULT_CONFIDENCE),
    return_image: bool = Form(False),
):
    # ⚡ VALIDACIÓN 1: Tipo de contenido básico
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Se requiere un archivo de imagen (JPG, PNG, etc.)"
        )
    
    # Leer archivo
    file_bytes = await file.read()
    
    # ⚡ VALIDACIÓN 2: Validación completa de imagen radiográfica
    print(f"[IMAGE] Validando archivo: {file.filename}")
    is_valid, error_msg, validation_details = validate_dental_xray(
        file_bytes, 
        file.filename
    )
    
    if not is_valid:
        print(f"[IMAGE] ❌ Imagen rechazada: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=error_msg
        )
    
    print(f"[IMAGE] ✅ Imagen válida")
    
    # ✅ Continuar con análisis YOLO
    img = pil_from_upload(file_bytes)
    annotated, payload = run_inference(img, confidence)
    if return_image:
        payload["image_base64"] = img_to_base64_png(annotated)
    return AnalyzeResponse(**payload)


# -------------------------------------------------------------------
# ANALYZE DESDE URL (público)
# -------------------------------------------------------------------
@router.post("/analyze-url", response_model=AnalyzeResponse, tags=["analyze"])
def analyze_url(req: AnalyzeUrlRequest):
    """
    NOTA: Este endpoint NO valida si es radiografía panorámica
    porque la imagen viene de URL externa sin acceso al archivo original.
    Se recomienda usar /analyze o /analyze-public para validación completa.
    """
    img = pil_from_url(str(req.url))
    annotated, payload = run_inference(img, req.confidence)
    if req.return_image:
        payload["image_base64"] = img_to_base64_png(annotated)
    return AnalyzeResponse(**payload)


# -------------------------------------------------------------------
# HISTORIAL (requiere login)
# -------------------------------------------------------------------
@router.get("/analyses", tags=["history"])
def list_analyses(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    rows = (
        db.query(models.Analysis)
        .filter(models.Analysis.user_id == user.id)
        .order_by(models.Analysis.created_at.desc())
        .all()
    )

    result = []
    for r in rows:
        teeth_map = {}
        if r.teeth_fdi_json:
            try:
                teeth_map = json.loads(r.teeth_fdi_json)
            except Exception:
                teeth_map = {}

        #  Ajuste de hora a Perú
        if r.created_at:
            created = r.created_at
            # Si no tiene tzinfo, asumimos que está en UTC
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            created_local = created.astimezone(PERU_TZ)
            created_iso = created_local.isoformat()
            created_display = created_local.strftime("%d/%m/%Y, %H:%M:%S")
        else:
            created_iso = None
            created_display = None

        result.append(
            {
                
                "id": r.per_user_index or r.id,
                "analysis_id": r.id,
                "per_user_index": r.per_user_index,
                "created_at": created_iso,          # ahora en hora local
                "created_at_display": created_display,  # string listo para mostrar
                "image_filename": r.image_filename,
                "model_used": r.model_used,
                # campos originales
                "total_detections": r.total_detections,
                "caries": r.caries_count,
                "retenido": r.diente_retenido_count,
                "perdida": r.perdida_osea_count,
                # alias para el frontend nuevo
                "total": r.total_detections,
                "osea": r.perdida_osea_count,
                "teeth_fdi": teeth_map,
                "image_base64": r.image_base64,
            }
        )

    return result


@router.delete("/analyses/{analysis_id}", tags=["history"])
def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    r = (
        db.query(models.Analysis)
        .filter(models.Analysis.id == analysis_id, models.Analysis.user_id == user.id)
        .first()
    )
    if not r:
        raise HTTPException(404, "No encontrado")

    db.delete(r)
    db.commit()
    return {"deleted": analysis_id}


# ═══════════════════════════════════════════════════════════════════════════
#  Información sobre dientes FDI
# ═══════════════════════════════════════════════════════════════════════════
@router.get("/fdi-info/{fdi_number}", tags=["info"])
def get_fdi_info(fdi_number: int):
    """
    Obtiene información detallada sobre un número FDI específico.
    
    Args:
        fdi_number: Número FDI del diente (11-48)
    
    Returns:
        Información del diente (cuadrante, posición, nombre)
    """
    if fdi_number < 11 or fdi_number > 48 or fdi_number % 10 == 0 or fdi_number % 10 > 8:
        raise HTTPException(400, "Número FDI inválido. Debe ser 11-18, 21-28, 31-38 o 41-48")
    
    quadrant = fdi_number // 10
    position = fdi_number % 10
    
    quadrant_names = {
        1: "Superior Derecho",
        2: "Superior Izquierdo",
        3: "Inferior Izquierdo",
        4: "Inferior Derecho"
    }
    
    tooth_names = {
        1: "Incisivo Central",
        2: "Incisivo Lateral",
        3: "Canino",
        4: "Primer Premolar",
        5: "Segundo Premolar",
        6: "Primer Molar",
        7: "Segundo Molar",
        8: "Tercer Molar (Muela del Juicio)"
    }
    
    tooth_type = "Permanente"
    if quadrant in [5, 6, 7, 8]:
        tooth_type = "Temporal (Deciduo)"
    
    return {
        "fdi": fdi_number,
        "quadrant": quadrant,
        "quadrant_name": quadrant_names.get(quadrant, "Desconocido"),
        "position": position,
        "tooth_name": tooth_names.get(position, "Desconocido"),
        "tooth_type": tooth_type,
        "full_name": f"{tooth_names.get(position, 'Desconocido')} {quadrant_names.get(quadrant, 'Desconocido')}",
        "description": f"Diente {fdi_number} - {tooth_names.get(position)} del cuadrante {quadrant_names.get(quadrant)}"
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Mapa completo de dientes FDI
# ═══════════════════════════════════════════════════════════════════════════
@router.get("/fdi-map", tags=["info"])
def get_fdi_map():
    """
    Retorna el mapa completo de la numeración FDI.
    """
    quadrants = {
        1: {"name": "Superior Derecho", "teeth": list(range(11, 19))},
        2: {"name": "Superior Izquierdo", "teeth": list(range(21, 29))},
        3: {"name": "Inferior Izquierdo", "teeth": list(range(31, 39))},
        4: {"name": "Inferior Derecho", "teeth": list(range(41, 49))},
    }
    
    tooth_names = {
        1: "Incisivo Central",
        2: "Incisivo Lateral",
        3: "Canino",
        4: "Primer Premolar",
        5: "Segundo Premolar",
        6: "Primer Molar",
        7: "Segundo Molar",
        8: "Tercer Molar"
    }
    
    return {
        "quadrants": quadrants,
        "tooth_positions": tooth_names,
        "total_permanent_teeth": 32,
        "description": "Sistema FDI de numeración dental internacional"
    }

# ═══════════════════════════════════════════════════════════════════════════
# GESTIÓN Y COMPARACIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/models/available", tags=["models"])
def list_available_models():
    """
    Retorna lista de modelos disponibles con métricas completas
    """
    models_data = [
        {
            "id": "best",
            "name": "best.pt",
            "architecture": "YOLOv8n/s",
            "version": "v8",
            "epochs": 100,
            "status": "production",
            "is_active": True,
            "metrics": {
                "map50": 0.790,
                "map50_95": 0.520,
                "precision": 0.790,
                "recall": 0.750
            },
            "per_class_metrics": {
                "Caries": {"precision": 0.82, "recall": 0.78, "map50": 0.80},
                "Diente_Retenido": {"precision": 0.75, "recall": 0.85, "map50": 0.82},
                "Perdida_Osea": {"precision": 0.80, "recall": 0.62, "map50": 0.75}
            },
            "training_info": {
                "duration_hours": 2.5,
                "train_images": 19308,
                "val_images": 2725,
                "date": "2025-10"
            },
            "description": "Modelo principal en producción. Mejor equilibrio entre precisión y velocidad."
        },
        {
            "id": "yolov11m",
            "name": "yolov11m_best.pt",
            "architecture": "YOLOv11m",
            "version": "v11",
            "epochs": 40,
            "status": "experimental",
            "is_active": False,
            "metrics": {
                "map50": 0.581,
                "map50_95": 0.365,
                "precision": 0.554,
                "recall": 0.590
            },
            "per_class_metrics": {
                "Caries": {"precision": 0.596, "recall": 0.550, "map50": 0.562},
                "Diente_Retenido": {"precision": 0.530, "recall": 0.947, "map50": 0.824},
                "Perdida_Osea": {"precision": 0.534, "recall": 0.273, "map50": 0.356}
            },
            "training_info": {
                "duration_hours": 2.62,
                "train_images": 19308,
                "val_images": 2725,
                "early_stopping": 28,
                "date": "2025-11"
            },
            "description": "Versión 11 Medium. Early stopping en epoch 28. Excelente recall en dientes retenidos."
        },
        {
            "id": "yolov11l",
            "name": "yolov11l_best.pt",
            "architecture": "YOLOv11l",
            "version": "v11",
            "epochs": 40,
            "status": "experimental",
            "is_active": False,
            "metrics": {
                "map50": 0.575,
                "map50_95": 0.367,
                "precision": 0.543,
                "recall": 0.590
            },
            "per_class_metrics": {
                "Caries": {"precision": 0.624, "recall": 0.539, "map50": 0.568},
                "Diente_Retenido": {"precision": 0.548, "recall": 0.944, "map50": 0.818},
                "Perdida_Osea": {"precision": 0.458, "recall": 0.287, "map50": 0.338}
            },
            "training_info": {
                "duration_hours": 5.68,
                "train_images": 19308,
                "val_images": 2725,
                "date": "2025-11"
            },
            "description": "Modelo Large v11. Mejor precisión en caries pero entrenamiento más lento."
        },
        {
            "id": "yolov10m",
            "name": "yolov10m_best.pt",
            "architecture": "YOLOv10m",
            "version": "v10",
            "epochs": 40,
            "status": "experimental",
            "is_active": False,
            "metrics": {
                "map50": 0.513,
                "map50_95": 0.315,
                "precision": 0.445,
                "recall": 0.547
            },
            "per_class_metrics": {
                "Caries": {"precision": 0.561, "recall": 0.450, "map50": 0.475},
                "Diente_Retenido": {"precision": 0.456, "recall": 0.940, "map50": 0.795},
                "Perdida_Osea": {"precision": 0.319, "recall": 0.243, "map50": 0.269}
            },
            "training_info": {
                "duration_hours": 2.37,
                "train_images": 19308,
                "val_images": 2725,
                "date": "2025-11"
            },
            "description": "Modelo v10 Medium. Menor rendimiento general pero rápido."
        },
        {
            "id": "yolov8x",
            "name": "yolov8x_best.pt",
            "architecture": "YOLOv8x",
            "version": "v8",
            "epochs": 28,
            "status": "incomplete",
            "is_active": False,
            "metrics": {
                "map50": 0.562,
                "map50_95": 0.345,
                "precision": 0.537,
                "recall": 0.567
            },
            "per_class_metrics": {
                "Caries": {"precision": 0.58, "recall": 0.52, "map50": 0.55},
                "Diente_Retenido": {"precision": 0.52, "recall": 0.89, "map50": 0.78},
                "Perdida_Osea": {"precision": 0.51, "recall": 0.29, "map50": 0.36}
            },
            "training_info": {
                "duration_hours": 3.0,
                "train_images": 19308,
                "val_images": 2725,
                "interrupted": True,
                "target_epochs": 50,
                "date": "2025-11"
            },
            "description": " Entrenamiento interrumpido en epoch 28/50. No usar en producción."
        }
    ]
    
    return {
        "models": models_data,
        "total": len(models_data),
        "active_model": next((m["id"] for m in models_data if m["is_active"]), "best")
    }


@router.get("/models/comparison", tags=["models"])
def get_models_comparison():
    """
    Retorna tabla comparativa de modelos
    """
    models = list_available_models()["models"]
    
    comparison = {
        "headers": ["Modelo", "Arquitectura", "mAP50", "mAP50-95", "Precision", "Recall", "Tiempo (h)", "Estado"],
        "rows": []
    }
    
    for model in models:
        row = {
            "id": model["id"],
            "name": model["name"],
            "architecture": model["architecture"],
            "map50": model["metrics"]["map50"],
            "map50_95": model["metrics"]["map50_95"],
            "precision": model["metrics"]["precision"],
            "recall": model["metrics"]["recall"],
            "duration": model["training_info"]["duration_hours"],
            "status": model["status"],
            "is_active": model["is_active"]
        }
        comparison["rows"].append(row)
    
    # Ordenar por mAP50 descendente
    comparison["rows"].sort(key=lambda x: x["map50"], reverse=True)
    
    return comparison


@router.post("/models/set-active/{model_id}", tags=["models"])
def set_active_model(model_id: str, user: models.User = Depends(get_current_user)):
    """
    Cambia el modelo activo (requiere autenticación)
    """
    from fastapi import HTTPException
    
    valid_models = ["best", "yolov11m", "yolov11l", "yolov10m", "yolov8x"]
    
    if model_id not in valid_models:
        raise HTTPException(status_code=400, detail=f"Modelo inválido. Opciones: {valid_models}")
    
    if model_id == "yolov8x":
        raise HTTPException(
            status_code=400, 
            detail="Este modelo tiene entrenamiento incompleto y no puede usarse en producción"
        )
    
    return {
        "success": True,
        "message": f"Modelo {model_id} establecido como activo",
        "model_id": model_id,
        "note": " Implementación simplificada. En producción debes recargar el modelo en inference.py"
    }