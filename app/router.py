# app/router.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
from sqlalchemy.orm import Session
import json
from datetime import datetime, timezone, timedelta  # ⬅️ NUEVO

from .settings import settings
from .image_io import pil_from_upload, pil_from_url, img_to_base64_png
from .inference import run_inference, CLASS_NAMES, CLASS_COLORS
from .model_store import get_model_path
from .schemas import AnalyzeResponse, AnalyzeUrlRequest

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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Se requiere un archivo de imagen.")

    img = pil_from_upload(await file.read())
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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Se requiere un archivo de imagen.")

    img = pil_from_upload(await file.read())
    annotated, payload = run_inference(img, confidence)
    if return_image:
        payload["image_base64"] = img_to_base64_png(annotated)
    return AnalyzeResponse(**payload)


# -------------------------------------------------------------------
# ANALYZE DESDE URL (público)
# -------------------------------------------------------------------
@router.post("/analyze-url", response_model=AnalyzeResponse, tags=["analyze"])
def analyze_url(req: AnalyzeUrlRequest):
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

        # ⬇️ Ajuste de hora a Perú
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
                # 👇 este "id" ya es el índice por usuario (1,2,3…)
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
