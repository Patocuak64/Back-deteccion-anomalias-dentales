# app/router.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List
from sqlalchemy.orm import Session

from .settings import settings
from .image_io import pil_from_upload, pil_from_url, img_to_base64_png
from .inference import run_inference, CLASS_NAMES, CLASS_COLORS
from .model_store import get_model_path
from .schemas import AnalyzeResponse, AnalyzeUrlRequest

# 👇 NUEVO: imports para auth, BD y modelos
from .dependencies import get_db
from .auth import get_current_user
from . import models

router = APIRouter()

# ------------------------
# INFO BÁSICA
# ------------------------
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
    classes = [{"id": k, "name": v, "color_rgb": CLASS_COLORS[k]} for k, v in CLASS_NAMES.items()]
    return {"classes": classes, "default_conf_threshold": settings.DEFAULT_CONFIDENCE}

# ------------------------
# ANALYZE (requiere login) + guarda en BD si save=true
# ------------------------
@router.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
async def analyze(
    file: UploadFile = File(...),
    confidence: float = Form(settings.DEFAULT_CONFIDENCE),
    return_image: bool = Form(False),
    save: bool = Form(False),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),  # 🔒 requiere token
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Se requiere un archivo de imagen.")

    img = pil_from_upload(await file.read())
    annotated, payload = run_inference(img, confidence)

    if return_image:
        payload["image_base64"] = img_to_base64_png(annotated)

    # 💾 Guardado opcional del análisis
    if save:
        detections = payload.get("detections", []) or []
        def _count(cls): return sum(1 for d in detections if d.get("class_name") == cls)

        row = models.Analysis(
            user_id=user.id,
            image_filename=file.filename,
            model_used=str(payload.get("model", "best.pt")),
            confidence=confidence,
            total_detections=len(detections),
            caries_count=_count("Caries"),
            diente_retenido_count=_count("Diente_Retenido"),
            perdida_osea_count=_count("Perdida_Osea"),
            results_json=AnalyzeResponse(**payload).model_dump_json(),
            report_text=(payload.get("summary") or {}).get("text") if isinstance(payload.get("summary"), dict) else None,
        )
        db.add(row); db.commit()

    return JSONResponse(content=AnalyzeResponse(**payload).model_dump())

# ------------------------
# ANALYZE PÚBLICO (no requiere login, no guarda)
# ------------------------
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

# ------------------------
# ANALYZE DESDE URL (opcional login; aquí lo dejamos público)
# ------------------------
@router.post("/analyze-url", response_model=AnalyzeResponse, tags=["analyze"])
def analyze_url(req: AnalyzeUrlRequest):
    img = pil_from_url(str(req.url))
    annotated, payload = run_inference(img, req.confidence)
    if req.return_image:
        payload["image_base64"] = img_to_base64_png(annotated)
    return AnalyzeResponse(**payload)

# ------------------------
# HISTORIAL (requiere login)
# ------------------------
@router.get("/analyses", tags=["history"])
def list_analyses(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    rows = (db.query(models.Analysis)
              .filter(models.Analysis.user_id == user.id)
              .order_by(models.Analysis.created_at.desc())
              .all())
    return [
        {
            "id": r.id,
            "created_at": r.created_at,
            "image_filename": r.image_filename,
            "model_used": r.model_used,
            "total_detections": r.total_detections,
            "caries": r.caries_count,
            "retenido": r.diente_retenido_count,
            "perdida": r.perdida_osea_count,
        } for r in rows
    ]

@router.delete("/analyses/{analysis_id}", tags=["history"])
def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    r = (db.query(models.Analysis)
            .filter(models.Analysis.id == analysis_id,
                    models.Analysis.user_id == user.id)
            .first())
    if not r:
        raise HTTPException(404, "No encontrado")
    db.delete(r); db.commit()
    return {"deleted": analysis_id}
