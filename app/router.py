from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from .settings import settings
from .image_io import pil_from_upload, pil_from_url, img_to_base64_png
from .inference import run_inference, CLASS_NAMES, CLASS_COLORS
from .model_store import get_model_path
from .schemas import AnalyzeResponse, AnalyzeUrlRequest

router = APIRouter()

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
    classes = [{"id":k,"name":v,"color_rgb":CLASS_COLORS[k]} for k,v in CLASS_NAMES.items()]
    return {
        "classes": classes,
        "default_conf_threshold": settings.DEFAULT_CONFIDENCE
    }

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    confidence: float = Form(settings.DEFAULT_CONFIDENCE),
    return_image: bool = Form(False),
    save: bool = Form(False),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Se requiere un archivo de imagen.")

    img = pil_from_upload(await file.read())
    annotated, payload = run_inference(img, confidence)

    if return_image:
        payload["image_base64"] = img_to_base64_png(annotated)

    # Guardado opcional (placeholder: ajusta a tu bucket)
    if save and settings.OUTPUT_BUCKET:
        # Aquí podrías subir annotated o solo el JSON
        # payload["saved_url"] = "https://storage.googleapis.com/...."
        pass

    return JSONResponse(content=AnalyzeResponse(**payload).model_dump())

@router.post("/analyze-url", response_model=AnalyzeResponse)
def analyze_url(req: AnalyzeUrlRequest):
    img = pil_from_url(str(req.url))
    annotated, payload = run_inference(img, req.confidence)
    if req.return_image:
        payload["image_base64"] = img_to_base64_png(annotated)
    if req.save and settings.OUTPUT_BUCKET:
        pass
    return AnalyzeResponse(**payload)
