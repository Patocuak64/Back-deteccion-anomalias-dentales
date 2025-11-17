# app/schemas.py

from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]
    fdi: Optional[int] = None  
    tooth_fdi: Optional[int] = None  

class AnalyzeResponse(BaseModel):
    summary: Dict
    detections: List[Detection]
    stats: Dict
    report_text: str
    teeth_fdi: Optional[Dict[str, List[int]]] = None  
    image_base64: Optional[str] = None
    saved_url: Optional[HttpUrl] = None

class AnalyzeUrlRequest(BaseModel):
    url: HttpUrl
    confidence: float = 0.25
    return_image: bool = False
    save: bool = False