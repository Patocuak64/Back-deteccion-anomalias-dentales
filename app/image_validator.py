# app/image_validator.py
"""
Validador de imágenes para radiografías dentales.

Objetivos:
- Bloquear cosas obviamente inválidas (fotos a color, dibujos, PDFs, videos).
- Aceptar prácticamente todas las radiografías (panorámicas o no), incluso pequeñas.
- La comprobación de "panorámica" es solo informativa, NO bloquea.
"""

import io
from typing import Tuple, Optional, Dict

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────
# 1. Validación básica de archivo
# ─────────────────────────────────────────────────────────────

def validate_image_file(file_bytes: bytes, filename: str) -> Tuple[bool, str, Optional[Image.Image]]:
    """Valida que el archivo sea una imagen válida (sin límite mínimo de tamaño)."""

    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    file_lower = filename.lower()

    # Extensión
    if not any(file_lower.endswith(ext) for ext in valid_extensions):
        if file_lower.endswith(".pdf"):
            return False, (
                "Archivo PDF detectado. Por favor exporta el PDF como imagen (JPG/PNG) "
                "antes de subirlo."
            ), None
        if any(file_lower.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".wmv"]):
            return False, "Se detectó un archivo de video. Sube una imagen de radiografía.", None
        return False, (
            "Tipo de archivo no soportado. Solo se aceptan imágenes: "
            "JPG, JPEG, PNG, BMP, TIFF."
        ), None

    # Magic bytes de PDF
    if file_bytes[:4] == b"%PDF":
        return False, (
            "El archivo parece ser un PDF disfrazado de imagen. "
            "Exporta la radiografía como imagen real."
        ), None

    # Apertura con PIL
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
    except Exception:
        return False, "El archivo está corrupto o no es una imagen válida.", None

    # Sin límite mínimo. Solo máximo para evitar cosas gigantes
    width, height = img.size
    if width > 10000 or height > 10000:
        return False, f"La imagen es demasiado grande ({width}x{height}px). Máximo 10000x10000px.", None

    return True, "", img


# ─────────────────────────────────────────────────────────────
# 2. ¿Parece radiografía? (filtro con HSV + grises)
# ─────────────────────────────────────────────────────────────

def validate_is_xray(img: Image.Image) -> Tuple[bool, str, float]:
    """
    Valida si la imagen tiene características de radiografía dental:
    - casi en escala de grises (baja saturación en HSV)
    - contraste razonable
    - mezcla de tonos oscuros/medios/claros
    """

    # Convertimos a HSV para analizar saturación
    hsv = img.convert("HSV")
    hsv_arr = np.array(hsv).astype(float)
    h = hsv_arr[:, :, 0]
    s = hsv_arr[:, :, 1] / 255.0  # 0–1
    v = hsv_arr[:, :, 2]

    total_pixels = float(s.size)

    # ── 2.1 Detección de color por saturación ──
    # Radiografías: saturación muy baja casi en toda la imagen.
    # Fotos (perro, personas, etc.): muchos píxeles con saturación media/alta.

    strong_color_pixels = (s > 0.35).sum()
    strong_color_ratio = strong_color_pixels / total_pixels

    medium_color_pixels = (s > 0.20).sum()
    medium_color_ratio = medium_color_pixels / total_pixels

    mean_saturation = float(s.mean())

    # Regla combinada:
    # - si más del 25% de píxeles tienen saturación > 0.35 → foto claramente a color
    # - o si más del 60% tienen saturación > 0.20 → imagen con dominante de color
    # - o si la saturación media es muy alta (> 0.18)
    if (
        strong_color_ratio > 0.25
        or medium_color_ratio > 0.60
        or mean_saturation > 0.18
    ):
        return (
            False,
            "La imagen contiene demasiados píxeles a color. "
            "Las radiografías dentales reales se presentan en escala de grises "
            "(tonos blanco, gris y negro, con saturación muy baja).",
            mean_saturation * 100.0,
        )

    # ── 2.2 Contraste e intensidades en escala de grises ──
    gray = img.convert("L")
    g_arr = np.array(gray).astype(float)

    std = float(g_arr.std())
    if std < 12:
        return False, "La imagen tiene muy poco contraste para ser una radiografía dental útil.", std

    # Distribución de intensidades
    dark_ratio = float((g_arr < 50).mean())
    bright_ratio = float((g_arr > 200).mean())
    mid_ratio = float(((g_arr >= 50) & (g_arr <= 200)).mean())

    # Heurística para detectar dibujos tipo manga/cómic:
    # muy poco gris medio y mucho blanco/negro
    if mid_ratio < 0.25 and (dark_ratio > 0.30 or bright_ratio > 0.30):
        return (
            False,
            "La imagen parece ser un dibujo en blanco y negro (cómic o ilustración), "
            "no una radiografía dental.",
            mid_ratio * 100.0,
        )

    # Necesitamos al menos algo de zonas oscuras y claras
    if dark_ratio < 0.01 or bright_ratio < 0.002:
        return (
            False,
            "La imagen no presenta el patrón de intensidades típico de una radiografía dental.",
            (dark_ratio + bright_ratio) * 100.0,
        )

    # ✅ Es radiografía válida
    confidence = std + mid_ratio * 50.0
    return True, "", confidence


# ─────────────────────────────────────────────────────────────
# 3. Chequeo SUAVE de “parece panorámica”
#    (solo informativo, NO bloquea)
# ─────────────────────────────────────────────────────────────

def soft_check_panoramic(img: Image.Image) -> Tuple[bool, str, float]:
    """
    Chequeo suave de formato panorámico.
    No se usa para rechazar imágenes, solo para info.
    """

    width, height = img.size
    aspect_ratio = width / float(height)

    # Aceptamos muchos formatos como "panorámicos" (rango ancho/alto bastante amplio)
    if 1.4 <= aspect_ratio <= 4.0:
        return True, "", aspect_ratio

    # Fuera de ese rango, la marcamos como "no claramente panorámica"
    return False, f"Relación ancho/alto {aspect_ratio:.2f}:1 (podría no ser panorámica).", aspect_ratio


# ─────────────────────────────────────────────────────────────
# 4. Función principal que usa el endpoint
# ─────────────────────────────────────────────────────────────

def validate_dental_xray(file_bytes: bytes, filename: str) -> Tuple[bool, str, Dict]:
    """
    Función principal de validación.

    Devuelve:
        - is_valid (bool)
        - mensaje (str)
        - details (dict) con información adicional
    """

    details: Dict[str, object] = {
        "is_valid_image": False,
        "is_xray": False,
        "is_panoramic": False,          # solo informativo
        "xray_confidence": 0.0,
        "panoramic_confidence": 0.0,
    }

    # 1) ¿Es imagen válida?
    is_img, msg, pil_img = validate_image_file(file_bytes, filename)
    if not is_img or pil_img is None:
        return False, msg, details

    details["is_valid_image"] = True

    # 2) ¿Tiene pinta de radiografía?
    is_xray, xray_msg, xray_conf = validate_is_xray(pil_img)
    details["xray_confidence"] = float(xray_conf)

    if not is_xray:
        return False, xray_msg, details

    details["is_xray"] = True

    # 3) Chequeo suave de panorámica (NO bloquea)
    is_pano_like, pano_msg, pano_score = soft_check_panoramic(pil_img)
    details["is_panoramic"] = is_pano_like
    details["panoramic_confidence"] = float(pano_score)

    if is_pano_like:
        return True, "✅ Radiografía dental válida (formato compatible con panorámica).", details
    else:
        # La aceptamos igual, solo avisamos en el mensaje
        return True, (
            "✅ Radiografía dental válida. "
            "Nota: por su formato podría no ser panorámica (periapical/bitewing u otro tipo)."
        ), details
