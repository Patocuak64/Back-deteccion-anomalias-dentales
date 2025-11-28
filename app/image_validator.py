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

    if file_bytes[:4] == b"%PDF":
        return False, (
            "El archivo parece ser un PDF disfrazado de imagen. "
            "Exporta la radiografía como imagen real."
        ), None

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
    except Exception:
        return False, "El archivo está corrupto o no es una imagen válida.", None

    width, height = img.size
    if width > 10000 or height > 10000:
        return False, f"La imagen es demasiado grande ({width}x{height}px). Máximo 10000x10000px.", None

    return True, "", img

# ─────────────────────────────────────────────────────────────
# 2. ¿Parece radiografía? (HSV + umbrales ajustados) ⚡
# ─────────────────────────────────────────────────────────────

def validate_is_xray(img: Image.Image) -> Tuple[bool, str, float]:
    """
    Valida si la imagen tiene características de radiografía dental.
    
    MEJORAS:
    - Umbrales de saturación ajustados para rechazar fotos de animales
    - strong_color: 25% → 10%
    - medium_color: 60% → 40%
    - mean_saturation: 0.18 → 0.10
    """

    # Convertimos a HSV para analizar saturación
    hsv = img.convert("HSV")
    hsv_arr = np.array(hsv).astype(float)
    s = hsv_arr[:, :, 1] / 255.0  # Saturación 0-1
    v = hsv_arr[:, :, 2]

    total_pixels = float(s.size)

    # ── 2.1 Calcular métricas de color ──
    
    # Píxeles con color fuerte (S > 0.35)
    strong_color_pixels = (s > 0.35).sum()
    strong_color_ratio = strong_color_pixels / total_pixels
    
    # Píxeles con color medio (S > 0.20)
    medium_color_pixels = (s > 0.20).sum()
    medium_color_ratio = medium_color_pixels / total_pixels
    
    # Saturación promedio
    mean_saturation = float(s.mean())

    # ── 2.2 LÓGICA MULTI-CRITERIO (más estricta) ──
    
    # CRITERIO 1: Saturación promedio MUY alta → foto claramente a color
    if mean_saturation > 0.10:
        return (
            False,
            f"La imagen tiene saturación muy alta ({mean_saturation*100:.1f}%). "
            "Las radiografías dentales son en escala de grises.",
            mean_saturation * 100.0,
        )
    
    # CRITERIO 2: Muchos píxeles MUY coloridos → foto con objetos de color
    if strong_color_ratio > 0.10:
        return (
            False,
            f"La imagen contiene muchos píxeles con colores fuertes ({strong_color_ratio*100:.1f}%). "
            "Las radiografías dentales no tienen objetos de colores vivos.",
            strong_color_ratio * 100.0,
        )
    
    # CRITERIO 3: Saturación media + MUCHOS píxeles con color medio
    # (Esto detecta fotos pero tolera radiografías con leve tinte)
    if mean_saturation > 0.05 and medium_color_ratio > 0.40:
        return (
            False,
            f"La imagen tiene saturación media ({mean_saturation*100:.1f}%) "
            f"con muchos píxeles coloridos ({medium_color_ratio*100:.1f}%). "
            "Esto es característico de fotografías, no radiografías.",
            mean_saturation * 100.0,
        )

    # CRITERIO 4: Saturación muy baja pero MUCHOS píxeles coloridos
    # (Detecta fotos con dominante gris pero accesorios coloridos)
    if medium_color_ratio > 0.45:
        return (
            False,
            f"La imagen tiene demasiados píxeles con color ({medium_color_ratio*100:.1f}%). "
            "Las radiografías dentales tienen saturación casi nula en toda la imagen.",
            medium_color_ratio * 100.0,
        )

    # ── 2.3 Contraste e intensidades en escala de grises ──
    gray = img.convert("L")
    g_arr = np.array(gray).astype(float)

    std = float(g_arr.std())
    if std < 12:
        return False, "La imagen tiene muy poco contraste para ser una radiografía dental útil.", std

    # Distribución de intensidades
    dark_ratio = float((g_arr < 50).mean())
    bright_ratio = float((g_arr > 200).mean())
    mid_ratio = float(((g_arr >= 50) & (g_arr <= 200)).mean())

    # Detectar dibujos tipo manga/cómic
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
# 3. Chequeo SUAVE de "parece panorámica"
# ─────────────────────────────────────────────────────────────

def soft_check_panoramic(img: Image.Image) -> Tuple[bool, str, float]:
    """Chequeo suave de formato panorámico (solo informativo)."""
    
    width, height = img.size
    aspect_ratio = width / float(height)

    if 1.4 <= aspect_ratio <= 4.0:
        return True, "", aspect_ratio

    return False, f"Relación ancho/alto {aspect_ratio:.2f}:1 (podría no ser panorámica).", aspect_ratio


# ─────────────────────────────────────────────────────────────
# 4. Función principal
# ─────────────────────────────────────────────────────────────

def validate_dental_xray(file_bytes: bytes, filename: str) -> Tuple[bool, str, Dict]:
    """Función principal de validación.""" 

    details: Dict[str, object] = {
        "is_valid_image": False,
        "is_xray": False,
        "is_panoramic": False,
        "xray_confidence": 0.0,
        "panoramic_confidence": 0.0,
    }

    is_img, msg, pil_img = validate_image_file(file_bytes, filename)
    if not is_img or pil_img is None:
        return False, msg, details

    details["is_valid_image"] = True

    is_xray, xray_msg, xray_conf = validate_is_xray(pil_img)
    details["xray_confidence"] = float(xray_conf)

    if not is_xray:
        return False, xray_msg, details

    details["is_xray"] = True

    is_pano_like, pano_msg, pano_score = soft_check_panoramic(pil_img)
    details["is_panoramic"] = is_pano_like
    details["panoramic_confidence"] = float(pano_score)

    if is_pano_like:
        return True, "✅ Radiografía dental válida (formato compatible con panorámica).", details
    else:
        return True, (
            "✅ Radiografía dental válida. "
            "Nota: por su formato podría no ser panorámica (periapical/bitewing u otro tipo)."
        ), details
