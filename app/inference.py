# app/inference.py

from typing import Tuple, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .model_store import get_model
from .settings import settings


CLASS_NAMES = {
    0: "Caries",
    1: "Diente_Retenido",
    2: "Perdida_Osea",
}

CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}


def _font_pair():
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    return font, font_small


# ═══════════════════════════════════════════════════════════════════════════
# ⭐ NUEVA FUNCIÓN: CALCULAR NÚMERO FDI
# ═══════════════════════════════════════════════════════════════════════════
def calculate_fdi(x_center_norm: float, y_center_norm: float) -> int:
    """
    Calcula el número FDI del diente según su posición en la radiografía.
    
    Sistema FDI (numeración dental internacional):
    - Cuadrante 1 (Superior Derecho): 11-18
    - Cuadrante 2 (Superior Izquierdo): 21-28
    - Cuadrante 3 (Inferior Izquierdo): 31-38
    - Cuadrante 4 (Inferior Derecho): 41-48
    
    IMPORTANTE: En radiografías panorámicas, la imagen está ESPEJADA.
    Lo que vemos a la izquierda es el lado DERECHO del paciente.
    
    Args:
        x_center_norm: Coordenada X del centro del bbox normalizada (0-1)
        y_center_norm: Coordenada Y del centro del bbox normalizada (0-1)
    
    Returns:
        Número FDI (11-48)
    """
    
    # Determinar cuadrante basado en posición
    if y_center_norm < 0.5:  # Parte SUPERIOR de la imagen
        if x_center_norm < 0.5:  # Izquierda imagen = Derecha paciente
            quadrant = 1  # Superior Derecho
        else:  # Derecha imagen = Izquierda paciente
            quadrant = 2  # Superior Izquierdo
    else:  # Parte INFERIOR de la imagen
        if x_center_norm < 0.5:  # Izquierda imagen = Derecha paciente
            quadrant = 4  # Inferior Derecho
        else:  # Derecha imagen = Izquierda paciente
            quadrant = 3  # Inferior Izquierdo
    
    # Calcular posición dentro del cuadrante (1-8)
    # Normalizar x dentro del cuadrante actual
    relative_x = (x_center_norm % 0.5) / 0.5  # 0 a 1
    
    # Dividir en 8 posiciones (dientes)
    # 1 = incisivo central, 8 = tercer molar
    tooth_position = min(int(relative_x * 8) + 1, 8)
    
    # Número FDI = Cuadrante * 10 + Posición
    fdi_number = quadrant * 10 + tooth_position
    
    return fdi_number


# ═══════════════════════════════════════════════════════════════════════════
# ⭐ FUNCIÓN PRINCIPAL MODIFICADA: run_inference
# ═══════════════════════════════════════════════════════════════════════════
def run_inference(image: Image.Image, confidence: float) -> Tuple[Image.Image, Dict[str, Any]]:
    model = get_model()
    results = model.predict(source=image, conf=confidence, verbose=False)
    result = results[0]
    boxes = result.boxes

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    font, font_small = _font_pair()

    # ⭐ NUEVO: obtener dimensiones de imagen para normalizar
    img_width, img_height = image.size

    class_counts = {0: 0, 1: 0, 2: 0}
    class_conf = {0: [], 1: [], 2: []}
    detections: List[Dict[str, Any]] = []
    
    # ⭐ NUEVO: mapa de dientes FDI por clase
    teeth_fdi_map = {
        "Caries": [],
        "Diente_Retenido": [],
        "Perdida_Osea": []
    }

    if len(boxes) == 0:
        return img_draw, {
            "summary": {"total": 0, "per_class": {}},
            "detections": [],
            "stats": {},
            "report_text": "No se detectaron problemas dentales en esta imagen.",
            "teeth_fdi": teeth_fdi_map,  # ⭐ NUEVO
        }

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cid = int(box.cls[0].cpu().numpy())

        # ⭐ NUEVO: calcular centro del bbox y normalizar
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        
        # ⭐ NUEVO: calcular número FDI
        fdi_number = calculate_fdi(x_center_norm, y_center_norm)

        class_counts[cid] += 1
        class_conf[cid].append(conf)
        cname = CLASS_NAMES.get(cid, f"cls_{cid}")
        color = CLASS_COLORS.get(cid, (255, 255, 0))

        # ⭐ MODIFICADO: incluir FDI en el label
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        label = f"{cname} [{fdi_number}]: {conf:.1%}"
        bbox_text = draw.textbbox((x1, y1 - 25), label, font=font_small)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 25), label, fill="white", font=font_small)

        # ⭐ MODIFICADO: agregar FDI a detections
        detections.append(
            {
                "class_id": cid,
                "class_name": cname,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "fdi": fdi_number,  # ⭐ NUEVO
                "tooth_fdi": fdi_number,  # ⭐ NUEVO (alias para compatibilidad)
            }
        )
        
        # ⭐ NUEVO: agregar a mapa de dientes
        if fdi_number not in teeth_fdi_map[cname]:
            teeth_fdi_map[cname].append(fdi_number)

    # stats
    per_class = {}
    stats = {}
    for cid, count in class_counts.items():
        if count > 0:
            cname = CLASS_NAMES[cid]
            arr = class_conf[cid]
            per_class[cname] = count
            stats[cname] = {
                "count": count,
                "conf_avg": float(np.mean(arr)),
                "conf_min": float(np.min(arr)),
                "conf_max": float(np.max(arr)),
            }

    # ⭐ MODIFICADO: reporte mejorado con FDI
    report_lines = ["ANÁLISIS DE RADIOGRAFÍA DENTAL", "=" * 50, ""]
    total = sum(class_counts.values())
    report_lines.append(f"Total de detecciones: {total}\n")
    
    for cid, count in class_counts.items():
        if count > 0:
            cname = CLASS_NAMES[cid]
            fdi_list = teeth_fdi_map[cname]
            fdi_str = ", ".join(map(str, sorted(fdi_list)))
            report_lines.append(
                f"{cname}: {count} (conf prom {np.mean(class_conf[cid]):.1%})"
            )
            report_lines.append(f"  └─ Dientes: {fdi_str}")
    
    report_lines += ["", "INTERPRETACIÓN:", "-" * 50]
    
    if class_counts[0] > 0:
        fdi_str = ", ".join(map(str, sorted(teeth_fdi_map["Caries"])))
        report_lines.append(f"⚠️ Caries detectadas en dientes: {fdi_str}")
    
    if class_counts[1] > 0:
        fdi_str = ", ".join(map(str, sorted(teeth_fdi_map["Diente_Retenido"])))
        report_lines.append(f"⚠️ Dientes retenidos: {fdi_str}")
    
    if class_counts[2] > 0:
        fdi_str = ", ".join(map(str, sorted(teeth_fdi_map["Perdida_Osea"])))
        report_lines.append(f"⚠️ Pérdida ósea en dientes: {fdi_str}")
    
    if total == 0:
        report_lines.append("✅ Sin hallazgos significativos")
    
    report_lines += [
        "",
        "NOTA: Herramienta de apoyo. No reemplaza diagnóstico profesional.",
    ]

    payload = {
        "summary": {"total": total, "per_class": per_class},
        "detections": detections,
        "stats": stats,
        "report_text": "\n".join(report_lines),
        "teeth_fdi": teeth_fdi_map,  # ⭐ NUEVO
    }
    return img_draw, payload