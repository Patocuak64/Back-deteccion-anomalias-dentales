# app/inference.py
from typing import Tuple, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

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


def calculate_fdi(x_center_norm: float, y_center_norm: float) -> int:
    """
    Calcula el número FDI del diente según su posición
    """
    if y_center_norm < 0.5:
        if x_center_norm < 0.5:
            quadrant = 1
        else:
            quadrant = 2
    else:
        if x_center_norm < 0.5:
            quadrant = 4
        else:
            quadrant = 3
    
    relative_x = (x_center_norm % 0.5) / 0.5
    tooth_position = min(int(relative_x * 8) + 1, 8)
    fdi_number = quadrant * 10 + tooth_position
    
    return fdi_number


def run_inference(image: Image.Image, confidence: float) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Ejecuta inferencia con medición de tiempos
    """
    total_start = time.time()
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. Obtener modelo (cacheado)
    # ═══════════════════════════════════════════════════════════════════
    model_start = time.time()
    model = get_model()
    model_time = (time.time() - model_start) * 1000
    print(f"[INFERENCE] Modelo obtenido en {model_time:.0f}ms")
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. Ejecutar predicción
    # ═══════════════════════════════════════════════════════════════════
    predict_start = time.time()
    results = model.predict(source=image, conf=confidence, verbose=False)
    predict_time = (time.time() - predict_start) * 1000
    print(f"[INFERENCE] Predicción en {predict_time:.0f}ms")
    
    result = results[0]
    boxes = result.boxes
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. Dibujar resultados
    # ═══════════════════════════════════════════════════════════════════
    draw_start = time.time()
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    font, font_small = _font_pair()
    
    img_width, img_height = image.size
    
    class_counts = {0: 0, 1: 0, 2: 0}
    class_conf = {0: [], 1: [], 2: []}
    detections: List[Dict[str, Any]] = []
    
    teeth_fdi_map = {
        "Caries": [],
        "Diente_Retenido": [],
        "Perdida_Osea": []
    }
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. Sin detecciones
    # ═══════════════════════════════════════════════════════════════════
    if len(boxes) == 0:
        total_time = (time.time() - total_start) * 1000
        print(f"[INFERENCE] ✓ Sin detecciones - Total: {total_time:.0f}ms")
        
        return img_draw, {
            "summary": {"total": 0, "per_class": {}},
            "detections": [],
            "stats": {},
            "report_text": "No se detectaron problemas dentales en esta imagen.",
            "teeth_fdi": teeth_fdi_map,
            "performance": {
                "total_ms": round(total_time, 2),
                "model_load_ms": round(model_time, 2),
                "prediction_ms": round(predict_time, 2),
            }
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # 5. Procesar detecciones
    # ═══════════════════════════════════════════════════════════════════
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cid = int(box.cls[0].cpu().numpy())
        
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        
        fdi_number = calculate_fdi(x_center_norm, y_center_norm)
        
        class_counts[cid] += 1
        class_conf[cid].append(conf)
        cname = CLASS_NAMES.get(cid, f"cls_{cid}")
        color = CLASS_COLORS.get(cid, (255, 255, 0))
        
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        label = f"{cname} [{fdi_number}]: {conf:.1%}"
        bbox_text = draw.textbbox((x1, y1 - 25), label, font=font_small)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 25), label, fill="white", font=font_small)
        
        detections.append({
            "class_id": cid,
            "class_name": cname,
            "confidence": conf,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "fdi": fdi_number,
            "tooth_fdi": fdi_number,
        })
        
        if fdi_number not in teeth_fdi_map[cname]:
            teeth_fdi_map[cname].append(fdi_number)
    
    draw_time = (time.time() - draw_start) * 1000
    print(f"[INFERENCE] Dibujo en {draw_time:.0f}ms")
    
    # ═══════════════════════════════════════════════════════════════════
    # 6. Generar estadísticas
    # ═══════════════════════════════════════════════════════════════════
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
    
    # ═══════════════════════════════════════════════════════════════════
    # 7. Generar reporte
    # ═══════════════════════════════════════════════════════════════════
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
    
    # ═══════════════════════════════════════════════════════════════════
    # 8. Resultado final
    # ═══════════════════════════════════════════════════════════════════
    total_time = (time.time() - total_start) * 1000
    print(f"[INFERENCE] ✓ Análisis completo - Total: {total_time:.0f}ms ({len(boxes)} detecciones)")
    
    payload = {
        "summary": {"total": total, "per_class": per_class},
        "detections": detections,
        "stats": stats,
        "report_text": "\n".join(report_lines),
        "teeth_fdi": teeth_fdi_map,
        "performance": {
            "total_ms": round(total_time, 2),
            "model_load_ms": round(model_time, 2),
            "prediction_ms": round(predict_time, 2),
            "draw_ms": round(draw_time, 2),
        }
    }
    
    return img_draw, payload