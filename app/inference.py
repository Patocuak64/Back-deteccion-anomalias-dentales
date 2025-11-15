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


def run_inference(image: Image.Image, confidence: float) -> Tuple[Image.Image, Dict[str, Any]]:
    model = get_model()
    results = model.predict(source=image, conf=confidence, verbose=False)
    result = results[0]
    boxes = result.boxes

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    font, font_small = _font_pair()

    class_counts = {0: 0, 1: 0, 2: 0}
    class_conf = {0: [], 1: [], 2: []}
    detections: List[Dict[str, Any]] = []

    if len(boxes) == 0:
        return img_draw, {
            "summary": {"total": 0, "per_class": {}},
            "detections": [],
            "stats": {},
            "report_text": "No se detectaron problemas dentales en esta imagen.",
        }

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cid = int(box.cls[0].cpu().numpy())

        class_counts[cid] += 1
        class_conf[cid].append(conf)
        cname = CLASS_NAMES.get(cid, f"cls_{cid}")
        color = CLASS_COLORS.get(cid, (255, 255, 0))

        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        label = f"{cname}: {conf:.1%}"
        bbox = draw.textbbox((x1, y1 - 25), label, font=font_small)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 25), label, fill="white", font=font_small)

        detections.append(
            {
                "class_id": cid,
                "class_name": cname,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }
        )

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

    report_lines = ["ANÁLISIS DE RADIOGRAFÍA DENTAL", "=" * 50, ""]
    total = sum(class_counts.values())
    report_lines.append(f"Total de detecciones: {total}\n")
    for cid, count in class_counts.items():
        if count > 0:
            cname = CLASS_NAMES[cid]
            report_lines.append(
                f"{cname}: {count} (conf prom {np.mean(class_conf[cid]):.1%})"
            )
    report_lines += ["", "INTERPRETACIÓN:", "-" * 50]
    if class_counts[0] > 0:
        report_lines.append(f"⚠️ Caries: {class_counts[0]}")
    if class_counts[1] > 0:
        report_lines.append(f"⚠️ Diente retenido: {class_counts[1]}")
    if class_counts[2] > 0:
        report_lines.append(f"⚠️ Pérdida ósea: {class_counts[2]}")
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
    }
    return img_draw, payload
