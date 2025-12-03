# test/test_config.py
"""
Configuración compartida para todos los tests de DentalSmart.

Este archivo centraliza:
- Rutas al dataset de validación
- Funciones para obtener imágenes aleatorias
- Configuraciones comunes de testing
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
import io


# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL DATASET
# ═══════════════════════════════════════════════════════════════════

# Ruta al dataset de validación (imágenes reales de radiografías)
# Puedes cambiar esta ruta o usar variable de entorno
DATASET_VALIDATION_PATH = os.getenv(
    "DENTAL_DATASET_PATH",
    r"C:\Users\jhonn\OneDrive\Desktop\dataset_dientes\el_candidato\YOLO\YOLO\dataset_3cls_clean\valid\images"
)

# Extensiones de imagen válidas
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


# ═══════════════════════════════════════════════════════════════════
# FUNCIONES PARA OBTENER IMÁGENES DEL DATASET
# ═══════════════════════════════════════════════════════════════════

def get_dataset_path() -> Path:
    """Retorna la ruta al dataset de validación."""
    return Path(DATASET_VALIDATION_PATH)


def dataset_exists() -> bool:
    """Verifica si el dataset existe y tiene imágenes."""
    path = get_dataset_path()
    if not path.exists():
        return False
    
    for ext in VALID_EXTENSIONS:
        if list(path.glob(f"*{ext}")):
            return True
        if list(path.glob(f"*{ext.upper()}")):
            return True
    
    return False


def list_dataset_images() -> List[Path]:
    """Lista todas las imágenes del dataset."""
    path = get_dataset_path()
    if not path.exists():
        return []
    
    images = []
    for ext in VALID_EXTENSIONS:
        images.extend(path.glob(f"*{ext}"))
        images.extend(path.glob(f"*{ext.upper()}"))
    
    return sorted(set(images))  # Eliminar duplicados


def get_random_dataset_image() -> Optional[Path]:
    """Obtiene una imagen aleatoria del dataset."""
    images = list_dataset_images()
    if not images:
        return None
    return random.choice(images)


def get_random_dataset_images(n: int = 5) -> List[Path]:
    """Obtiene N imágenes aleatorias del dataset."""
    images = list_dataset_images()
    if not images:
        return []
    
    if n >= len(images):
        result = list(images)
        random.shuffle(result)
        return result
    
    return random.sample(images, n)


def load_random_image_as_bytes() -> Optional[Tuple[bytes, str]]:
    """Carga una imagen aleatoria como bytes."""
    img_path = get_random_dataset_image()
    if not img_path:
        return None
    
    try:
        with open(img_path, 'rb') as f:
            return f.read(), img_path.name
    except Exception:
        return None


def load_random_image_as_pil() -> Optional[Tuple[Image.Image, str]]:
    """Carga una imagen aleatoria como PIL Image."""
    img_path = get_random_dataset_image()
    if not img_path:
        return None
    
    try:
        img = Image.open(img_path)
        return img, img_path.name
    except Exception:
        return None


def load_random_image_as_buffer() -> Optional[Tuple[io.BytesIO, str]]:
    """Carga una imagen aleatoria como BytesIO buffer."""
    result = load_random_image_as_bytes()
    if not result:
        return None
    
    img_bytes, filename = result
    buffer = io.BytesIO(img_bytes)
    buffer.seek(0)
    return buffer, filename


# ═══════════════════════════════════════════════════════════════════
# INFORMACIÓN DEL DATASET
# ═══════════════════════════════════════════════════════════════════

def get_dataset_info() -> dict:
    """Obtiene información sobre el dataset."""
    path = get_dataset_path()
    images = list_dataset_images()
    
    return {
        "path": str(path),
        "exists": path.exists(),
        "total_images": len(images),
        "sample_images": [img.name for img in images[:5]] if images else [],
        "extensions_found": list(set(img.suffix.lower() for img in images)) if images else []
    }


# ═══════════════════════════════════════════════════════════════════
# EJECUCIÓN DIRECTA
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURACIÓN DE DATASET PARA TESTS")
    print("=" * 60)
    
    info = get_dataset_info()
    
    print(f"\n📁 Ruta: {info['path']}")
    print(f"✅ Existe: {info['exists']}")
    print(f"📊 Imágenes: {info['total_images']}")
    
    if info['extensions_found']:
        print(f"📎 Extensiones: {', '.join(info['extensions_found'])}")
    
    if info['sample_images']:
        print(f"\n🖼️  Muestra:")
        for img in info['sample_images']:
            print(f"   - {img}")
    
    # Test
    print("\n" + "=" * 60)
    random_img = get_random_dataset_image()
    if random_img:
        print(f"✅ Imagen aleatoria: {random_img.name}")
    else:
        print("❌ No se pudo obtener imagen")




























































































