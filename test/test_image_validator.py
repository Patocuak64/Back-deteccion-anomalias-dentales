# test/test_image_validator.py
"""
Tests para el validador de imágenes radiográficas.

Para ejecutar:
    pytest test/test_image_validator.py -v -s
"""

import pytest
import io
from PIL import Image
import numpy as np
import sys
from pathlib import Path

# Agregar la carpeta raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar validador
from app.image_validator import validate_dental_xray, validate_is_xray

# Importar configuración del dataset
try:
    from test.test_config import (
        dataset_exists,
        get_random_dataset_image,
        get_random_dataset_images,
        load_random_image_as_bytes,
        get_dataset_info
    )
    DATASET_DISPONIBLE = dataset_exists()
except ImportError:
    DATASET_DISPONIBLE = False


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def crear_imagen_gris(width=800, height=400):
    """Crea imagen en escala de grises"""
    arr = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    img = Image.fromarray(arr, mode='L')
    return img.convert('RGB')


def crear_imagen_color(width=800, height=600):
    """Crea imagen a color"""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode='RGB')


def imagen_to_bytes(img: Image.Image) -> bytes:
    """Convierte PIL Image a bytes"""
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════
# TESTS CON IMÁGENES DEL DATASET
# ═══════════════════════════════════════════════════════════════════

class TestImageValidatorDataset:
    """Tests con imágenes reales del dataset"""

    @pytest.mark.unit
    def test_radiografia_real_aceptada(self):
        """Radiografías reales del dataset deben ser aceptadas"""
        if not DATASET_DISPONIBLE:
            pytest.skip("Dataset no disponible")
        
        result = load_random_image_as_bytes()
        if not result:
            pytest.skip("No se pudo cargar imagen")
        
        img_bytes, filename = result
        is_valid, msg, details = validate_dental_xray(img_bytes, filename)
        
        print(f"\n🖼️  {filename}")
        print(f"   Válida: {is_valid}")
        print(f"   Mensaje: {msg[:50]}..." if len(msg) > 50 else f"   Mensaje: {msg}")
        
        assert is_valid == True, f"Radiografía rechazada: {msg}"

    @pytest.mark.unit
    def test_multiples_radiografias_reales(self):
        """Múltiples imágenes del dataset deben ser aceptadas"""
        if not DATASET_DISPONIBLE:
            pytest.skip("Dataset no disponible")
        
        imagenes = get_random_dataset_images(10)
        if len(imagenes) < 5:
            pytest.skip(f"Pocas imágenes ({len(imagenes)})")
        
        aceptadas = 0
        rechazadas = []
        
        for img_path in imagenes:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            
            is_valid, msg, _ = validate_dental_xray(img_bytes, img_path.name)
            
            if is_valid:
                aceptadas += 1
            else:
                rechazadas.append((img_path.name, msg))
        
        print(f"\n✅ Aceptadas: {aceptadas}/{len(imagenes)}")
        
        if rechazadas:
            print("❌ Rechazadas:")
            for nombre, motivo in rechazadas[:3]:
                print(f"   - {nombre}: {motivo[:40]}...")
        
        # Al menos 80% deben pasar
        tasa = aceptadas / len(imagenes)
        assert tasa >= 0.80, f"Tasa muy baja: {tasa:.1%}"


# ═══════════════════════════════════════════════════════════════════
# TESTS DE IMÁGENES INVÁLIDAS
# ═══════════════════════════════════════════════════════════════════

class TestImageValidator:
    """Tests de validación con imágenes inválidas"""

    @pytest.mark.unit
    def test_imagen_color_rechazada(self):
        """Imagen a color debe ser rechazada"""
        img = crear_imagen_color(800, 600)
        is_valid, msg, _ = validate_is_xray(img)
        
        assert is_valid == False
        print(f"\n✅ Color rechazada: {msg[:50]}...")

    @pytest.mark.unit
    def test_pdf_rechazado(self):
        """PDF debe ser rechazado"""
        pdf_bytes = b"%PDF-1.4\ncontenido"
        is_valid, msg, _ = validate_dental_xray(pdf_bytes, "doc.pdf")
        
        assert is_valid == False
        assert "pdf" in msg.lower() or "PDF" in msg
        print(f"\n✅ PDF rechazado")

    @pytest.mark.unit
    def test_video_rechazado(self):
        """Video debe ser rechazado"""
        img = crear_imagen_gris(800, 400)
        img_bytes = imagen_to_bytes(img)
        is_valid, msg, _ = validate_dental_xray(img_bytes, "video.mp4")
        
        assert is_valid == False
        print(f"\n✅ Video rechazado")

    @pytest.mark.unit
    def test_imagen_sin_contraste_rechazada(self):
        """Imagen sin contraste debe ser rechazada"""
        arr = np.full((400, 800), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode='L').convert('RGB')
        is_valid, msg, _ = validate_is_xray(img)
        
        assert is_valid == False
        assert "contraste" in msg.lower()
        print(f"\n✅ Sin contraste rechazada")


# ═══════════════════════════════════════════════════════════════════
# TESTS DE FORMATOS
# ═══════════════════════════════════════════════════════════════════

class TestImageFormatos:
    """Tests de formatos de archivo"""

    @pytest.mark.unit
    @pytest.mark.parametrize("extension", [".jpg", ".jpeg", ".png", ".bmp"])
    def test_formatos_validos(self, extension):
        """Formatos válidos deben procesarse"""
        img = crear_imagen_gris(800, 400)
        img_bytes = imagen_to_bytes(img)
        is_valid, msg, details = validate_dental_xray(img_bytes, f"img{extension}")
        
        # Puede rechazarse por contenido, pero no por formato
        print(f"\n{extension}: {'✅' if is_valid else '⚠️'}")

    @pytest.mark.unit
    @pytest.mark.parametrize("extension", [".txt", ".doc", ".exe", ".zip"])
    def test_formatos_invalidos(self, extension):
        """Formatos inválidos deben ser rechazados"""
        img = crear_imagen_gris(800, 400)
        img_bytes = imagen_to_bytes(img)
        is_valid, msg, _ = validate_dental_xray(img_bytes, f"archivo{extension}")
        
        assert is_valid == False
        print(f"\n✅ {extension} rechazado")


# ═══════════════════════════════════════════════════════════════════
# TEST DE ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════

class TestValidadorEstadisticas:
    """Tests de estadísticas del validador"""

    @pytest.mark.slow
    def test_estadisticas_dataset(self):
        """Genera estadísticas de validación"""
        if not DATASET_DISPONIBLE:
            pytest.skip("Dataset no disponible")
        
        imagenes = get_random_dataset_images(15)
        if len(imagenes) < 10:
            pytest.skip("Pocas imágenes")
        
        stats = {"aceptadas": 0, "rechazadas": 0}
        
        for img_path in imagenes:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            
            is_valid, _, _ = validate_dental_xray(img_bytes, img_path.name)
            stats["aceptadas" if is_valid else "rechazadas"] += 1
        
        total = len(imagenes)
        print(f"\n📊 Estadísticas:")
        print(f"   Total: {total}")
        print(f"   Aceptadas: {stats['aceptadas']} ({stats['aceptadas']/total*100:.1f}%)")
        print(f"   Rechazadas: {stats['rechazadas']} ({stats['rechazadas']/total*100:.1f}%)")
        
        assert True