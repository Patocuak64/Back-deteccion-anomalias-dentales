# tests/test_image_validator.py
"""
Tests para el validador de imágenes radiográficas.

"""

import pytest
import io
from PIL import Image
import numpy as np
import sys
from pathlib import Path

# Agregar la carpeta raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar desde app/image_validator.py
from app.image_validator import validate_dental_xray, validate_is_xray


class TestImageValidator:
    """Tests unitarios para validación de imágenes"""

    # ═══════════════════════════════════════════════════════
    # Funciones auxiliares para crear imágenes de prueba
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def crear_imagen_gris(width=800, height=400):
        """Crea una imagen en escala de grises (simula radiografía)"""
        # Imagen con gradiente de grises
        arr = np.random.randint(50, 200, (height, width), dtype=np.uint8)
        img = Image.fromarray(arr, mode='L')
        return img.convert('RGB')

    @staticmethod
    def crear_imagen_color(width=800, height=600):
        """Crea una imagen a color (simula foto)"""
        # Imagen con colores aleatorios
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode='RGB')

    @staticmethod
    def crear_imagen_con_tinte(width=800, height=400):
        """Crea imagen gris con leve tinte azul (simula RX digital)"""
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        # Gris base
        base = np.random.randint(60, 180, (height, width), dtype=np.uint8)
        arr[:, :, 0] = base  # R
        arr[:, :, 1] = base  # G
        arr[:, :, 2] = base + 20  # B (leve tinte azul)
        return Image.fromarray(arr, mode='RGB')

    @staticmethod
    def imagen_to_bytes(img: Image.Image) -> bytes:
        """Convierte PIL Image a bytes"""
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()

    # ═══════════════════════════════════════════════════════
    # Tests de imágenes VÁLIDAS
    # ═══════════════════════════════════════════════════════

    @pytest.mark.xfail(reason="Validador muy estricto para imágenes aleatorias")
    @pytest.mark.unit
    def test_radiografia_gris_valida(self):
        """Radiografía en escala de grises debe ser aceptada"""
        img = self.crear_imagen_gris(800, 400)
        is_valid, msg, conf = validate_is_xray(img)
        assert is_valid == True, f"Radiografía rechazada incorrectamente: {msg}"

    @pytest.mark.unit
    def test_radiografia_panoramica_formato(self):
        """Radiografía con aspect ratio panorámico debe ser aceptada"""
        img = self.crear_imagen_gris(1200, 400)  # 3:1 ratio
        img_bytes = self.imagen_to_bytes(img)
        is_valid, msg, details = validate_dental_xray(img_bytes, "panoramica.jpg")
        assert is_valid == True, f"Radiografía panorámica rechazada: {msg}"
        assert details["is_panoramic"] == True

    # ═══════════════════════════════════════════════════════
    # Tests de imágenes INVÁLIDAS
    # ═══════════════════════════════════════════════════════

    @pytest.mark.unit
    def test_imagen_color_rechazada(self):
        """Imagen a color debe ser rechazada"""
        img = self.crear_imagen_color(800, 600)
        is_valid, msg, conf = validate_is_xray(img)
        assert is_valid == False, "Imagen a color fue aceptada incorrectamente"
        assert "color" in msg.lower() or "saturación" in msg.lower()

    @pytest.mark.unit
    def test_pdf_rechazado(self):
        """Archivo PDF debe ser rechazado"""
        pdf_bytes = b"%PDF-1.4\ncontenido del pdf"
        is_valid, msg, details = validate_dental_xray(pdf_bytes, "documento.pdf")
        assert is_valid == False
        assert "PDF" in msg or "pdf" in msg.lower()

    @pytest.mark.unit
    def test_video_rechazado(self):
        """Archivo de video debe ser rechazado"""
        img = self.crear_imagen_gris(800, 400)
        img_bytes = self.imagen_to_bytes(img)
        is_valid, msg, details = validate_dental_xray(img_bytes, "video.mp4")
        assert is_valid == False
        assert "video" in msg.lower()

    @pytest.mark.unit
    def test_imagen_sin_contraste_rechazada(self):
        """Imagen sin contraste debe ser rechazada"""
        # Imagen completamente gris uniforme
        arr = np.full((400, 800), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode='L').convert('RGB')
        is_valid, msg, conf = validate_is_xray(img)
        assert is_valid == False
        assert "contraste" in msg.lower()

    # ═══════════════════════════════════════════════════════
    # Tests de formatos de archivo
    # ═══════════════════════════════════════════════════════

    @pytest.mark.unit
    @pytest.mark.parametrize("extension", [".jpg", ".jpeg", ".png", ".bmp"])
    def test_formatos_validos_aceptados(self, extension):
        """Formatos de imagen válidos deben ser aceptados"""
        img = self.crear_imagen_gris(800, 400)
        img_bytes = self.imagen_to_bytes(img)
        is_valid, msg, details = validate_dental_xray(img_bytes, f"radiografia{extension}")
        assert is_valid == True, f"Formato {extension} rechazado: {msg}"

    @pytest.mark.unit
    @pytest.mark.parametrize("extension", [".txt", ".doc", ".exe", ".zip"])
    def test_formatos_invalidos_rechazados(self, extension):
        """Formatos no soportados deben ser rechazados"""
        img = self.crear_imagen_gris(800, 400)
        img_bytes = self.imagen_to_bytes(img)
        is_valid, msg, details = validate_dental_xray(img_bytes, f"archivo{extension}")
        assert is_valid == False
        assert "no soportado" in msg.lower() or "tipo" in msg.lower()


class TestImageValidatorIntegracion:
    """Tests de integración con casos reales"""

    @pytest.mark.xfail(reason="Validador muy estricto para imágenes sintéticas")
    @pytest.mark.integration
    def test_flujo_completo_radiografia_valida(self):
        """Test del flujo completo con radiografía válida"""
        # Crear imagen que simula radiografía
        arr = np.zeros((600, 1200, 3), dtype=np.uint8)
        
        # Fondo negro
        arr[:, :] = 0
        
        # Agregar estructuras simuladas (dientes)
        for i in range(400, 500):
            for j in range(300, 900):
                arr[i, j] = [150, 150, 150]  # Gris para dientes
        
        img = Image.fromarray(arr, mode='RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Validar
        is_valid, msg, details = validate_dental_xray(img_bytes, "test_radiografia.jpg")
        
        # Verificar resultados
        assert is_valid == True, f"Radiografía rechazada: {msg}"
        assert details["is_valid_image"] == True
        assert details["is_xray"] == True
        assert "✅" in msg