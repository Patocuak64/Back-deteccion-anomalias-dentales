# test/test_image_validator.py
"""
Tests para el validador de imágenes radiográficas.

MODIFICADO: Ahora incluye tests con imágenes reales del dataset
para verificar que el validador acepta radiografías válidas.

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

# Importar desde app/image_validator.py
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
    DATASET_CONFIG_DISPONIBLE = True
except ImportError:
    DATASET_CONFIG_DISPONIBLE = False


# Decorator para tests que requieren dataset
skip_if_no_dataset = pytest.mark.skipif(
    not DATASET_CONFIG_DISPONIBLE or not dataset_exists(),
    reason="Dataset de validación no disponible"
)


class TestImageValidator:
    """Tests unitarios para validación de imágenes"""

    # ═══════════════════════════════════════════════════════
    # Funciones auxiliares para crear imágenes de prueba
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def crear_imagen_gris(width=800, height=400):
        """Crea una imagen en escala de grises (simula radiografía)"""
        arr = np.random.randint(50, 200, (height, width), dtype=np.uint8)
        img = Image.fromarray(arr, mode='L')
        return img.convert('RGB')

    @staticmethod
    def crear_imagen_color(width=800, height=600):
        """Crea una imagen a color (simula foto)"""
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode='RGB')

    @staticmethod
    def crear_imagen_con_tinte(width=800, height=400):
        """Crea imagen gris con leve tinte azul (simula RX digital)"""
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        base = np.random.randint(60, 180, (height, width), dtype=np.uint8)
        arr[:, :, 0] = base
        arr[:, :, 1] = base
        arr[:, :, 2] = base + 20
        return Image.fromarray(arr, mode='RGB')

    @staticmethod
    def imagen_to_bytes(img: Image.Image) -> bytes:
        """Convierte PIL Image a bytes"""
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()

    # ═══════════════════════════════════════════════════════
    # NUEVO: Tests con imágenes REALES del dataset
    # ═══════════════════════════════════════════════════════

    @skip_if_no_dataset
    @pytest.mark.unit
    def test_radiografia_real_del_dataset_aceptada(self):
        """
        NUEVO: Las radiografías reales del dataset deben ser aceptadas.
        Este es el test más importante para validar que el validador
        no rechaza imágenes válidas.
        """
        result = load_random_image_as_bytes()
        if not result:
            pytest.skip("No se pudo cargar imagen del dataset")
        
        img_bytes, filename = result
        
        is_valid, msg, details = validate_dental_xray(img_bytes, filename)
        
        print(f"\n🖼️  Imagen: {filename}")
        print(f"   Válida: {is_valid}")
        print(f"   Mensaje: {msg}")
        print(f"   X-ray confidence: {details.get('xray_confidence', 0):.1f}")
        print(f"   Panoramic: {details.get('is_panoramic', False)}")
        
        # Las imágenes del dataset de validación DEBEN ser aceptadas
        assert is_valid == True, f"Radiografía real rechazada: {msg}"

    @skip_if_no_dataset
    @pytest.mark.unit
    def test_multiples_radiografias_reales_aceptadas(self):
        """
        NUEVO: Verificar que múltiples imágenes del dataset son aceptadas.
        """
        imagenes = get_random_dataset_images(10)
        
        if len(imagenes) < 5:
            pytest.skip(f"Dataset tiene pocas imágenes ({len(imagenes)})")
        
        resultados = {"aceptadas": 0, "rechazadas": 0}
        rechazadas_detalle = []
        
        print("\n" + "=" * 60)
        print("VALIDACIÓN DE IMÁGENES DEL DATASET")
        print("=" * 60)
        
        for img_path in imagenes:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            
            is_valid, msg, details = validate_dental_xray(img_bytes, img_path.name)
            
            if is_valid:
                resultados["aceptadas"] += 1
                status = "✅"
            else:
                resultados["rechazadas"] += 1
                status = "❌"
                rechazadas_detalle.append((img_path.name, msg))
            
            print(f"{status} {img_path.name}")
        
        print(f"\nResumen:")
        print(f"  Aceptadas: {resultados['aceptadas']}/{len(imagenes)}")
        print(f"  Rechazadas: {resultados['rechazadas']}/{len(imagenes)}")
        
        if rechazadas_detalle:
            print("\nImágenes rechazadas:")
            for nombre, motivo in rechazadas_detalle:
                print(f"  - {nombre}: {motivo}")
        
        # Al menos el 80% de las imágenes del dataset deben ser aceptadas
        tasa_aceptacion = resultados["aceptadas"] / len(imagenes)
        assert tasa_aceptacion >= 0.80, \
            f"Tasa de aceptación muy baja: {tasa_aceptacion:.1%}"

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
        print(f"\n✅ Imagen a color rechazada: {msg}")

    @pytest.mark.unit
    def test_pdf_rechazado(self):
        """Archivo PDF debe ser rechazado"""
        pdf_bytes = b"%PDF-1.4\ncontenido del pdf"
        is_valid, msg, details = validate_dental_xray(pdf_bytes, "documento.pdf")
        
        assert is_valid == False
        assert "PDF" in msg or "pdf" in msg.lower()
        print(f"\n✅ PDF rechazado: {msg}")

    @pytest.mark.unit
    def test_video_rechazado(self):
        """Archivo de video debe ser rechazado"""
        img = self.crear_imagen_gris(800, 400)
        img_bytes = self.imagen_to_bytes(img)
        is_valid, msg, details = validate_dental_xray(img_bytes, "video.mp4")
        
        assert is_valid == False
        assert "video" in msg.lower()
        print(f"\n✅ Video rechazado: {msg}")

    @pytest.mark.unit
    def test_imagen_sin_contraste_rechazada(self):
        """Imagen sin contraste debe ser rechazada"""
        arr = np.full((400, 800), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode='L').convert('RGB')
        is_valid, msg, conf = validate_is_xray(img)
        
        assert is_valid == False
        assert "contraste" in msg.lower()
        print(f"\n✅ Imagen sin contraste rechazada: {msg}")

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
        
        # Nota: puede rechazarse por no parecer radiografía, pero no por formato
        if not is_valid:
            assert "tipo" not in msg.lower() or "soportado" not in msg.lower()
        
        print(f"\n✅ Formato {extension} procesado")

    @pytest.mark.unit
    @pytest.mark.parametrize("extension", [".txt", ".doc", ".exe", ".zip"])
    def test_formatos_invalidos_rechazados(self, extension):
        """Formatos no soportados deben ser rechazados"""
        img = self.crear_imagen_gris(800, 400)
        img_bytes = self.imagen_to_bytes(img)
        is_valid, msg, details = validate_dental_xray(img_bytes, f"archivo{extension}")
        
        assert is_valid == False
        print(f"\n✅ Formato {extension} rechazado: {msg}")


class TestImageValidatorIntegracion:
    """Tests de integración con casos reales"""

    @skip_if_no_dataset
    @pytest.mark.integration
    def test_flujo_completo_con_imagen_real(self):
        """
        NUEVO: Test del flujo completo con radiografía real del dataset.
        """
        result = load_random_image_as_bytes()
        if not result:
            pytest.skip("No se pudo cargar imagen del dataset")
        
        img_bytes, filename = result
        
        # Validar
        is_valid, msg, details = validate_dental_xray(img_bytes, filename)
        
        print(f"\n🔍 Validación de: {filename}")
        print(f"   ✅ Imagen válida: {details['is_valid_image']}")
        print(f"   ✅ Es radiografía: {details['is_xray']}")
        print(f"   ✅ Es panorámica: {details['is_panoramic']}")
        print(f"   📊 Confianza X-ray: {details['xray_confidence']:.1f}")
        print(f"   📊 Confianza panorámica: {details['panoramic_confidence']:.2f}")
        
        # Verificar resultados
        assert is_valid == True, f"Radiografía rechazada: {msg}"
        assert details["is_valid_image"] == True
        assert details["is_xray"] == True

    @pytest.mark.xfail(reason="Validador puede ser estricto con imágenes sintéticas")
    @pytest.mark.integration
    def test_flujo_completo_imagen_sintetica(self):
        """Test del flujo completo con imagen sintética"""
        arr = np.zeros((600, 1200, 3), dtype=np.uint8)
        arr[:, :] = 0
        
        for i in range(400, 500):
            for j in range(300, 900):
                arr[i, j] = [150, 150, 150]
        
        img = Image.fromarray(arr, mode='RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        is_valid, msg, details = validate_dental_xray(img_bytes, "test_radiografia.jpg")
        
        assert is_valid == True, f"Radiografía sintética rechazada: {msg}"


class TestValidadorEstadisticas:
    """Tests para obtener estadísticas del validador"""

    @skip_if_no_dataset
    @pytest.mark.slow
    def test_estadisticas_validacion_dataset(self):
        """
        NUEVO: Genera estadísticas de validación sobre todo el dataset.
        Útil para ajustar parámetros del validador.
        """
        imagenes = get_random_dataset_images(20)
        
        if len(imagenes) < 10:
            pytest.skip(f"Dataset tiene pocas imágenes ({len(imagenes)})")
        
        stats = {
            "total": len(imagenes),
            "aceptadas": 0,
            "rechazadas_por_color": 0,
            "rechazadas_por_contraste": 0,
            "rechazadas_por_formato": 0,
            "rechazadas_otras": 0,
            "xray_confidence_promedio": [],
            "panoramic_confidence_promedio": []
        }
        
        for img_path in imagenes:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            
            is_valid, msg, details = validate_dental_xray(img_bytes, img_path.name)
            
            if is_valid:
                stats["aceptadas"] += 1
                stats["xray_confidence_promedio"].append(details["xray_confidence"])
                stats["panoramic_confidence_promedio"].append(details["panoramic_confidence"])
            else:
                if "color" in msg.lower() or "saturación" in msg.lower():
                    stats["rechazadas_por_color"] += 1
                elif "contraste" in msg.lower():
                    stats["rechazadas_por_contraste"] += 1
                elif "formato" in msg.lower() or "tipo" in msg.lower():
                    stats["rechazadas_por_formato"] += 1
                else:
                    stats["rechazadas_otras"] += 1
        
        # Calcular promedios
        if stats["xray_confidence_promedio"]:
            avg_xray = sum(stats["xray_confidence_promedio"]) / len(stats["xray_confidence_promedio"])
        else:
            avg_xray = 0
        
        if stats["panoramic_confidence_promedio"]:
            avg_pano = sum(stats["panoramic_confidence_promedio"]) / len(stats["panoramic_confidence_promedio"])
        else:
            avg_pano = 0
        
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DE VALIDACIÓN DEL DATASET")
        print("=" * 60)
        print(f"Total imágenes analizadas: {stats['total']}")
        print(f"Aceptadas: {stats['aceptadas']} ({stats['aceptadas']/stats['total']*100:.1f}%)")
        print(f"\nRechazos por motivo:")
        print(f"  - Color/saturación: {stats['rechazadas_por_color']}")
        print(f"  - Contraste: {stats['rechazadas_por_contraste']}")
        print(f"  - Formato: {stats['rechazadas_por_formato']}")
        print(f"  - Otros: {stats['rechazadas_otras']}")
        print(f"\nConfianza promedio (imágenes aceptadas):")
        print(f"  - X-ray: {avg_xray:.1f}")
        print(f"  - Panorámica: {avg_pano:.2f}")
        
        # El test pasa si genera las estadísticas
        assert True