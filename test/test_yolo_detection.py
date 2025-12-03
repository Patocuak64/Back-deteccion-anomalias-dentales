# test/test_yolo_detection.py
"""
Tests para detección YOLO.

Para ejecutar:
    pytest test/test_yolo_detection.py -v -s
"""

import pytest
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import time

# Agregar la carpeta raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulo de inferencia
try:
    from app.inference import run_inference, calculate_fdi
    YOLO_DISPONIBLE = True
except ImportError as e:
    print(f"❌ Error importando inference: {e}")
    YOLO_DISPONIBLE = False

# Importar configuración del dataset
try:
    from test.test_config import (
        dataset_exists,
        get_random_dataset_image,
        get_random_dataset_images,
        load_random_image_as_pil,
        get_dataset_info
    )
    DATASET_DISPONIBLE = dataset_exists()
except ImportError:
    DATASET_DISPONIBLE = False


# Decorators
skip_if_no_yolo = pytest.mark.skipif(not YOLO_DISPONIBLE, reason="YOLO no disponible")
skip_if_no_dataset = pytest.mark.skipif(not DATASET_DISPONIBLE, reason="Dataset no disponible")


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def imagen_sintetica():
    """Imagen sintética de prueba"""
    arr = np.random.randint(60, 180, (600, 1200), dtype=np.uint8)
    arr = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(arr, mode='RGB')


@pytest.fixture
def imagen_dataset():
    """Imagen real del dataset"""
    if not DATASET_DISPONIBLE:
        pytest.skip("Dataset no disponible")
    
    result = load_random_image_as_pil()
    if not result:
        pytest.skip("No se pudo cargar imagen")
    
    img, filename = result
    print(f"\n🖼️  Dataset: {filename}")
    return img


# ═══════════════════════════════════════════════════════════════════
# TESTS DE INFERENCIA
# ═══════════════════════════════════════════════════════════════════

class TestYOLOInference:
    """Tests de inferencia YOLO"""

    @skip_if_no_yolo
    @pytest.mark.slow
    def test_inference_imagen_sintetica(self, imagen_sintetica):
        """Inferencia funciona con imagen sintética"""
        img_anotada, resultados = run_inference(imagen_sintetica, confidence=0.25)
        
        assert isinstance(img_anotada, Image.Image)
        assert isinstance(resultados, dict)
        print(f"\n✅ Inferencia OK")

    @skip_if_no_yolo
    @skip_if_no_dataset
    @pytest.mark.slow
    def test_inference_imagen_real(self, imagen_dataset):
        """Inferencia con imagen real del dataset"""
        inicio = time.time()
        img_anotada, resultados = run_inference(imagen_dataset, confidence=0.25)
        tiempo = time.time() - inicio
        
        assert isinstance(img_anotada, Image.Image)
        assert isinstance(resultados, dict)
        
        total = resultados["summary"]["total"]
        print(f"\n📊 Detecciones: {total}")
        print(f"⏱️  Tiempo: {tiempo:.2f}s")

    @skip_if_no_yolo
    @pytest.mark.slow
    def test_estructura_resultado(self, imagen_sintetica):
        """Resultado tiene estructura correcta"""
        _, resultados = run_inference(imagen_sintetica, confidence=0.25)
        
        # Campos requeridos
        assert "summary" in resultados
        assert "detections" in resultados
        assert "stats" in resultados
        assert "report_text" in resultados
        assert "teeth_fdi" in resultados
        assert "performance" in resultados
        
        # Estructura de summary
        assert "total" in resultados["summary"]
        assert "per_class" in resultados["summary"]
        
        print(f"\n✅ Estructura correcta")

    @skip_if_no_yolo
    @skip_if_no_dataset
    @pytest.mark.slow
    def test_multiples_imagenes_dataset(self):
        """Test con múltiples imágenes del dataset"""
        imagenes = get_random_dataset_images(5)
        if len(imagenes) < 3:
            pytest.skip("Pocas imágenes")
        
        total_detecciones = 0
        
        print("\n" + "=" * 50)
        for img_path in imagenes:
            img = Image.open(img_path)
            _, resultados = run_inference(img, confidence=0.25)
            
            det = resultados["summary"]["total"]
            total_detecciones += det
            
            status = "✅" if det > 0 else "⚪"
            print(f"{status} {img_path.name}: {det} detecciones")
        
        print(f"\nTotal: {total_detecciones} detecciones")


# ═══════════════════════════════════════════════════════════════════
# TESTS DE FDI
# ═══════════════════════════════════════════════════════════════════

class TestCalculateFDI:
    """Tests de cálculo FDI"""

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_fdi_cuadrante_1(self):
        """Cuadrante 1: FDI 11-18"""
        fdi = calculate_fdi(0.25, 0.25)
        assert 11 <= fdi <= 18
        print(f"\n✅ Q1: {fdi}")

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_fdi_cuadrante_2(self):
        """Cuadrante 2: FDI 21-28"""
        fdi = calculate_fdi(0.75, 0.25)
        assert 21 <= fdi <= 28
        print(f"\n✅ Q2: {fdi}")

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_fdi_cuadrante_3(self):
        """Cuadrante 3: FDI 31-38"""
        fdi = calculate_fdi(0.75, 0.75)
        assert 31 <= fdi <= 38
        print(f"\n✅ Q3: {fdi}")

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_fdi_cuadrante_4(self):
        """Cuadrante 4: FDI 41-48"""
        fdi = calculate_fdi(0.25, 0.75)
        assert 41 <= fdi <= 48
        print(f"\n✅ Q4: {fdi}")

    @skip_if_no_yolo
    @pytest.mark.unit
    @pytest.mark.parametrize("x,y,q_esperado", [
        (0.1, 0.1, 1),
        (0.9, 0.1, 2),
        (0.9, 0.9, 3),
        (0.1, 0.9, 4),
    ])
    def test_fdi_todos_cuadrantes(self, x, y, q_esperado):
        """FDI correcto para cada cuadrante"""
        fdi = calculate_fdi(x, y)
        q_obtenido = fdi // 10
        
        assert q_obtenido == q_esperado
        assert 11 <= fdi <= 48


# ═══════════════════════════════════════════════════════════════════
# TESTS DEL MODELO
# ═══════════════════════════════════════════════════════════════════

class TestYOLOModelo:
    """Tests del modelo YOLO"""

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_modelo_carga(self):
        """Modelo puede cargarse"""
        from app.model_store import get_model
        modelo = get_model()
        assert modelo is not None
        print(f"\n✅ Modelo cargado")

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_clases_correctas(self):
        """Modelo tiene 3 clases"""
        from app.inference import CLASS_NAMES, CLASS_COLORS
        
        assert len(CLASS_NAMES) == 3
        assert len(CLASS_COLORS) == 3
        
        print(f"\n✅ Clases:")
        for id, nombre in CLASS_NAMES.items():
            print(f"   {id}: {nombre}")


# ═══════════════════════════════════════════════════════════════════
# TEST DE DISPONIBILIDAD
# ═══════════════════════════════════════════════════════════════════

class TestYOLODisponibilidad:
    """Test de disponibilidad"""

    @pytest.mark.unit
    def test_modulo_disponible(self):
        """Módulo inference disponible"""
        assert YOLO_DISPONIBLE == True
        print(f"\n✅ Módulo disponible")