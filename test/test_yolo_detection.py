# tests/test_yolo_detection.py
"""
Tests para detección YOLO usando app/inference.py (estructura real).
"""

import pytest
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Agregar la carpeta raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar desde app/inference.py (TU estructura real)
try:
    from app.inference import run_inference, calculate_fdi
    YOLO_DISPONIBLE = True
    print("✅ Módulo inference importado correctamente")
except ImportError as e:
    print(f"❌ Error importando inference: {e}")
    YOLO_DISPONIBLE = False


# Decorator para saltar tests si no hay YOLO
skip_if_no_yolo = pytest.mark.skipif(
    not YOLO_DISPONIBLE,
    reason="Módulo YOLO no disponible"
)


class TestYOLOInference:
    """Tests para la función run_inference"""

    @staticmethod
    def crear_imagen_test():
        """Crea una imagen de prueba"""
        arr = np.random.randint(60, 180, (600, 1200), dtype=np.uint8)
        arr = np.stack([arr, arr, arr], axis=2)
        return Image.fromarray(arr, mode='RGB')

    # ═══════════════════════════════════════════════════════
    # Tests básicos de la función
    # ═══════════════════════════════════════════════════════

    @skip_if_no_yolo
    @pytest.mark.slow
    @pytest.mark.integration
    def test_run_inference_funciona(self):
        """La función run_inference debe ejecutarse sin errores"""
        img = self.crear_imagen_test()
        
        # Ejecutar inferencia
        img_anotada, resultados = run_inference(img, confidence=0.25)
        
        # Verificar que retorna los tipos correctos
        assert isinstance(img_anotada, Image.Image)
        assert isinstance(resultados, dict)

    @skip_if_no_yolo
    @pytest.mark.slow
    def test_resultado_tiene_estructura_correcta(self):
        """El resultado debe tener la estructura esperada"""
        img = self.crear_imagen_test()
        _, resultados = run_inference(img, confidence=0.25)
        
        # Verificar campos principales
        assert "summary" in resultados
        assert "detections" in resultados
        assert "stats" in resultados
        assert "report_text" in resultados
        assert "teeth_fdi" in resultados
        assert "performance" in resultados
        
        # Verificar estructura de summary
        assert "total" in resultados["summary"]
        assert "per_class" in resultados["summary"]
        
        # Verificar que detections es una lista
        assert isinstance(resultados["detections"], list)

    @skip_if_no_yolo
    @pytest.mark.slow
    def test_detecciones_tienen_formato_correcto(self):
        """Cada detección debe tener los campos necesarios"""
        img = self.crear_imagen_test()
        _, resultados = run_inference(img, confidence=0.25)
        
        # Si hay detecciones, verificar formato
        for deteccion in resultados["detections"]:
            assert "class_id" in deteccion
            assert "class_name" in deteccion
            assert "confidence" in deteccion
            assert "bbox" in deteccion
            assert "fdi" in deteccion
            
            # Verificar tipos
            assert isinstance(deteccion["class_id"], int)
            assert isinstance(deteccion["class_name"], str)
            assert isinstance(deteccion["confidence"], float)
            assert isinstance(deteccion["bbox"], list)
            assert len(deteccion["bbox"]) == 4
            assert isinstance(deteccion["fdi"], int)

    @skip_if_no_yolo
    @pytest.mark.slow
    def test_performance_metrics_presentes(self):
        """Los tiempos de performance deben estar presentes"""
        img = self.crear_imagen_test()
        _, resultados = run_inference(img, confidence=0.25)
        
        perf = resultados["performance"]
        
        assert "total_ms" in perf
        assert "model_load_ms" in perf
        assert "prediction_ms" in perf
        
        # Verificar que son números positivos
        assert perf["total_ms"] > 0
        assert perf["model_load_ms"] >= 0
        assert perf["prediction_ms"] > 0

    # ═══════════════════════════════════════════════════════
    # Tests con imágenes reales (si existen)
    # ═══════════════════════════════════════════════════════

    @skip_if_no_yolo
    @pytest.mark.slow
    @pytest.mark.integration
    def test_con_imagen_real_caries(self):
        """Test con imagen real de caries (si existe)"""
        imagen_path = Path("tests/test_images/radiografia_con_caries.jpg")
        
        if not imagen_path.exists():
            pytest.skip("Imagen de prueba no disponible")
        
        img = Image.open(imagen_path)
        img_anotada, resultados = run_inference(img, confidence=0.25)
        
        # Verificar que se ejecutó correctamente
        assert resultados["summary"]["total"] >= 0
        
        # Si detectó algo, verificar formato
        if resultados["summary"]["total"] > 0:
            assert len(resultados["detections"]) > 0
            assert resultados["report_text"] != ""

    @skip_if_no_yolo
    @pytest.mark.slow
    def test_con_imagen_real_normal(self):
        """Test con radiografía normal (sin anomalías)"""
        imagen_path = Path("tests/test_images/radiografia_normal.jpg")
        
        if not imagen_path.exists():
            pytest.skip("Imagen de prueba no disponible")
        
        img = Image.open(imagen_path)
        _, resultados = run_inference(img, confidence=0.25)
        
        # Verificar que se ejecutó
        assert "summary" in resultados
        
        # Si no hay detecciones, debe decirlo en el reporte
        if resultados["summary"]["total"] == 0:
            assert "No se detectaron problemas" in resultados["report_text"]


class TestCalculateFDI:
    """Tests para la función calculate_fdi"""

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_fdi_cuadrante_1(self):
        """Cuadrante 1 (superior derecho) - FDI 11-18"""
        # Punto en cuadrante superior derecho
        fdi = calculate_fdi(x_center_norm=0.25, y_center_norm=0.25)
        assert 11 <= fdi <= 18

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_fdi_cuadrante_2(self):
        """Cuadrante 2 (superior izquierdo) - FDI 21-28"""
        # Punto en cuadrante superior izquierdo
        fdi = calculate_fdi(x_center_norm=0.75, y_center_norm=0.25)
        assert 21 <= fdi <= 28

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_fdi_cuadrante_3(self):
        """Cuadrante 3 (inferior izquierdo) - FDI 31-38"""
        # Punto en cuadrante inferior izquierdo
        fdi = calculate_fdi(x_center_norm=0.75, y_center_norm=0.75)
        assert 31 <= fdi <= 38

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_fdi_cuadrante_4(self):
        """Cuadrante 4 (inferior derecho) - FDI 41-48"""
        # Punto en cuadrante inferior derecho
        fdi = calculate_fdi(x_center_norm=0.25, y_center_norm=0.75)
        assert 41 <= fdi <= 48

    @skip_if_no_yolo
    @pytest.mark.unit
    @pytest.mark.parametrize("x,y", [
        (0.1, 0.1),   # Superior derecho
        (0.9, 0.1),   # Superior izquierdo
        (0.9, 0.9),   # Inferior izquierdo
        (0.1, 0.9),   # Inferior derecho
    ])
    def test_fdi_retorna_numero_valido(self, x, y):
        """FDI siempre debe retornar número válido (11-48)"""
        fdi = calculate_fdi(x, y)
        assert isinstance(fdi, int)
        assert 11 <= fdi <= 48


class TestYOLOModelo:
    """Tests básicos del modelo"""

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_modelo_se_puede_cargar(self):
        """El modelo debe poder cargarse"""
        try:
            from app.model_store import get_model
            modelo = get_model()
            assert modelo is not None
        except Exception as e:
            pytest.fail(f"Error al cargar modelo: {e}")

    @skip_if_no_yolo
    @pytest.mark.unit
    def test_clases_definidas_correctamente(self):
        """Las clases deben estar bien definidas"""
        from app.inference import CLASS_NAMES, CLASS_COLORS
        
        # Verificar que existen 3 clases
        assert len(CLASS_NAMES) == 3, f"Se esperaban 3 clases, se encontraron {len(CLASS_NAMES)}"
        assert len(CLASS_COLORS) == 3
        
        # Verificar que tienen valores
        for clase_id, nombre in CLASS_NAMES.items():
            assert isinstance(nombre, str)
            assert len(nombre) > 0


# ═══════════════════════════════════════════════════════════
# Test de disponibilidad
# ═══════════════════════════════════════════════════════════

class TestYOLODisponibilidad:
    """Test de que el módulo está disponible"""
    
    @pytest.mark.unit
    def test_modulo_inference_existe(self):
        """El módulo inference debe existir"""
        assert YOLO_DISPONIBLE == True, "Módulo inference no disponible"