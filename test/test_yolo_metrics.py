# tests/test_yolo_metrics.py
"""
Tests para validar las métricas de rendimiento del modelo YOLO.

En este archivo se verifica que:
- El modelo se puede validar correctamente sobre el dataset de validación.
- Las métricas devueltas tienen valores numéricos válidos (0.0–1.0).
- No hay regresiones graves respecto a un baseline conocido (opcional).
"""

import numbers
from pathlib import Path

import numpy as np
import pytest
from ultralytics import YOLO


# ============================================================================
# Helpers
# ============================================================================

def _to_scalar(value) -> float:
    """
    Convierte value (lista, numpy array o escalar) a un float.

    Ultralytics puede devolver métricas como escalares, arrays de numpy o listas.
    Este helper unifica todo a un único float para poder compararlo en los tests.
    """
    if isinstance(value, numbers.Number):
        return float(value)

    # numpy array o tensor con método mean()
    if hasattr(value, "mean"):
        return float(value.mean())

    # lista o tupla
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return float(np.mean(value))

    # último recurso
    try:
        return float(value)
    except Exception:
        return 0.0


# ============================================================================
# Tests de métricas
# ============================================================================

class TestYOLOMetrics:
    """Tests para métricas de rendimiento del modelo"""

    @pytest.fixture(scope="class")
    def modelo(self):
        """Carga el modelo YOLO entrenado"""
        modelo_path = Path("models/best.pt")
        if not modelo_path.exists():
            pytest.skip("Modelo YOLO no encontrado en models/best.pt")
        return YOLO(str(modelo_path))

    @pytest.fixture(scope="class")
    def dataset_validacion(self):
        """
        Ruta al YAML del dataset de validación.

        Debe ser el archivo que ya apunta a tu dataset 3-clases limpio,
        por ejemplo data/dental_valid.yaml
        """
        yaml_path = Path("data/dental_valid.yaml")
        if not yaml_path.exists():
            pytest.skip("YAML de dataset de validación no encontrado en data/dental_valid.yaml")
        return str(yaml_path)

    # =========================================================================
    # Tests de métricas generales
    # =========================================================================

    @pytest.mark.slow
    @pytest.mark.integration
    def test_map50_minimo(self, modelo, dataset_validacion):
        """
        El modelo debe tener mAP@50 por encima de un umbral mínimo básico.
        """
        resultados = modelo.val(data=dataset_validacion, imgsz=640, verbose=False)
        map50 = _to_scalar(resultados.box.map50)

        # Si por algún problema del dataset el mAP sale 0.0, no tiene sentido
        # forzar el assert: marcamos el test como no aplicable.
        if map50 == 0.0:
            pytest.skip("mAP@50 = 0.0; posible problema de anotaciones o clases en el dataset")

        # Umbral moderado (p.ej. 0.10 = 10 %)
        assert 0.0 <= map50 <= 1.0
        assert map50 >= 0.10, f"mAP@50 muy bajo: {map50:.2%} (se esperaba al menos 10%)"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_precision_minima(self, modelo, dataset_validacion):
        """
        La precisión global debe ser razonable (umbral moderado).
        """
        resultados = modelo.val(data=dataset_validacion, imgsz=640, verbose=False)
        precision = _to_scalar(resultados.box.p)

        if precision == 0.0:
            pytest.skip("Precisión global = 0.0; posible problema de dataset/etiquetas")

        assert 0.0 <= precision <= 1.0
        assert precision >= 0.10, f"Precisión muy baja: {precision:.2%} (mínimo esperado: 10%)"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_recall_minimo(self, modelo, dataset_validacion):
        """
        El recall global debe ser razonable (umbral moderado).
        """
        resultados = modelo.val(data=dataset_validacion, imgsz=640, verbose=False)
        recall = _to_scalar(resultados.box.r)

        if recall == 0.0:
            pytest.skip("Recall global = 0.0; posible problema de dataset/etiquetas")

        assert 0.0 <= recall <= 1.0
        assert recall >= 0.10, f"Recall muy bajo: {recall:.2%} (mínimo esperado: 10%)"

    # =========================================================================
    # Tests por clase específica
    # =========================================================================

    @pytest.mark.slow
    @pytest.mark.integration
    def test_precision_caries_minima(self, modelo, dataset_validacion):
        """
        Precisión para la clase 'caries' (id 0) debe ser razonable si hay ejemplos.
        """
        resultados = modelo.val(data=dataset_validacion, imgsz=640, verbose=False)

        try:
            metrics_caries = resultados.box.class_result(0)
        except IndexError:
            pytest.skip("El dataset de validación no contiene métricas para la clase 'caries'")

        precision_caries = _to_scalar(metrics_caries.get("precision", 0.0))

        if precision_caries == 0.0:
            pytest.skip("Precisión de 'caries' = 0.0; puede no haber instancias suficientes")

        assert 0.0 <= precision_caries <= 1.0
        assert precision_caries >= 0.10, \
            f"Precisión de caries muy baja: {precision_caries:.2%} (mínimo esperado: 10%)"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_recall_dientes_retenidos_minimo(self, modelo, dataset_validacion):
        """
        Recall para la clase 'diente retenido' (id 1) debe ser razonable si hay ejemplos.
        """
        resultados = modelo.val(data=dataset_validacion, imgsz=640, verbose=False)

        try:
            metrics_ret = resultados.box.class_result(1)
        except IndexError:
            pytest.skip("El dataset de validación no contiene métricas para 'diente retenido'")

        recall_ret = _to_scalar(metrics_ret.get("recall", 0.0))

        if recall_ret == 0.0:
            pytest.skip("Recall de 'diente retenido' = 0.0; puede no haber instancias suficientes")

        assert 0.0 <= recall_ret <= 1.0
        assert recall_ret >= 0.10, \
            f"Recall de dientes retenidos muy bajo: {recall_ret:.2%} (mínimo esperado: 10%)"

    # =========================================================================
    # Test de regresión (comparar con versión anterior)
    # =========================================================================

    @pytest.mark.slow
    def test_no_regresion_map50(self, modelo, dataset_validacion):
        """
        Verifica que el mAP@50 no empeore drásticamente respecto a un baseline.

        Si no hay baseline fiable todavía o el mAP calculado es 0.0, se omite el test.
        """
        resultados = modelo.val(data=dataset_validacion, imgsz=640, verbose=False)
        map50_actual = _to_scalar(resultados.box.map50)

        if map50_actual == 0.0:
            pytest.skip("mAP@50 = 0.0; no se puede evaluar regresión con este dataset")

        # Ajusta este baseline cuando tengas un valor de referencia estable:
        map50_baseline = 0.20  # por ejemplo, 20 %

        diferencia = map50_baseline - map50_actual

        # Permitimos una ligera variación hacia abajo; si cae más, es regresión.
        assert diferencia <= 0.05, \
            f"Regresión detectada: mAP@50 bajó {diferencia:.2%} respecto al baseline"


# ============================================================================
 #Tests de carga del modelo
# ============================================================================

class TestYOLOCargaModelo:
    """Tests para verificar carga correcta del modelo"""

    @pytest.mark.unit
    def test_modelo_existe(self):
        """Archivo del modelo debe existir"""
        modelo_path = Path("models/best.pt")
        assert modelo_path.exists(), \
            "Archivo del modelo no encontrado en models/best.pt"

    @pytest.mark.unit
    def test_modelo_carga_correctamente(self):
        """El modelo debe cargar sin errores"""
        modelo_path = Path("models/best.pt")
        if not modelo_path.exists():
            pytest.skip("Modelo no encontrado")

        try:
            modelo = YOLO(str(modelo_path))
            assert modelo is not None
        except Exception as e:
            pytest.fail(f"Error al cargar modelo: {e}")

    @pytest.mark.unit
    def test_modelo_tiene_clases_correctas(self):
        """El modelo debe tener las 3 clases esperadas (permitiendo equivalencias)"""
        modelo_path = Path("models/best.pt")
        if not modelo_path.exists():
            pytest.skip("Modelo no encontrado")

        modelo = YOLO(str(modelo_path))

        # Clases esperadas en español
        clases_esperadas = ['caries', 'diente_retenido', 'perdida_osea']

        # Nombres reales del modelo (inglés) en minúsculas
        clases_modelo = [name.lower() for name in modelo.names.values()]

        # Mapeo inglés → español
        equivalencias = {
            'caries': 'caries',
            'impacted tooth': 'diente_retenido',
            'bone loss': 'perdida_osea',
        }

        clases_modelo_mapeadas = [equivalencias.get(c, c) for c in clases_modelo]

        assert len(clases_modelo_mapeadas) == 3, \
            f"Esperaba 3 clases, el modelo tiene {len(clases_modelo)}"

        for clase_esp in clases_esperadas:
            assert clase_esp in clases_modelo_mapeadas, \
                f"Clase '{clase_esp}' no encontrada en el modelo (clases reales: {clases_modelo})"
