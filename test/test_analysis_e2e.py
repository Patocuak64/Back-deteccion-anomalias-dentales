# tests/test_analysis_e2e.py
"""
Tests End-to-End para el flujo completo de análisis.

Requisitos:
- Backend corriendo en http://localhost:8000
- Base de datos limpia (o usar usuario único por test)
- Imágenes de prueba en test/test_images/

Para ejecutar:
1. Iniciar backend: uvicorn app.main:app --reload
2. Ejecutar: pytest tests/test_analysis_e2e.py -v -s
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import time
from PIL import Image
import io
import numpy as np
import uuid

# Agregar la carpeta raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar la app
from app.main import app

# Cliente de prueba
client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES (datos compartidos entre tests)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def usuario_test():
    """
    Crea un usuario de prueba único.
    Se ejecuta UNA VEZ por módulo (todos los tests lo comparten).
    """
    # Email único para evitar conflictos
    email = f"test_{uuid.uuid4().hex[:8]}@dentalsmart.dev"
    password = "TestPassword123!"
    name = "Usuario Test E2E"

    payload = {
        "email": email,
        "password": password,
        "name": name,
    }

    response = client.post("/auth/register", json=payload)

    # 200 → se registró; 400 → ya existía (también válido)
    if response.status_code not in [200, 400]:
        pytest.fail(
            f"Error al registrar usuario: {response.status_code} - {response.text}"
        )

    print(f"\n✅ Usuario de prueba creado: {email}")

    return {
        "email": email,
        "password": password,
        "name": name,
    }


@pytest.fixture(scope="module")
def token_auth(usuario_test):
    """
    Obtiene token JWT válido para el usuario de prueba.
    IMPORTANTE: el endpoint /auth/login usa OAuth2PasswordRequestForm,
    por lo que espera campos 'username' y 'password' en FORM DATA,
    NO JSON.
    """
    payload = {
        "username": usuario_test["email"],  # el backend usa username, nosotros le pasamos el email
        "password": usuario_test["password"],
    }

    # NOTA: usamos data=..., NO json=...
    response = client.post("/auth/login", data=payload)

    if response.status_code != 200:
        pytest.fail(
            f"Error al hacer login: {response.status_code} - {response.text}"
        )

    data = response.json()
    token = data.get("access_token")

    if not token:
        pytest.fail("No se recibió access_token en la respuesta")

    print(f"✅ Token obtenido: {token[:20]}...")

    return token


@pytest.fixture
def imagen_radiografia_test():
    """
    Crea una imagen de radiografía de prueba.
    Se ejecuta CADA VEZ que un test la necesite.
    """
    # Crear imagen en escala de grises (pseudo-rx)
    arr = np.random.randint(60, 180, (600, 1200), dtype=np.uint8)
    arr = np.stack([arr, arr, arr], axis=2)

    img = Image.fromarray(arr, mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return buffer


# ═══════════════════════════════════════════════════════════════════
# TESTS END-TO-END
# ═══════════════════════════════════════════════════════════════════


class TestAnalysisE2EBasico:
    """Tests básicos del flujo completo"""

    @pytest.mark.e2e
    def test_health_check(self):
        """El endpoint /health debe estar disponible"""
        response = client.get("/health")
        assert response.status_code == 200

    @pytest.mark.e2e
    def test_usuario_puede_hacer_login(self, usuario_test):
        """El usuario de prueba puede hacer login"""

        payload = {
            "username": usuario_test["email"],
            "password": usuario_test["password"],
        }

        response = client.post("/auth/login", data=payload)

        assert response.status_code == 200, response.text
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] != ""


class TestAnalysisE2EFlujoCompleto:
    """Tests del flujo completo de análisis"""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_flujo_completo_con_imagen_sintetica(
        self, token_auth, imagen_radiografia_test
    ):
        """
        Flujo completo:
        1. Usuario autenticado
        2. Sube imagen
        3. Sistema analiza
        4. Retorna resultados
        """
        files = {
            "file": ("radiografia_test.jpg", imagen_radiografia_test, "image/jpeg")
        }
        headers = {"Authorization": f"Bearer {token_auth}"}

        inicio = time.time()
        response = client.post("/analyze", files=files, headers=headers)
        tiempo_total = time.time() - inicio

        print(f"\n⏱️  Tiempo total de análisis: {tiempo_total:.2f}s")

        if response.status_code == 400:
            print(
                f"⚠️  Imagen sintética rechazada (esperado): "
                f"{response.json().get('detail')}"
            )
            pytest.skip("Imagen sintética rechazada por validación")

        assert response.status_code == 200, f"Error: {response.text}"

        data = response.json()
        assert "summary" in data or "detections" in data or "detecciones" in data

        print("✅ Análisis completado exitosamente")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_flujo_con_imagen_real(self, token_auth):
        """Test con imagen real (si existe)"""

        # IMPORTANTE: tu carpeta es 'test/test_images', no 'tests/...'
        imagen_path = Path("test/test_images/radiografia_con_caries.jpg")

        if not imagen_path.exists():
            pytest.skip("Imagen de prueba no disponible")

        with imagen_path.open("rb") as f:
            files = {"file": ("radiografia.jpg", f, "image/jpeg")}
            headers = {"Authorization": f"Bearer {token_auth}"}

            response = client.post("/analyze", files=files, headers=headers)

        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            print("\n✅ Análisis con imagen real completado")
            if "detections" in data or "detecciones" in data:
                detecciones = data.get("detections") or data.get(
                    "detecciones", []
                )
                print(f"   Detecciones: {len(detecciones)}")

    @pytest.mark.e2e
    def test_analisis_sin_autenticacion_rechazado(
        self, imagen_radiografia_test
    ):
        """Análisis sin token debe ser rechazado"""
        files = {"file": ("test.jpg", imagen_radiografia_test, "image/jpeg")}
        response = client.post("/analyze", files=files)
        assert response.status_code in [401, 422]

    @pytest.mark.e2e
    def test_analisis_con_token_invalido_rechazado(
        self, imagen_radiografia_test
    ):
        """Análisis con token inválido debe ser rechazado"""
        files = {"file": ("test.jpg", imagen_radiografia_test, "image/jpeg")}
        headers = {"Authorization": "Bearer token_falso_12345"}
        response = client.post("/analyze", files=files, headers=headers)
        assert response.status_code in [401, 403, 422]


class TestAnalysisE2EHistorial:
    """Tests del historial de análisis"""

    @pytest.mark.e2e
    def test_usuario_puede_ver_historial(self, token_auth):
        headers = {"Authorization": f"Bearer {token_auth}"}
        response = client.get("/analyses", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

        print(f"\n✅ Historial consultado: {len(data)} análisis")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_analisis_aparece_en_historial(
        self, token_auth, imagen_radiografia_test
    ):
        headers = {"Authorization": f"Bearer {token_auth}"}

        # 1) Historial antes
        response_antes = client.get("/analyses", headers=headers)
        historial_antes = response_antes.json()
        cantidad_antes = len(historial_antes)

        # 2) Nuevo análisis
        files = {"file": ("test.jpg", imagen_radiografia_test, "image/jpeg")}
        response_analisis = client.post("/analyze", files=files, headers=headers)

        if response_analisis.status_code == 400:
            pytest.skip("Imagen rechazada por validación")

        # 3) Historial después
        response_despues = client.get("/analyses", headers=headers)
        historial_despues = response_despues.json()
        cantidad_despues = len(historial_despues)

        if response_analisis.status_code == 200:
            assert cantidad_despues >= cantidad_antes
            print(
                f"\n✅ Análisis agregado al historial "
                f"({cantidad_antes} → {cantidad_despues})"
            )


class TestAnalysisE2ERendimiento:
    """Tests de rendimiento"""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_tiempo_analisis_aceptable(
        self, token_auth, imagen_radiografia_test
    ):
        files = {"file": ("test.jpg", imagen_radiografia_test, "image/jpeg")}
        headers = {"Authorization": f"Bearer {token_auth}"}

        inicio = time.time()
        response = client.post("/analyze", files=files, headers=headers)
        tiempo_total = time.time() - inicio

        print(f"\n⏱️  Tiempo de análisis: {tiempo_total:.2f}s")

        assert tiempo_total < 30, f"Análisis muy lento: {tiempo_total:.2f}s"


class TestAnalysisE2EErrores:
    """Tests de manejo de errores"""

    @pytest.mark.e2e
    def test_analisis_sin_archivo(self, token_auth):
        headers = {"Authorization": f"Bearer {token_auth}"}
        response = client.post("/analyze", headers=headers)
        assert response.status_code == 422

    @pytest.mark.e2e
    def test_analisis_archivo_no_imagen(self, token_auth):
        txt_bytes = b"Este es un archivo de texto, no una imagen"
        files = {"file": ("documento.txt", io.BytesIO(txt_bytes), "text/plain")}
        headers = {"Authorization": f"Bearer {token_auth}"}

        response = client.post("/analyze", files=files, headers=headers)
        assert response.status_code in [400, 422]
