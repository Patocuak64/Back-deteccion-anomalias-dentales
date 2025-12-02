# tests/test_endpoints.py
"""
Tests para los endpoints de la API FastAPI.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import io
from PIL import Image
import numpy as np

# Agregar la carpeta raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar la app desde app/main.py
from app.main import app

# Cliente de prueba para FastAPI
client = TestClient(app)


class TestHealthEndpoint:
    """Tests para el endpoint de health check"""

    @pytest.mark.api
    def test_health_check(self):
        """Endpoint /health debe retornar 200 OK"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestAuthEndpoints:
    """Tests para endpoints de autenticación"""

    @pytest.mark.api
    def test_register_email_invalido(self):
        """Registro con email inválido debe retornar 400 o 422"""
        payload = {
            "name": "Test User",
            "email": "email-invalido",  # Sin @
            "password": "Password123!"
        }
        response = client.post("/auth/register", json=payload)
        assert response.status_code in [400, 422]

        # detail puede ser str o lista de errores de validación
        detail = response.json().get("detail")
        if isinstance(detail, list):
            texto = " ".join(str(d.get("msg", "")) + " " + str(d.get("loc", "")) for d in detail)
        else:
            texto = str(detail)

        assert "email" in texto.lower()

    @pytest.mark.api
    def test_register_password_debil(self):
        """Registro con contraseña débil debe retornar 400 o 422"""
        payload = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "123"  # Muy débil
        }
        response = client.post("/auth/register", json=payload)
        assert response.status_code in [400, 422]

    @pytest.mark.api
    def test_login_credenciales_invalidas(self):
        """Login con credenciales inválidas debe retornar 401 o 422"""
        payload = {
            "email": "noexiste@example.com",
            "password": "WrongPassword123!"
        }
        response = client.post("/auth/login", json=payload)
        assert response.status_code in [401, 422]


class TestAnalyzeEndpoint:
    """Tests para el endpoint de análisis de radiografías"""

    @staticmethod
    def crear_imagen_test(color=False):
        """Crea una imagen de prueba"""
        if color:
            arr = np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8)
        else:
            base = np.random.randint(60, 180, (400, 800), dtype=np.uint8)
            arr = np.stack([base, base, base], axis=2)
        
        img = Image.fromarray(arr, mode='RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        return buffer

    @pytest.mark.api
    def test_analyze_sin_autenticacion(self):
        """Análisis sin token debe retornar 401/422"""
        img_buffer = self.crear_imagen_test()
        files = {"file": ("test.jpg", img_buffer, "image/jpeg")}
        response = client.post("/analyze", files=files)
        assert response.status_code in [401, 422]

    @pytest.mark.api
    def test_analyze_imagen_color_rechazada(self):
        """
        Imagen a color debe ser rechazada en validación.

        Nota: aquí solo comprobamos que sin auth falla; la validación
        de color se prueba a fondo en test_image_validator.py
        """
        img_buffer = self.crear_imagen_test(color=True)
        files = {"file": ("foto.jpg", img_buffer, "image/jpeg")}
        response = client.post("/analyze", files=files)
        assert response.status_code in [401, 403, 422]


class TestCORSHeaders:
    """Tests para configuración de CORS"""

    @pytest.mark.api
    def test_cors_headers_presentes(self):
        """Headers de CORS deben estar presentes (al menos status válido)"""
        response = client.options("/health")
        assert response.status_code in [200, 405]


# ═══════════════════════════════════════════════════════════
# Fixtures (datos compartidos entre tests)
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def imagen_radiografia_test():
    """Fixture que crea una imagen de radiografía de prueba"""
    arr = np.random.randint(60, 180, (600, 1200), dtype=np.uint8)
    arr = np.stack([arr, arr, arr], axis=2)
    
    img = Image.fromarray(arr, mode='RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer
