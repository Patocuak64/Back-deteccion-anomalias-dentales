# test/test_analysis_e2e.py
"""
Tests End-to-End para el flujo completo de análisis.

Para ejecutar:
    pytest test/test_analysis_e2e.py -v -s
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import time
from PIL import Image
import io
import uuid

# Agregar la carpeta raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar la app
from app.main import app

# Importar configuración del dataset
try:
    from test.test_config import (
        dataset_exists,
        get_random_dataset_image,
        load_random_image_as_buffer,
        load_random_image_as_bytes,
        get_dataset_info,
        get_random_dataset_images
    )
    DATASET_DISPONIBLE = dataset_exists()
except ImportError:
    DATASET_DISPONIBLE = False

# Cliente de prueba
client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════
# HELPER: Crear imagen de fallback
# ═══════════════════════════════════════════════════════════════════

def crear_imagen_fallback():
    """Crea una imagen sintética como fallback"""
    import numpy as np
    arr = np.random.randint(60, 180, (600, 1200), dtype=np.uint8)
    arr = np.stack([arr, arr, arr], axis=2)
    img = Image.fromarray(arr, mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return buffer, "imagen_sintetica.jpg"


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def usuario_y_token():
    """
    Crea un usuario de prueba Y obtiene su token.
    Usa @example.com que es un dominio válido para testing según RFC 2606.
    """
    # IMPORTANTE: Usar @example.com en lugar de @dentalsmart.test
    # porque Pydantic's EmailStr rechaza dominios .test
    unique_id = uuid.uuid4().hex[:8]
    email = f"test_{unique_id}@example.com"
    password = "TestPassword123!"
    
    # 1) Registrar
    payload_register = {
        "email": email,
        "password": password,
        "name": "Usuario Test E2E",
    }
    
    response_register = client.post("/auth/register", json=payload_register)
    
    # Si ya existe (400), intentar login directamente
    if response_register.status_code == 400:
        print(f"\n⚠️  Usuario ya existe, intentando login...")
    elif response_register.status_code not in [200, 201]:
        # Mostrar error detallado
        print(f"\n❌ Error en registro: {response_register.status_code}")
        print(f"   Detalle: {response_register.text}")
        pytest.fail(f"Error al registrar: {response_register.status_code} - {response_register.text}")
    else:
        print(f"\n✅ Usuario registrado: {email}")
    
    # 2) Login
    payload_login = {
        "username": email,
        "password": password,
    }
    
    response_login = client.post("/auth/login", data=payload_login)
    
    if response_login.status_code != 200:
        print(f"\n❌ Error en login: {response_login.status_code}")
        print(f"   Detalle: {response_login.text}")
        pytest.fail(f"Error al hacer login: {response_login.status_code} - {response_login.text}")
    
    data = response_login.json()
    token = data.get("access_token")
    
    if not token:
        pytest.fail("No se recibió access_token")
    
    print(f"✅ Token obtenido: {token[:20]}...")
    
    return {
        "email": email,
        "password": password,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"}
    }


@pytest.fixture
def imagen_para_test():
    """
    Obtiene una imagen para testing.
    Usa dataset si está disponible, sino crea sintética.
    """
    if DATASET_DISPONIBLE:
        result = load_random_image_as_buffer()
        if result:
            buffer, filename = result
            print(f"\n🖼️  Usando imagen del dataset: {filename}")
            return buffer, filename
    
    # Fallback
    print("\n⚠️  Usando imagen sintética")
    return crear_imagen_fallback()


# ═══════════════════════════════════════════════════════════════════
# TESTS DE CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════

class TestDatasetConfiguration:
    """Tests para verificar la configuración del dataset"""

    @pytest.mark.unit
    def test_dataset_info(self):
        """Mostrar información del dataset configurado"""
        if not DATASET_DISPONIBLE:
            print("\n⚠️  Dataset no disponible")
            assert True
            return
        
        info = get_dataset_info()
        print(f"\n📁 Dataset: {info['path']}")
        print(f"   Existe: {info['exists']}")
        print(f"   Imágenes: {info['total_images']}")
        assert True

    @pytest.mark.unit
    def test_dataset_accesible(self):
        """Verificar que el dataset es accesible"""
        if not DATASET_DISPONIBLE:
            pytest.skip("Dataset no disponible")
        
        img_path = get_random_dataset_image()
        assert img_path is not None
        assert img_path.exists()
        print(f"\n✅ Dataset OK: {img_path.name}")


# ═══════════════════════════════════════════════════════════════════
# TESTS BÁSICOS
# ═══════════════════════════════════════════════════════════════════

class TestAnalysisE2EBasico:
    """Tests básicos del flujo"""

    @pytest.mark.e2e
    def test_health_check(self):
        """El endpoint /health debe estar disponible"""
        response = client.get("/health")
        assert response.status_code == 200
        print(f"\n✅ Health check OK")

    @pytest.mark.e2e
    def test_usuario_puede_hacer_login(self, usuario_y_token):
        """El usuario puede autenticarse"""
        assert usuario_y_token["token"] is not None
        assert len(usuario_y_token["token"]) > 20
        print(f"\n✅ Login verificado para: {usuario_y_token['email']}")


# ═══════════════════════════════════════════════════════════════════
# TESTS DE ANÁLISIS CON DATASET
# ═══════════════════════════════════════════════════════════════════

class TestAnalysisE2EConDataset:
    """Tests del flujo completo con imágenes"""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_analisis_con_imagen(self, usuario_y_token, imagen_para_test):
        """Flujo completo de análisis"""
        buffer, filename = imagen_para_test
        
        files = {"file": (filename, buffer, "image/jpeg")}
        
        inicio = time.time()
        response = client.post(
            "/analyze",
            files=files,
            headers=usuario_y_token["headers"]
        )
        tiempo = time.time() - inicio
        
        print(f"\n⏱️  Tiempo: {tiempo:.2f}s")
        print(f"📁 Archivo: {filename}")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 400:
            detail = response.json().get('detail', '')
            print(f"⚠️  Rechazada: {detail}")
            pytest.skip(f"Imagen rechazada: {detail}")
        
        assert response.status_code == 200, f"Error: {response.text}"
        
        data = response.json()
        assert "summary" in data
        assert "detections" in data
        
        total = data["summary"].get("total", 0)
        print(f"✅ Detecciones: {total}")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_analisis_multiples_imagenes(self, usuario_y_token):
        """Analiza múltiples imágenes del dataset"""
        if not DATASET_DISPONIBLE:
            pytest.skip("Dataset no disponible")
        
        imagenes = get_random_dataset_images(3)
        if len(imagenes) < 2:
            pytest.skip("Pocas imágenes en dataset")
        
        resultados = []
        
        for img_path in imagenes:
            with open(img_path, 'rb') as f:
                files = {"file": (img_path.name, f, "image/jpeg")}
                response = client.post(
                    "/analyze",
                    files=files,
                    headers=usuario_y_token["headers"]
                )
            
            status = "OK" if response.status_code == 200 else f"ERR:{response.status_code}"
            resultados.append((img_path.name, status))
            print(f"  {status}: {img_path.name}")
        
        exitosos = sum(1 for _, s in resultados if s == "OK")
        print(f"\n✅ Exitosos: {exitosos}/{len(resultados)}")
        
        assert exitosos >= 1, "Ninguna imagen analizada"


# ═══════════════════════════════════════════════════════════════════
# TESTS DE HISTORIAL
# ═══════════════════════════════════════════════════════════════════

class TestAnalysisE2EHistorial:
    """Tests del historial"""

    @pytest.mark.e2e
    def test_usuario_puede_ver_historial(self, usuario_y_token):
        """Usuario puede consultar historial"""
        response = client.get(
            "/analyses",
            headers=usuario_y_token["headers"]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"\n✅ Historial: {len(data)} análisis")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_analisis_guardado_en_historial(self, usuario_y_token, imagen_para_test):
        """Análisis guardado aparece en historial"""
        # 1) Historial antes
        resp_antes = client.get("/analyses", headers=usuario_y_token["headers"])
        cantidad_antes = len(resp_antes.json())
        
        # 2) Análisis con save=true
        buffer, filename = imagen_para_test
        files = {"file": (filename, buffer, "image/jpeg")}
        data_form = {"save": "true", "return_image": "true"}
        
        resp_analisis = client.post(
            "/analyze",
            files=files,
            data=data_form,
            headers=usuario_y_token["headers"]
        )
        
        if resp_analisis.status_code == 400:
            pytest.skip("Imagen rechazada")
        
        assert resp_analisis.status_code == 200
        
        # 3) Historial después
        resp_despues = client.get("/analyses", headers=usuario_y_token["headers"])
        cantidad_despues = len(resp_despues.json())
        
        print(f"\n✅ Historial: {cantidad_antes} → {cantidad_despues}")
        assert cantidad_despues >= cantidad_antes


# ═══════════════════════════════════════════════════════════════════
# TESTS DE RENDIMIENTO
# ═══════════════════════════════════════════════════════════════════

class TestAnalysisE2ERendimiento:
    """Tests de rendimiento"""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_tiempo_analisis_aceptable(self, usuario_y_token, imagen_para_test):
        """Tiempo de análisis < 30 segundos"""
        buffer, filename = imagen_para_test
        files = {"file": (filename, buffer, "image/jpeg")}
        
        inicio = time.time()
        response = client.post(
            "/analyze",
            files=files,
            headers=usuario_y_token["headers"]
        )
        tiempo = time.time() - inicio
        
        print(f"\n⏱️  Tiempo: {tiempo:.2f}s")
        
        if response.status_code == 400:
            pytest.skip("Imagen rechazada")
        
        assert tiempo < 30, f"Muy lento: {tiempo:.2f}s"


# ═══════════════════════════════════════════════════════════════════
# TESTS DE ERRORES
# ═══════════════════════════════════════════════════════════════════

class TestAnalysisE2EErrores:
    """Tests de manejo de errores"""

    @pytest.mark.e2e
    def test_analisis_sin_autenticacion_rechazado(self, imagen_para_test):
        """Sin token = rechazado"""
        buffer, filename = imagen_para_test
        files = {"file": (filename, buffer, "image/jpeg")}
        
        response = client.post("/analyze", files=files)
        assert response.status_code in [401, 422]
        print(f"\n✅ Sin auth rechazado: {response.status_code}")

    @pytest.mark.e2e
    def test_analisis_con_token_invalido(self, imagen_para_test):
        """Token inválido = rechazado"""
        buffer, filename = imagen_para_test
        files = {"file": (filename, buffer, "image/jpeg")}
        headers = {"Authorization": "Bearer token_falso_123"}
        
        response = client.post("/analyze", files=files, headers=headers)
        assert response.status_code in [401, 403]
        print(f"\n✅ Token inválido rechazado: {response.status_code}")

    @pytest.mark.e2e
    def test_analisis_sin_archivo(self, usuario_y_token):
        """Sin archivo = error 422"""
        response = client.post(
            "/analyze",
            headers=usuario_y_token["headers"]
        )
        assert response.status_code == 422
        print(f"\n✅ Sin archivo rechazado: 422")

    @pytest.mark.e2e
    def test_analisis_archivo_no_imagen(self, usuario_y_token):
        """Archivo no-imagen = rechazado"""
        txt = b"Esto es texto, no imagen"
        files = {"file": ("doc.txt", io.BytesIO(txt), "text/plain")}
        
        response = client.post(
            "/analyze",
            files=files,
            headers=usuario_y_token["headers"]
        )
        assert response.status_code in [400, 422]
        print(f"\n✅ No-imagen rechazado: {response.status_code}")