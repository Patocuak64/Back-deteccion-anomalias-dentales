# app/email_validator.py
"""
Validador de correos electrónicos para el backend.
Valida formato y estructura para evitar correos inválidos como .....@gmail.com
"""

import re
from typing import Tuple


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Valida un correo electrónico.
    
    Args:
        email: Correo a validar
        
    Returns:
        Tupla (es_valido, mensaje_error)
        - es_valido: True si el correo es válido
        - mensaje_error: None si es válido, mensaje descriptivo si no
    """
    
    if not email or not isinstance(email, str):
        return False, "El correo es requerido"
    
    # Normalizar
    email = email.strip().lower()
    
    # Regex básico
    basic_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    if not re.match(basic_pattern, email):
        return False, "Formato de correo inválido"
    
    # Dividir en usuario y dominio
    try:
        user, domain = email.split('@')
    except ValueError:
        return False, "El correo debe contener un @"
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VALIDAR USUARIO (parte antes del @)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if not user:
        return False, "La parte del usuario no puede estar vacía"
    
    # No puede tener puntos consecutivos (..)
    if '..' in user:
        return False, "El correo no puede contener puntos consecutivos (..)"
    
    # No puede empezar con punto
    if user.startswith('.'):
        return False, "El correo no puede empezar con un punto"
    
    # No puede terminar con punto
    if user.endswith('.'):
        return False, "El correo no puede terminar con un punto antes del @"
    
    # Debe tener al menos 1 carácter alfanumérico
    if not re.search(r'[a-z0-9]', user):
        return False, "El correo debe contener al menos una letra o número"
    
    # No puede consistir solo de puntos
    if user.replace('.', '') == '':
        return False, "El correo no puede consistir solo de puntos"
    
    # Caracteres permitidos en el usuario
    # Letras, números, puntos, guiones, guiones bajos
    if not re.match(r'^[a-z0-9._-]+$', user):
        return False, "El usuario contiene caracteres no permitidos"
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VALIDAR DOMINIO (parte después del @)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if not domain:
        return False, "El dominio no puede estar vacío"
    
    # Debe tener un punto
    if '.' not in domain:
        return False, "El dominio debe tener una extensión válida"
    
    # No puede tener puntos consecutivos
    if '..' in domain:
        return False, "El dominio no puede contener puntos consecutivos"
    
    # No puede empezar o terminar con punto
    if domain.startswith('.') or domain.endswith('.'):
        return False, "El dominio no puede empezar o terminar con un punto"
    
    # Dividir dominio en partes
    parts = domain.split('.')
    
    # Debe tener al menos 2 partes (nombre + extensión)
    if len(parts) < 2:
        return False, "El dominio debe tener al menos un nombre y una extensión"
    
    # Validar cada parte del dominio
    for part in parts:
        if not part:
            return False, "El dominio contiene partes vacías"
        
        # Solo letras, números y guiones
        if not re.match(r'^[a-z0-9-]+$', part):
            return False, f"Parte del dominio '{part}' contiene caracteres inválidos"
        
        # No puede empezar o terminar con guion
        if part.startswith('-') or part.endswith('-'):
            return False, f"Parte del dominio '{part}' no puede empezar o terminar con guion"
    
    # Validar extensión (última parte)
    extension = parts[-1]
    
    # Extensión debe tener al menos 2 caracteres
    if len(extension) < 2:
        return False, "La extensión del dominio debe tener al menos 2 caracteres"
    
    # Extensión solo puede contener letras
    if not re.match(r'^[a-z]+$', extension):
        return False, "La extensión del dominio solo puede contener letras"
    
    # Longitud máxima razonable
    if len(email) > 320:  # Estándar RFC 5321
        return False, "El correo es demasiado largo"
    
    #  Correo válido
    return True, None


def normalize_email(email: str) -> str:
    """
    Normaliza un correo electrónico (lowercase, strip)
    """
    if not email:
        return email
    return email.strip().lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CASOS DE PRUEBA (para desarrollo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Casos de prueba
    test_cases = [
        # (email, deberia_ser_valido)
        ("usuario@gmail.com", True),
        ("user.name@example.co.uk", True),
        ("user_name@domain.com", True),
        ("user-name@domain.com", True),
        
        # Casos INVÁLIDOS
        (".....@gmail.com", False),
        ("..user@gmail.com", False),
        ("user..name@gmail.com", False),
        ("user.@gmail.com", False),
        (".user@gmail.com", False),
        ("@gmail.com", False),
        ("user@", False),
        ("user", False),
        ("user@domain", False),
        ("user@domain.", False),
        ("user@.domain.com", False),
        ("user@domain..com", False),
        ("user name@domain.com", False),
        ("user@domain .com", False),
        ("", False),
    ]
    
    print("Ejecutando pruebas de validación de email...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for email, should_be_valid in test_cases:
        is_valid, error = validate_email(email)
        
        if is_valid == should_be_valid:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{status} | '{email}' | Esperado: {should_be_valid}, Got: {is_valid}")
        if error:
            print(f"         Error: {error}")
    
    print("=" * 60)
    print(f"Resultados: {passed} passed, {failed} failed")
