# app/email_validator.py
"""
Validador de correos electrónicos optimizado
Valida formato y estructura para evitar correos inválidos
"""

import re
from typing import Tuple


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Valida un correo electrónico con reglas optimizadas.
    
    Returns:
        (es_valido, mensaje_error)
    """
    
    # 1. Validaciones básicas
    if not email or not isinstance(email, str):
        return False, "El correo es requerido"
    
    email = email.strip().lower()
    
    if '@' not in email:
        return False, "El correo debe contener un @"
    
    # 2. Dividir y validar estructura básica
    try:
        user, domain = email.split('@', 1)
    except ValueError:
        return False, "Formato de correo inválido"
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VALIDAR USUARIO
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Longitud mínima
    if len(user) < 3:
        return False, "El nombre de usuario debe tener al menos 3 caracteres"
    
    # Caracteres permitidos (letras, números, punto, guion, guion bajo)
    if not re.match(r'^[a-z0-9._-]+$', user):
        return False, "El usuario contiene caracteres no permitidos"
    
    # No puede empezar/terminar con punto
    if user[0] == '.' or user[-1] == '.':
        return False, "El usuario no puede empezar o terminar con punto"
    
    # No puede tener puntos consecutivos
    if '..' in user:
        return False, "El correo no puede contener puntos consecutivos"
    
    # No puede ser solo números (después de quitar ._-)
    if user.replace('.', '').replace('_', '').replace('-', '').isdigit():
        return False, "El nombre de usuario no puede ser solo números"
    
    # Debe tener al menos 2 letras
    if sum(1 for c in user if c.isalpha()) < 2:
        return False, "El nombre de usuario debe contener al menos 2 letras"
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VALIDAR DOMINIO
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Debe tener punto y no estar vacío
    if not domain or '.' not in domain:
        return False, "El dominio debe tener una extensión válida"
    
    # No puede tener puntos consecutivos
    if '..' in domain:
        return False, "El dominio no puede contener puntos consecutivos"
    
    # No puede empezar/terminar con punto
    if domain[0] == '.' or domain[-1] == '.':
        return False, "El dominio no puede empezar o terminar con punto"
    
    # Validar partes del dominio
    parts = domain.split('.')
    
    if len(parts) < 2:
        return False, "El dominio debe tener al menos un nombre y extensión"
    
    for part in parts:
        if not part:
            return False, "El dominio contiene partes vacías"
        
        # Solo letras, números y guiones
        if not re.match(r'^[a-z0-9-]+$', part):
            return False, "El dominio contiene caracteres no permitidos"
        
        # No puede empezar/terminar con guion
        if part[0] == '-' or part[-1] == '-':
            return False, "Las partes del dominio no pueden empezar/terminar con guion"
    
    # Extensión debe tener al menos 2 caracteres y solo letras
    extension = parts[-1]
    if len(extension) < 2 or not extension.isalpha():
        return False, "La extensión del dominio debe tener al menos 2 letras"
    
    # Longitud total máxima
    if len(email) > 320:
        return False, "El correo es demasiado largo"
    
    # Correo válido
    return True, None


def normalize_email(email: str) -> str:
    """Normaliza un correo (lowercase, strip)"""
    return email.strip().lower() if email else email


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CASOS DE PRUEBA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    test_cases = [
        # Válidos
        ("usuario@gmail.com", True),
        ("user.name@example.co.uk", True),
        ("user_name@domain.com", True),
        ("user-name@domain.com", True),
        ("pera@gmail.com", True),
        ("juan.perez@gmail.com", True),
        ("doctor123@hospital.com", True),
        
        # Inválidos - Estructura
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
        
        # Inválidos - Validaciones adicionales
        ("1@gmail.com", False),
        ("12@gmail.com", False),
        ("123@gmail.com", False),
        ("a1@gmail.com", False),
        ("ab@gmail.com", False),
    ]
    
    print("Pruebas de validación de email...")
    print("=" * 60)
    
    passed = failed = 0
    
    for email, should_be_valid in test_cases:
        is_valid, error = validate_email(email)
        
        if is_valid == should_be_valid:
            print(f"✅ | '{email}'")
            passed += 1
        else:
            print(f"❌ | '{email}' | Esperado: {should_be_valid}, Got: {is_valid}")
            if error:
                print(f"   Error: {error}")
            failed += 1
    
    print("=" * 60)
    print(f"Resultado: {passed} passed, {failed} failed")




































