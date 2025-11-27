# app/auth.py
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, field_validator
import time

from . import models
from .dependencies import get_db
from .settings import settings
from .email_validator import validate_email  # ⚡ AGREGADO

router = APIRouter(prefix="/auth", tags=["auth"])

# ═══════════════════════════════════════════════════════════════════════
# ⚡ OPTIMIZACIÓN: Configurar bcrypt con rounds ajustables
# ═══════════════════════════════════════════════════════════════════════
pwd_ctx = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.BCRYPT_ROUNDS
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str | None = None
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v):
        """Validación de contraseña"""
        if len(v) < 6:
            raise ValueError('La contraseña debe tener al menos 6 caracteres')
        if len(v) > 128:
            raise ValueError('La contraseña es demasiado larga')
        return v


class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: str | None
    class Config: 
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


def get_password_hash(p: str) -> str:
    """Hash de contraseña con timing"""
    start = time.time()
    hashed = pwd_ctx.hash(p)
    elapsed = (time.time() - start) * 1000
    print(f"[AUTH] Hash generado en {elapsed:.0f}ms (rounds={settings.BCRYPT_ROUNDS})")
    return hashed


def verify_password(p: str, h: str) -> bool:
    """
    Verificación de contraseña con timing y manejo de errores
    """
    start = time.time()
    try:
        result = pwd_ctx.verify(p, h)
        elapsed = (time.time() - start) * 1000
        print(f"[AUTH] Verificación en {elapsed:.0f}ms")
        return result
    except Exception as e:
        # Si falla la verificación (hash antiguo, corrupto, o inválido)
        elapsed = (time.time() - start) * 1000
        print(f"[AUTH] ❌ Error al verificar hash en {elapsed:.0f}ms: {type(e).__name__}")
        return False


def create_access_token(data: dict, minutes: int = None):
    """Crea token JWT"""
    if minutes is None:
        minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(minutes=minutes)
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def get_current_user(
    db: Session = Depends(get_db), 
    token: str = Depends(oauth2_scheme)
) -> models.User:
    """Obtiene usuario actual desde token JWT"""
    cred_exc = HTTPException(
        status_code=401, 
        detail="Credenciales inválidas", 
        headers={"WWW-Authenticate": "Bearer"}
    )
    
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        email: str = payload.get("sub")
        if not email: 
            raise cred_exc
    except JWTError as e:
        print(f"[AUTH] Error JWT: {e}")
        raise cred_exc
    
    user = db.query(models.User).filter(
        models.User.email == email
    ).first()
    
    if not user: 
        raise cred_exc
    
    return user


@router.post("/register", response_model=Token)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """⚡ OPTIMIZADO: Registro de usuario con validación de email"""
    total_start = time.time()
    
    # Normalizar email
    normalized_email = user_in.email.strip().lower()
    print(f"[AUTH] Registrando usuario: {normalized_email}")
    
    # ⚡ VALIDACIÓN DE EMAIL (AGREGADA)
    is_valid_email, email_error = validate_email(normalized_email)
    if not is_valid_email:
        print(f"[AUTH] ❌ Email rechazado: {email_error}")
        raise HTTPException(
            status_code=400,
            detail=f"Email inválido: {email_error}"
        )
    
    # Verificar existencia
    check_start = time.time()
    existing_user = db.query(models.User).filter(
        models.User.email == normalized_email
    ).first()
    check_time = (time.time() - check_start) * 1000
    print(f"[AUTH] Verificación de email en {check_time:.0f}ms")
    
    if existing_user:
        raise HTTPException(
            status_code=400, 
            detail="Este correo ya está registrado"
        )
    
    # Hash de contraseña
    password_hash = get_password_hash(user_in.password)
    
    # Crear usuario
    user = models.User(
        email=normalized_email,
        password_hash=password_hash,
        name=user_in.name
    )
    
    # Guardar en BD
    db_start = time.time()
    db.add(user)
    db.commit()
    db.refresh(user)
    db_time = (time.time() - db_start) * 1000
    print(f"[AUTH] Guardado en BD en {db_time:.0f}ms")
    
    # Generar token
    token = create_access_token({"sub": user.email})
    
    total_time = (time.time() - total_start) * 1000
    print(f"[AUTH] ✓ Registro completo en {total_time:.0f}ms")
    
    return {"access_token": token, "token_type": "bearer"}


@router.post("/login", response_model=Token)
def login(
    form: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    """⚡ OPTIMIZADO: Inicio de sesión con validación de email"""
    total_start = time.time()
    
    # Normalizar email
    normalized_email = form.username.strip().lower()
    print(f"[AUTH] Login: {normalized_email}")
    
    # ⚡ VALIDACIÓN DE EMAIL (AGREGADA)
    is_valid_email, email_error = validate_email(normalized_email)
    if not is_valid_email:
        # Mensaje genérico por seguridad (no revelar si es email o password)
        print(f"[AUTH] ❌ Email inválido rechazado en login: {email_error}")
        raise HTTPException(
            status_code=401, 
            detail="Correo o contraseña incorrectos"
        )
    
    # Buscar usuario
    query_start = time.time()
    user = db.query(models.User).filter(
        models.User.email == normalized_email
    ).first()
    query_time = (time.time() - query_start) * 1000
    print(f"[AUTH] Query usuario en {query_time:.0f}ms")
    
    if not user:
        raise HTTPException(
            status_code=401, 
            detail="Correo o contraseña incorrectos"
        )
    
    # Verificar contraseña
    if not verify_password(form.password, user.password_hash):
        raise HTTPException(
            status_code=401, 
            detail="Correo o contraseña incorrectos"
        )
    
    # Generar token
    token = create_access_token({"sub": user.email})
    
    total_time = (time.time() - total_start) * 1000
    print(f"[AUTH] ✓ Login completo en {total_time:.0f}ms")
    
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me", response_model=UserOut)
def me(current: models.User = Depends(get_current_user)):
    """Obtiene información del usuario actual"""
    return current


# ═══════════════════════════════════════════════════════════════════════
# ⚡ ENDPOINT DE DIAGNÓSTICO
# ═══════════════════════════════════════════════════════════════════════
@router.get("/performance-test")
def performance_test():
    """Prueba velocidad de hashing para diagnóstico"""
    import time
    
    test_password = "test_password_123"
    
    # Test 1: Hash
    start = time.time()
    hash_result = pwd_ctx.hash(test_password)
    hash_time = (time.time() - start) * 1000
    
    # Test 2: Verify
    start = time.time()
    pwd_ctx.verify(test_password, hash_result)
    verify_time = (time.time() - start) * 1000
    
    return {
        "bcrypt_rounds": settings.BCRYPT_ROUNDS,
        "hash_time_ms": round(hash_time, 2),
        "verify_time_ms": round(verify_time, 2),
        "total_auth_time_ms": round(hash_time + verify_time, 2),
        "recommendation": (
            "RÁPIDO (< 100ms)" if (hash_time + verify_time) < 100
            else "ACEPTABLE (100-300ms)" if (hash_time + verify_time) < 300
            else "LENTO (> 300ms) - Considera reducir BCRYPT_ROUNDS"
        )
    }