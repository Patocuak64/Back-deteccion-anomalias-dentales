# app/cache.py
"""
 Sistema de caché en memoria para resultados de análisis
Evita re-procesar la misma imagen múltiples veces
"""

import hashlib
import time
from typing import Dict, Any, Optional
from PIL import Image
import io

from .settings import settings


class ResultCache:
    """
    Caché simple en memoria para resultados de análisis
    """
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.enabled = settings.ENABLE_RESULT_CACHE
        self.ttl = settings.CACHE_TTL_SECONDS
        print(f"[CACHE] Inicializado (enabled={self.enabled}, TTL={self.ttl}s)")
    
    def _get_image_hash(self, img: Image.Image) -> str:
        """
        Genera hash único de una imagen
        """
        # Convertir imagen a bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()
        
        # Calcular hash SHA256
        return hashlib.sha256(img_bytes).hexdigest()
    
    def get(self, img: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Obtiene resultado cacheado si existe y no ha expirado
        
        Returns:
            Dict con resultado o None si no existe/expiró
        """
        if not self.enabled:
            return None
        
        img_hash = self._get_image_hash(img)
        
        if img_hash in self.cache:
            entry = self.cache[img_hash]
            age = time.time() - entry['timestamp']
            
            if age < self.ttl:
                print(f"[CACHE] ✓ HIT (age={age:.1f}s, hash={img_hash[:8]}...)")
                return entry['result']
            else:
                # Expirado, eliminar
                print(f"[CACHE] × EXPIRED (age={age:.1f}s, hash={img_hash[:8]}...)")
                del self.cache[img_hash]
        
        print(f"[CACHE] × MISS (hash={img_hash[:8]}...)")
        return None
    
    def set(self, img: Image.Image, result: Dict[str, Any]):
        """
        Guarda resultado en caché
        """
        if not self.enabled:
            return
        
        img_hash = self._get_image_hash(img)
        
        self.cache[img_hash] = {
            'result': result,
            'timestamp': time.time()
        }
        
        print(f"[CACHE] ✓ STORED (hash={img_hash[:8]}..., entries={len(self.cache)})")
        
        # Limpieza automática si hay muchas entradas
        if len(self.cache) > 100:
            self._cleanup_old_entries()
    
    def _cleanup_old_entries(self):
        """
        Elimina entradas antiguas del caché
        """
        now = time.time()
        old_keys = [
            k for k, v in self.cache.items()
            if now - v['timestamp'] > self.ttl
        ]
        
        for key in old_keys:
            del self.cache[key]
        
        if old_keys:
            print(f"[CACHE] Limpieza: {len(old_keys)} entradas eliminadas")
    
    def clear(self):
        """
        Limpia todo el caché
        """
        count = len(self.cache)
        self.cache.clear()
        print(f"[CACHE] Cache limpiado ({count} entradas)")
    
    def stats(self) -> Dict[str, Any]:
        """
        Estadísticas del caché
        """
        now = time.time()
        ages = [now - v['timestamp'] for v in self.cache.values()]
        
        return {
            "enabled": self.enabled,
            "entries": len(self.cache),
            "ttl_seconds": self.ttl,
            "oldest_entry_age": max(ages) if ages else 0,
            "newest_entry_age": min(ages) if ages else 0,
        }


# Instancia global del caché
_result_cache = None


def get_cache() -> ResultCache:
    """
    Obtiene instancia singleton del caché
    """
    global _result_cache
    if _result_cache is None:
        _result_cache = ResultCache()
    return _result_cache
