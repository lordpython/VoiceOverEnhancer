import pickle
import hashlib
from typing import Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.cache = {}
        self.ttl_map = {}

    def create_key(self, prefix: str, data: str) -> str:
        """Create a cache key with prefix"""
        hash_value = hashlib.md5(data.encode()).hexdigest()
        return f"{prefix}:{hash_value}"

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache"""
        try:
            if key not in self.cache:
                return None
            
            # Check TTL
            if key in self.ttl_map and datetime.now() > self.ttl_map[key]:
                del self.cache[key]
                del self.ttl_map[key]
                return None
                
            return self.cache[key]
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 86400):
        """Store item in cache"""
        try:
            self.cache[key] = value
            self.ttl_map[key] = datetime.now() + timedelta(seconds=ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
