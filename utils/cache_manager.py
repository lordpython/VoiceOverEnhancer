import pickle
import hashlib
from typing import Optional, Any
import aioredis
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost"):
        try:
            self.redis = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self.redis = None

    def create_key(self, prefix: str, data: str) -> str:
        """Create a cache key with prefix"""
        hash_value = hashlib.md5(data.encode()).hexdigest()
        return f"{prefix}:{hash_value}"

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache"""
        if not self.redis:
            return None
        try:
            data = await self.redis.get(key)
            return pickle.loads(data.encode('latin1')) if data else None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 86400):
        """Store item in cache"""
        if not self.redis:
            return
        try:
            serialized = pickle.dumps(value).decode('latin1')
            await self.redis.set(key, serialized, ex=ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
