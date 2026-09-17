"""API key authentication."""

import json
from typing import Optional
from fastapi import Header, HTTPException, Depends
from config.settings import config
from config.logging_config import logger


def _load_keys() -> dict:
    path = config.auth.api_keys_file
    if not path.exists():
        logger.warning(f"API keys file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("keys", {})


def get_current_user(x_api_key: Optional[str] = Header(None)) -> dict:
    """Validate X-API-Key header. Returns user info or raises 401."""
    if not config.auth.enabled:
        return {"user": "anonymous", "role": "admin"}

    if not x_api_key:
        raise HTTPException(401, "Missing X-API-Key header")

    keys = _load_keys()
    if x_api_key not in keys:
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(401, "Invalid API key")

    return keys[x_api_key]


def require_role(*allowed_roles: str):
    """Dependency factory: require one of the given roles."""
    def checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(
                403,
                f"Role '{user['role']}' not permitted. Required: {allowed_roles}",
            )
        return user
    return checker