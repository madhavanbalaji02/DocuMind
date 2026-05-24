"""SSM Parameter Store helpers for DocuMind secrets."""

from __future__ import annotations

import os
import boto3
from botocore.exceptions import ClientError

from aws.config import REGION, SSM_PREFIX


class SecretsManager:
    """Read DocuMind secrets from SSM Parameter Store."""

    def __init__(self) -> None:
        self._ssm = boto3.client("ssm", region_name=REGION)
        self._cache: dict[str, str] = {}

    def get(self, param_name: str, default: str = "") -> str:
        """Get a parameter value. Caches results in memory."""
        if param_name in self._cache:
            return self._cache[param_name]
        full_name = param_name if param_name.startswith("/") else f"{SSM_PREFIX}/{param_name}"
        try:
            resp = self._ssm.get_parameter(Name=full_name, WithDecryption=True)
            value = resp["Parameter"]["Value"]
            self._cache[param_name] = value
            return value
        except ClientError:
            return default

    def get_all(self) -> dict[str, str]:
        """Load all /documind/* parameters into env."""
        try:
            paginator = self._ssm.get_paginator("get_parameters_by_path")
            result: dict[str, str] = {}
            for page in paginator.paginate(Path=SSM_PREFIX, WithDecryption=True):
                for param in page["Parameters"]:
                    key = param["Name"].split("/")[-1].upper().replace("-", "_")
                    result[key] = param["Value"]
            return result
        except ClientError:
            return {}

    def inject_into_env(self) -> None:
        """Load all /documind/* params into os.environ (idempotent)."""
        for key, value in self.get_all().items():
            if key not in os.environ:
                os.environ[key] = value


# Singleton
_manager: SecretsManager | None = None


def get_secret(name: str, default: str = "") -> str:
    global _manager
    if _manager is None:
        _manager = SecretsManager()
    return _manager.get(name, default)
