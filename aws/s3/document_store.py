"""S3 document store — upload, download, and manage document objects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError

from aws.config import S3_BUCKET, REGION


class DocumentStore:
    """Thin wrapper around S3 for DocuMind document storage."""

    def __init__(self) -> None:
        self.bucket = os.environ.get("S3_BUCKET", S3_BUCKET)
        self._s3 = boto3.client("s3", region_name=REGION)

    def upload_file(self, file_path: str | Path, s3_key: str) -> str:
        """Upload a local file to S3. Returns the S3 key."""
        self._s3.upload_file(str(file_path), self.bucket, s3_key)
        return s3_key

    def upload_bytes(self, data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> str:
        self._s3.put_object(Bucket=self.bucket, Key=s3_key, Body=data, ContentType=content_type)
        return s3_key

    def upload_fileobj(self, fileobj: BinaryIO, s3_key: str) -> str:
        self._s3.upload_fileobj(fileobj, self.bucket, s3_key)
        return s3_key

    def download_bytes(self, s3_key: str) -> bytes:
        response = self._s3.get_object(Bucket=self.bucket, Key=s3_key)
        return response["Body"].read()

    def download_to_file(self, s3_key: str, local_path: str | Path) -> None:
        self._s3.download_file(self.bucket, s3_key, str(local_path))

    def put_json(self, data: dict, s3_key: str) -> str:
        import json
        body = json.dumps(data, default=str).encode()
        return self.upload_bytes(body, s3_key, content_type="application/json")

    def get_json(self, s3_key: str) -> dict:
        import json
        return json.loads(self.download_bytes(s3_key))

    def exists(self, s3_key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def list_prefix(self, prefix: str) -> list[str]:
        paginator = self._s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def presigned_url(self, s3_key: str, expires_in: int = 3600) -> str:
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": s3_key},
            ExpiresIn=expires_in,
        )
