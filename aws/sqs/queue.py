"""SQS queue helper for DocuMind ingestion messages."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator

import boto3

from aws.config import REGION, SQS_QUEUE_NAME, ACCOUNT_ID


@dataclass
class IngestionMessage:
    s3_key: str
    filename: str
    doc_type: str
    bucket: str
    size_bytes: int
    receipt_handle: str = ""


class IngestionQueue:
    """Send and receive DocuMind ingestion messages from SQS."""

    def __init__(self) -> None:
        self._sqs = boto3.client("sqs", region_name=REGION)
        self.queue_url = os.environ.get(
            "SQS_QUEUE_URL",
            f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT_ID}/{SQS_QUEUE_NAME}",
        )

    def send(self, s3_key: str, filename: str, doc_type: str,
             bucket: str, size_bytes: int) -> str:
        """Enqueue a document for ingestion. Returns message ID."""
        body = json.dumps({
            "s3_key": s3_key,
            "filename": filename,
            "doc_type": doc_type,
            "bucket": bucket,
            "size_bytes": size_bytes,
        })
        resp = self._sqs.send_message(QueueUrl=self.queue_url, MessageBody=body)
        return resp["MessageId"]

    def receive(self, max_messages: int = 5, wait_seconds: int = 20) -> list[IngestionMessage]:
        resp = self._sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_seconds,
            AttributeNames=["All"],
        )
        messages: list[IngestionMessage] = []
        for msg in resp.get("Messages", []):
            body = json.loads(msg["Body"])
            messages.append(IngestionMessage(
                s3_key=body["s3_key"],
                filename=body["filename"],
                doc_type=body["doc_type"],
                bucket=body["bucket"],
                size_bytes=body.get("size_bytes", 0),
                receipt_handle=msg["ReceiptHandle"],
            ))
        return messages

    def delete(self, receipt_handle: str) -> None:
        self._sqs.delete_message(
            QueueUrl=self.queue_url, ReceiptHandle=receipt_handle
        )

    def approximate_count(self) -> int:
        resp = self._sqs.get_queue_attributes(
            QueueUrl=self.queue_url,
            AttributeNames=["ApproximateNumberOfMessages"],
        )
        return int(resp["Attributes"].get("ApproximateNumberOfMessages", 0))
