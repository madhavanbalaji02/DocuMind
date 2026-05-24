"""SQS ingestion worker — polls SQS and starts Step Functions executions.

Started as a daemon thread in the FastAPI lifespan.
"""

from __future__ import annotations

import json
import os
import threading
import time

import boto3

from src.core.logging import get_logger

logger = get_logger(__name__)


class SQSIngestionWorker:
    """Background worker: SQS messages → Step Functions executions."""

    def __init__(self) -> None:
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
        self._sqs = boto3.client("sqs", region_name=region)
        self._sf = boto3.client("stepfunctions", region_name=region)
        self.queue_url = os.environ.get("SQS_QUEUE_URL", "")
        self.state_machine_arn = os.environ.get("STATE_MACHINE_ARN", "")
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        """Main loop — runs forever until stop() is called."""
        if not self.queue_url or not self.state_machine_arn:
            logger.warning(
                "SQSIngestionWorker: SQS_QUEUE_URL or STATE_MACHINE_ARN not set — "
                "worker disabled. Set these env vars to enable event-driven ingestion."
            )
            return

        logger.info("SQSIngestionWorker started (queue=%s)", self.queue_url.split("/")[-1])

        while self._running:
            try:
                resp = self._sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=5,
                    WaitTimeSeconds=20,  # long polling
                    AttributeNames=["All"],
                )
                messages = resp.get("Messages", [])
                for msg in messages:
                    self._process_message(msg)

            except Exception as e:
                logger.error("SQSIngestionWorker error: %s", e)
                time.sleep(5)  # back off on error

    def _process_message(self, msg: dict) -> None:
        receipt = msg["ReceiptHandle"]
        try:
            payload = json.loads(msg["Body"])
            s3_key = payload["s3_key"]
            execution_name = f"doc-{abs(hash(s3_key)) % 1_000_000:06d}-{int(time.time())}"

            self._sf.start_execution(
                stateMachineArn=self.state_machine_arn,
                name=execution_name,
                input=json.dumps(payload),
            )
            logger.info("Started execution for %s (name=%s)", s3_key, execution_name)

            # Delete successfully processed message
            self._sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt)

        except Exception as e:
            logger.error("Failed to process SQS message: %s", e)
            # Don't delete — let SQS retry (up to 3 times, then DLQ)


def start_worker_thread() -> SQSIngestionWorker:
    """Start the SQS worker as a daemon thread. Returns the worker instance."""
    worker = SQSIngestionWorker()
    thread = threading.Thread(target=worker.run, daemon=True, name="sqs-ingestion-worker")
    thread.start()
    return worker
