"""Step Functions execution helper for DocuMind."""

from __future__ import annotations

import json
import os
import time

import boto3

from aws.config import REGION, ACCOUNT_ID, STATE_MACHINE_NAME


class PipelineExecutor:
    """Start and monitor Step Functions executions."""

    def __init__(self) -> None:
        self._sf = boto3.client("stepfunctions", region_name=REGION)
        self.state_machine_arn = os.environ.get(
            "STATE_MACHINE_ARN",
            f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{STATE_MACHINE_NAME}",
        )

    def start(self, s3_key: str, filename: str, doc_type: str,
              bucket: str, size_bytes: int = 0) -> str:
        """Start a pipeline execution. Returns execution ARN."""
        payload = {
            "s3_key": s3_key,
            "filename": filename,
            "doc_type": doc_type,
            "bucket": bucket,
            "size_bytes": size_bytes,
        }
        execution_name = f"doc-{hash(s3_key) % 1_000_000:06d}-{int(time.time())}"
        resp = self._sf.start_execution(
            stateMachineArn=self.state_machine_arn,
            name=execution_name,
            input=json.dumps(payload),
        )
        return resp["executionArn"]

    def status(self, execution_arn: str) -> dict:
        resp = self._sf.describe_execution(executionArn=execution_arn)
        return {
            "status": resp["status"],
            "start_date": resp.get("startDate", "").isoformat() if resp.get("startDate") else "",
            "stop_date": resp.get("stopDate", "").isoformat() if resp.get("stopDate") else "",
        }

    def wait_for_completion(self, execution_arn: str, timeout: int = 600) -> str:
        """Poll until execution completes. Returns final status."""
        start = time.time()
        while time.time() - start < timeout:
            info = self.status(execution_arn)
            if info["status"] in ("SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"):
                return info["status"]
            time.sleep(10)
        return "TIMEOUT"
