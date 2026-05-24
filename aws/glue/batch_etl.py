"""AWS Glue batch ETL for bulk document processing (>10 docs).

Single documents use the Lambda/Step Functions pipeline (cheaper).
This Glue job is for batch corpus processing: 100+ documents at once.

COST: G.1X = $0.44/DPU-hour. 2 DPU × 1 hour = $0.88 per batch run.
Free tier: 1M DPU-seconds = ~277 DPU-hours. Covers many batch runs.

Usage: python aws/glue/batch_etl.py --start (triggers a Glue job run)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from aws.config import (
    REGION, ACCOUNT_ID, S3_BUCKET, IAM_ROLE_NAME,
    GLUE_DB_NAME, GLUE_CRAWLER_NAME,
)

GLUE_JOB_NAME = "documind-batch-processor"
GLUE_SCRIPT_KEY = "glue/etl_script.py"

ETL_SCRIPT = '''
"""DocuMind Glue ETL Script — runs in Glue spark context."""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import boto3
import json
import hashlib

args = getResolvedOptions(sys.argv, ["JOB_NAME", "S3_BUCKET"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

s3 = boto3.client("s3")
bucket = args["S3_BUCKET"]

# Find all unprocessed documents in uploads/
paginator = s3.get_paginator("list_objects_v2")
documents = []
for page in paginator.paginate(Bucket=bucket, Prefix="uploads/"):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        # Check if already in parsed/
        parsed_key = f"parsed/{key}.json"
        try:
            s3.head_object(Bucket=bucket, Key=parsed_key)
            # Already processed
        except:
            documents.append({
                "s3_key": key,
                "filename": key.split("/")[-1],
                "size": obj["Size"],
            })

print(f"Found {len(documents)} unprocessed documents")

# Dedup by content hash
seen_hashes = set()
unique_docs = []
for doc in documents:
    try:
        obj = s3.get_object(Bucket=bucket, Key=doc["s3_key"])
        content = obj["Body"].read(4096)  # first 4KB for hash
        h = hashlib.sha256(content).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)
    except Exception as e:
        print(f"Error hashing {doc['s3_key']}: {e}")

print(f"Unique documents after dedup: {len(unique_docs)}")

# Write job manifest to S3 for Step Functions to pick up
manifest = {"documents": unique_docs, "total": len(unique_docs)}
s3.put_object(
    Bucket=bucket,
    Key="glue/batch_manifest.json",
    Body=json.dumps(manifest).encode(),
    ContentType="application/json",
)
print(f"Manifest written to s3://{bucket}/glue/batch_manifest.json")

job.commit()
'''


class GlueBatchProcessor:
    """Trigger and monitor Glue ETL jobs for bulk document processing."""

    MIN_DOCS_FOR_GLUE = 10  # use Lambda pipeline for fewer docs

    def __init__(self) -> None:
        self._glue = boto3.client("glue", region_name=REGION)
        self._s3 = boto3.client("s3", region_name=REGION)
        self._role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{IAM_ROLE_NAME}"

    def ensure_job_exists(self) -> None:
        """Create or update the Glue ETL job."""
        # Upload ETL script to S3
        try:
            self._s3.put_object(
                Bucket=S3_BUCKET,
                Key=GLUE_SCRIPT_KEY,
                Body=ETL_SCRIPT.encode(),
                ContentType="text/x-python",
            )
        except Exception as e:
            print(f"Warning: could not upload Glue script: {e}")

        script_location = f"s3://{S3_BUCKET}/{GLUE_SCRIPT_KEY}"
        job_args = {
            "Name": GLUE_JOB_NAME,
            "Role": self._role_arn,
            "Command": {
                "Name": "glueetl",
                "ScriptLocation": script_location,
                "PythonVersion": "3",
            },
            "DefaultArguments": {
                "--S3_BUCKET": S3_BUCKET,
                "--job-language": "python",
            },
            "GlueVersion": "4.0",
            "WorkerType": "G.1X",    # cheapest Glue worker
            "NumberOfWorkers": 2,
            "Timeout": 60,
        }

        try:
            self._glue.create_job(**job_args)
            print(f"Created Glue job: {GLUE_JOB_NAME}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "AlreadyExistsException":
                # Update existing
                update_args = {k: v for k, v in job_args.items() if k != "Name"}
                self._glue.update_job(JobName=GLUE_JOB_NAME, JobUpdate=update_args)
                print(f"Updated Glue job: {GLUE_JOB_NAME}")
            else:
                raise

    def should_use_glue(self, queue_depth: int) -> bool:
        return queue_depth >= self.MIN_DOCS_FOR_GLUE

    def start_run(self) -> str:
        """Start a Glue job run. Returns the job run ID."""
        resp = self._glue.start_job_run(
            JobName=GLUE_JOB_NAME,
            Arguments={"--S3_BUCKET": S3_BUCKET},
        )
        run_id = resp["JobRunId"]
        print(f"Glue job started: {run_id}")
        return run_id

    def get_status(self, run_id: str) -> str:
        resp = self._glue.get_job_run(JobName=GLUE_JOB_NAME, RunId=run_id)
        return resp["JobRun"]["JobRunState"]

    def wait_for_completion(self, run_id: str, timeout: int = 3600) -> str:
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status(run_id)
            if status in ("SUCCEEDED", "FAILED", "TIMEOUT", "STOPPED"):
                return status
            print(f"Glue job {run_id}: {status}")
            time.sleep(30)
        return "TIMEOUT"


if __name__ == "__main__":
    processor = GlueBatchProcessor()
    processor.ensure_job_exists()
    run_id = processor.start_run()
    print(f"Glue run: {run_id}")
