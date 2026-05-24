"""Central AWS configuration for DocuMind.

All resource names and IDs are derived from account_id to keep them unique
and idempotent across multiple setup runs.
"""

from __future__ import annotations

import os

import boto3

# ── Account info ──────────────────────────────────────────────────────────────
ACCOUNT_ID: str = os.environ.get("AWS_ACCOUNT_ID", "574772738370")
REGION: str = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

# ── Resource names ─────────────────────────────────────────────────────────────
S3_BUCKET: str = f"documind-documents-{ACCOUNT_ID}"
SQS_QUEUE_NAME: str = "documind-ingestion-queue"
SQS_DLQ_NAME: str = "documind-ingestion-dlq"
DYNAMODB_TABLE: str = "documind-jobs"
STATE_MACHINE_NAME: str = "documind-ingestion-pipeline"
GLUE_DB_NAME: str = "documind_catalog"
GLUE_CRAWLER_NAME: str = "documind-s3-crawler"
IAM_ROLE_NAME: str = "documind-execution-role"

# Lambda function names (one per Step Functions stage)
LAMBDA_S3_TRIGGER: str = "documind-s3-trigger"
LAMBDA_STAGE1: str = "documind-stage1-parse"
LAMBDA_STAGE2: str = "documind-stage2-embed"
LAMBDA_STAGE3: str = "documind-stage3-store"

# SSM parameter paths
SSM_PREFIX: str = "/documind"
SSM_ANTHROPIC_KEY: str = f"{SSM_PREFIX}/anthropic-api-key"
SSM_RDS_CONN: str = f"{SSM_PREFIX}/rds-connection-string"
SSM_S3_BUCKET: str = f"{SSM_PREFIX}/s3-bucket-name"
SSM_SQS_URL: str = f"{SSM_PREFIX}/sqs-queue-url"

# ── RDS (pgvector) ─────────────────────────────────────────────────────────────
# NOTE: db.t3.micro free tier — SEPARATE instance from other projects.
# Free tier = 750 hrs/month. Stop when not demoing to save hours.
# aws rds stop-db-instance --db-instance-identifier documind-db
RDS_INSTANCE_ID: str = "documind-db"
RDS_ENGINE: str = "postgres"
RDS_ENGINE_VERSION: str = "15.18"
RDS_INSTANCE_CLASS: str = "db.t3.micro"  # FREE TIER
RDS_DB_NAME: str = os.environ.get("RDS_DB", "documind")
RDS_USERNAME: str = os.environ.get("RDS_USER", "documindadmin")
RDS_PASSWORD: str = os.environ.get("RDS_PASSWORD") or "changeme_in_production"

# ── Cost constants ─────────────────────────────────────────────────────────────
# TEXTRACT: AnalyzeDocument $0.015/page, DetectDocumentText $0.0015/page
# COMPREHEND: $0.0001/unit (100 chars)
# BEDROCK Haiku: $0.00025/1K in, $0.00125/1K out
TEXTRACT_MAX_PAGES: int = 50  # reject documents larger than this

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client(service: str, **kwargs):
    return boto3.client(service, region_name=REGION, **kwargs)


def get_resource(service: str, **kwargs):
    return boto3.resource(service, region_name=REGION, **kwargs)
