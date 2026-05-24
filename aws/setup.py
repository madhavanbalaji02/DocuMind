"""DocuMind AWS resource provisioner — idempotent.

Creates all required AWS resources if they don't already exist.
Run: python aws/setup.py

Resources created:
  S3 bucket with versioning + lifecycle
  SQS queue + DLQ
  DynamoDB jobs table
  IAM execution role with least-privilege policies
  SSM parameters
  RDS db.t3.micro with pgvector (FREE TIER)
  Step Functions state machine
  AWS Glue database + crawler
  Lambda functions (s3-trigger, stage1, stage2, stage3)
  CloudWatch billing alarm at $50
"""

from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from aws.config import (
    ACCOUNT_ID, REGION, S3_BUCKET, SQS_QUEUE_NAME, SQS_DLQ_NAME,
    DYNAMODB_TABLE, STATE_MACHINE_NAME, IAM_ROLE_NAME, GLUE_DB_NAME,
    GLUE_CRAWLER_NAME, LAMBDA_S3_TRIGGER, LAMBDA_STAGE1, LAMBDA_STAGE2,
    LAMBDA_STAGE3, RDS_INSTANCE_ID, RDS_DB_NAME, RDS_USERNAME, RDS_PASSWORD,
    RDS_ENGINE, RDS_ENGINE_VERSION, RDS_INSTANCE_CLASS, SSM_ANTHROPIC_KEY, SSM_S3_BUCKET,
    SSM_SQS_URL, SSM_RDS_CONN,
)

console = Console()

# ── Clients ───────────────────────────────────────────────────────────────────
s3 = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)
iam = boto3.client("iam", region_name=REGION)
ssm = boto3.client("ssm", region_name=REGION)
rds = boto3.client("rds", region_name=REGION)
sf = boto3.client("stepfunctions", region_name=REGION)
glue = boto3.client("glue", region_name=REGION)
lmb = boto3.client("lambda", region_name=REGION)
ddb = boto3.client("dynamodb", region_name=REGION)
cw = boto3.client("cloudwatch", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)

created: list[tuple[str, str, str]] = []  # (resource, name, ARN/URL)


# ── Helpers ───────────────────────────────────────────────────────────────────

def exists_or(func):
    """Call func; on ResourceAlreadyExists / EntityAlreadyExists return None."""
    try:
        return func()
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "EntityAlreadyExists",
                    "ResourceAlreadyExistsException", "ResourceInUseException",
                    "QueueAlreadyExists", "TableAlreadyExistsException",
                    "DBInstanceAlreadyExists", "StateMachineAlreadyExists"):
            return None
        raise


def record(resource: str, name: str, arn: str) -> None:
    created.append((resource, name, arn))
    console.print(f"  [green]✓[/green] {resource}: {name}")


# ── S3 ────────────────────────────────────────────────────────────────────────

def setup_s3() -> str:
    console.rule("S3")
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        console.print(f"  [green]✓[/green] Created bucket: {S3_BUCKET}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            console.print(f"  [yellow]~[/yellow] Bucket exists: {S3_BUCKET}")
        else:
            raise

    s3.put_bucket_versioning(
        Bucket=S3_BUCKET,
        VersioningConfiguration={"Status": "Enabled"},
    )
    s3.put_bucket_lifecycle_configuration(
        Bucket=S3_BUCKET,
        LifecycleConfiguration={
            "Rules": [{
                "ID": "ia-then-delete",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Transitions": [{"Days": 30, "StorageClass": "STANDARD_IA"}],
                "Expiration": {"Days": 365},
            }]
        },
    )
    arn = f"arn:aws:s3:::{S3_BUCKET}"
    record("S3 Bucket", S3_BUCKET, arn)
    return arn


# ── SQS ───────────────────────────────────────────────────────────────────────

def setup_sqs() -> tuple[str, str]:
    console.rule("SQS")
    dlq = sqs.create_queue(
        QueueName=SQS_DLQ_NAME,
        Attributes={"MessageRetentionPeriod": "345600"},  # 4 days
    )
    dlq_url = dlq["QueueUrl"]
    dlq_attrs = sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=["QueueArn"])
    dlq_arn = dlq_attrs["Attributes"]["QueueArn"]
    record("SQS DLQ", SQS_DLQ_NAME, dlq_arn)

    q = sqs.create_queue(
        QueueName=SQS_QUEUE_NAME,
        Attributes={
            "VisibilityTimeout": "300",
            "MessageRetentionPeriod": "345600",
            "RedrivePolicy": json.dumps({
                "deadLetterTargetArn": dlq_arn,
                "maxReceiveCount": "3",
            }),
        },
    )
    queue_url = q["QueueUrl"]
    q_attrs = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])
    queue_arn = q_attrs["Attributes"]["QueueArn"]
    record("SQS Queue", SQS_QUEUE_NAME, queue_arn)
    return queue_url, queue_arn


# ── DynamoDB ──────────────────────────────────────────────────────────────────

def setup_dynamodb() -> str:
    console.rule("DynamoDB")
    try:
        resp = ddb.create_table(
            TableName=DYNAMODB_TABLE,
            KeySchema=[{"AttributeName": "s3_key", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "s3_key", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        arn = resp["TableDescription"]["TableArn"]
        # Wait for table to be ACTIVE before enabling TTL
        waiter = ddb.get_waiter("table_exists")
        waiter.wait(TableName=DYNAMODB_TABLE, WaiterConfig={"Delay": 5, "MaxAttempts": 20})
        ddb.update_time_to_live(
            TableName=DYNAMODB_TABLE,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            arn = f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/{DYNAMODB_TABLE}"
            console.print(f"  [yellow]~[/yellow] DynamoDB table exists: {DYNAMODB_TABLE}")
        else:
            raise
    record("DynamoDB", DYNAMODB_TABLE, arn)
    return arn


# ── IAM ───────────────────────────────────────────────────────────────────────

def setup_iam() -> str:
    console.rule("IAM")
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": [
                "lambda.amazonaws.com",
                "states.amazonaws.com",
                "glue.amazonaws.com",
            ]},
            "Action": "sts:AssumeRole",
        }],
    }
    try:
        resp = iam.create_role(
            RoleName=IAM_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="DocuMind execution role (least privilege)",
        )
        role_arn = resp["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            role_arn = iam.get_role(RoleName=IAM_ROLE_NAME)["Role"]["Arn"]
            console.print(f"  [yellow]~[/yellow] IAM role exists: {IAM_ROLE_NAME}")
        else:
            raise

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3DocuMind",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{S3_BUCKET}",
                    f"arn:aws:s3:::{S3_BUCKET}/*",
                ],
            },
            {
                "Sid": "Textract",
                "Effect": "Allow",
                "Action": [
                    "textract:StartDocumentAnalysis",
                    "textract:GetDocumentAnalysis",
                    "textract:StartDocumentTextDetection",
                    "textract:GetDocumentTextDetection",
                ],
                "Resource": "*",
            },
            {
                "Sid": "Comprehend",
                "Effect": "Allow",
                "Action": ["comprehend:DetectEntities", "comprehend:DetectKeyPhrases"],
                "Resource": "*",
            },
            {
                "Sid": "BedrockHaikuTitan",
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": [
                    f"arn:aws:bedrock:{REGION}::foundation-model/anthropic.claude-haiku*",
                    f"arn:aws:bedrock:{REGION}::foundation-model/amazon.titan-embed*",
                    f"arn:aws:bedrock:us.*::foundation-model/anthropic.claude-haiku*",
                ],
            },
            {
                "Sid": "SQS",
                "Effect": "Allow",
                "Action": ["sqs:SendMessage", "sqs:ReceiveMessage",
                           "sqs:DeleteMessage", "sqs:GetQueueAttributes"],
                "Resource": f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:{SQS_QUEUE_NAME}",
            },
            {
                "Sid": "StepFunctions",
                "Effect": "Allow",
                "Action": ["states:StartExecution", "states:DescribeExecution"],
                "Resource": f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{STATE_MACHINE_NAME}",
            },
            {
                "Sid": "CloudWatch",
                "Effect": "Allow",
                "Action": ["cloudwatch:PutMetricData"],
                "Resource": "*",
            },
            {
                "Sid": "SSMDocuMind",
                "Effect": "Allow",
                "Action": ["ssm:GetParameter", "ssm:GetParametersByPath"],
                "Resource": f"arn:aws:ssm:{REGION}:{ACCOUNT_ID}:parameter/documind/*",
            },
            {
                "Sid": "RDS",
                "Effect": "Allow",
                "Action": ["rds:DescribeDBInstances"],
                "Resource": "*",
            },
            {
                "Sid": "DynamoDB",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:GetItem", "dynamodb:PutItem",
                    "dynamodb:UpdateItem", "dynamodb:Query",
                ],
                "Resource": f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/{DYNAMODB_TABLE}",
            },
            {
                "Sid": "Lambda",
                "Effect": "Allow",
                "Action": ["lambda:InvokeFunction"],
                "Resource": f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:documind-*",
            },
            {
                "Sid": "Logs",
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": "arn:aws:logs:*:*:*",
            },
        ],
    }
    try:
        iam.put_role_policy(
            RoleName=IAM_ROLE_NAME,
            PolicyName="documind-inline-policy",
            PolicyDocument=json.dumps(policy),
        )
    except ClientError:
        pass

    record("IAM Role", IAM_ROLE_NAME, role_arn)
    return role_arn


# ── SSM ───────────────────────────────────────────────────────────────────────

def setup_ssm(queue_url: str) -> None:
    console.rule("SSM Parameters")

    def put(name: str, value: str, secret: bool = False) -> None:
        try:
            ssm.put_parameter(
                Name=name,
                Value=value,
                Type="SecureString" if secret else "String",
                Overwrite=True,
            )
            console.print(f"  [green]✓[/green] SSM: {name}")
        except ClientError as e:
            console.print(f"  [yellow]~[/yellow] SSM {name}: {e}")

    put(SSM_S3_BUCKET, S3_BUCKET)
    put(SSM_SQS_URL, queue_url)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "PLACEHOLDER")
    put(SSM_ANTHROPIC_KEY, anthropic_key, secret=True)
    rds_host = os.environ.get("RDS_HOST", f"documind-db.placeholder.{REGION}.rds.amazonaws.com")
    conn_str = f"postgresql://{RDS_USERNAME}:{RDS_PASSWORD}@{rds_host}:5432/{RDS_DB_NAME}"
    put(SSM_RDS_CONN, conn_str, secret=True)


# ── RDS ───────────────────────────────────────────────────────────────────────

def setup_rds() -> str:
    console.rule("RDS (pgvector)")
    console.print(
        "[yellow]  RDS db.t3.micro — FREE TIER 750 hrs/month.[/yellow]\n"
        "[yellow]  STOP when not demoing: aws rds stop-db-instance --db-instance-identifier documind-db[/yellow]"
    )
    try:
        resp = rds.create_db_instance(
            DBInstanceIdentifier=RDS_INSTANCE_ID,
            DBInstanceClass=RDS_INSTANCE_CLASS,
            Engine=RDS_ENGINE,
            EngineVersion=RDS_ENGINE_VERSION,
            MasterUsername=RDS_USERNAME,
            MasterUserPassword=RDS_PASSWORD,
            DBName=RDS_DB_NAME,
            AllocatedStorage=20,
            StorageType="gp2",
            PubliclyAccessible=True,
            BackupRetentionPeriod=1,
            MultiAZ=False,
            Tags=[{"Key": "Project", "Value": "DocuMind"}],
        )
        db_arn = resp["DBInstance"]["DBInstanceArn"]
        console.print("  [green]✓[/green] RDS instance creating (takes ~5 min) …")
    except ClientError as e:
        if e.response["Error"]["Code"] == "DBInstanceAlreadyExists":
            info = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE_ID)
            db_arn = info["DBInstances"][0]["DBInstanceArn"]
            console.print(f"  [yellow]~[/yellow] RDS instance exists: {RDS_INSTANCE_ID}")
        else:
            raise
    record("RDS PostgreSQL", RDS_INSTANCE_ID, db_arn)
    return db_arn


# ── Glue ──────────────────────────────────────────────────────────────────────

def setup_glue(role_arn: str) -> None:
    console.rule("AWS Glue")
    try:
        glue.create_database(DatabaseInput={"Name": GLUE_DB_NAME})
        console.print(f"  [green]✓[/green] Glue DB: {GLUE_DB_NAME}")
    except ClientError as e:
        if "AlreadyExistsException" in str(e):
            console.print(f"  [yellow]~[/yellow] Glue DB exists: {GLUE_DB_NAME}")
        else:
            raise
    try:
        glue.create_crawler(
            Name=GLUE_CRAWLER_NAME,
            Role=role_arn,
            DatabaseName=GLUE_DB_NAME,
            Targets={"S3Targets": [{"Path": f"s3://{S3_BUCKET}/uploads/"}]},
            SchemaChangePolicy={"UpdateBehavior": "UPDATE_IN_DATABASE",
                                "DeleteBehavior": "DEPRECATE_IN_DATABASE"},
        )
        console.print(f"  [green]✓[/green] Glue Crawler: {GLUE_CRAWLER_NAME}")
    except ClientError as e:
        if "AlreadyExistsException" in str(e):
            console.print(f"  [yellow]~[/yellow] Glue Crawler exists: {GLUE_CRAWLER_NAME}")
        else:
            raise
    record("Glue Database", GLUE_DB_NAME, f"arn:aws:glue:{REGION}:{ACCOUNT_ID}:database/{GLUE_DB_NAME}")


# ── Lambda ────────────────────────────────────────────────────────────────────

def _make_zip(source_file: str) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_file, os.path.basename(source_file))
    return buf.getvalue()


def _create_or_update_lambda(
    name: str,
    handler: str,
    zip_bytes: bytes,
    role_arn: str,
    env_vars: dict[str, str],
    timeout: int = 300,
    memory: int = 512,
) -> str:
    try:
        resp = lmb.create_function(
            FunctionName=name,
            Runtime="python3.12",
            Role=role_arn,
            Handler=handler,
            Code={"ZipFile": zip_bytes},
            Timeout=timeout,
            MemorySize=memory,
            Environment={"Variables": env_vars},
        )
        arn = resp["FunctionArn"]
        console.print(f"  [green]✓[/green] Lambda created: {name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            lmb.update_function_code(FunctionName=name, ZipFile=zip_bytes)
            lmb.update_function_configuration(
                FunctionName=name, Timeout=timeout, MemorySize=memory,
                Environment={"Variables": env_vars},
            )
            arn = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{name}"
            console.print(f"  [yellow]~[/yellow] Lambda updated: {name}")
        else:
            raise
    return arn


def setup_lambdas(role_arn: str, queue_url: str) -> dict[str, str]:
    console.rule("Lambda Functions")
    project_root = Path(__file__).parent.parent
    lambda_dir = project_root / "aws" / "lambda_functions"

    env_base = {
        "S3_BUCKET": S3_BUCKET,
        "SQS_QUEUE_URL": queue_url,
        "AWS_ACCOUNT_ID": ACCOUNT_ID,
        "REGION": REGION,
        "DYNAMODB_TABLE": DYNAMODB_TABLE,
        "EMBEDDING_MODE": "local",
    }

    arns: dict[str, str] = {}
    for name, src_file, handler, timeout, mem in [
        (LAMBDA_S3_TRIGGER, "s3_trigger.py", "s3_trigger.lambda_handler", 30, 128),
        (LAMBDA_STAGE1, "stage1_parse.py", "stage1_parse.lambda_handler", 600, 512),
        (LAMBDA_STAGE2, "stage2_embed.py", "stage2_embed.lambda_handler", 600, 1024),
        (LAMBDA_STAGE3, "stage3_store.py", "stage3_store.lambda_handler", 300, 512),
    ]:
        src = lambda_dir / src_file
        if not src.exists():
            console.print(f"  [yellow]![/yellow] Lambda source missing: {src} — skipping")
            arns[name] = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{name}"
            continue
        zip_bytes = _make_zip(str(src))
        arn = _create_or_update_lambda(name, handler, zip_bytes, role_arn, env_base, timeout, mem)
        arns[name] = arn
        record("Lambda", name, arn)

    # Wire S3 event notification → Lambda s3_trigger
    try:
        trigger_arn = arns.get(LAMBDA_S3_TRIGGER, "")
        if trigger_arn:
            lmb.add_permission(
                FunctionName=LAMBDA_S3_TRIGGER,
                StatementId="s3-invoke",
                Action="lambda:InvokeFunction",
                Principal="s3.amazonaws.com",
                SourceArn=f"arn:aws:s3:::{S3_BUCKET}",
            )
            s3.put_bucket_notification_configuration(
                Bucket=S3_BUCKET,
                NotificationConfiguration={
                    "LambdaFunctionConfigurations": [{
                        "LambdaFunctionArn": trigger_arn,
                        "Events": ["s3:ObjectCreated:*"],
                        "Filter": {"Key": {"FilterRules": [
                            {"Name": "prefix", "Value": "uploads/"},
                        ]}},
                    }]
                },
            )
            console.print(f"  [green]✓[/green] S3 → Lambda notification wired")
    except ClientError as e:
        console.print(f"  [yellow]~[/yellow] S3 notification: {e.response['Error']['Code']}")

    return arns


# ── Step Functions ─────────────────────────────────────────────────────────────

def setup_step_functions(role_arn: str, lambda_arns: dict[str, str]) -> str:
    console.rule("Step Functions")
    project_root = Path(__file__).parent.parent
    pipeline_json = project_root / "aws" / "step_functions" / "pipeline.json"

    definition = pipeline_json.read_text() if pipeline_json.exists() else json.dumps({
        "Comment": "DocuMind 3-stage ingestion pipeline",
        "StartAt": "Stage1_Parse",
        "States": {
            "Stage1_Parse": {"Type": "Task",
                              "Resource": "arn:aws:states:::lambda:invoke",
                              "Parameters": {"FunctionName": lambda_arns.get(LAMBDA_STAGE1, ""),
                                             "Payload.$": "$"},
                              "ResultPath": "$.parse_result",
                              "Next": "Stage2_Embed"},
            "Stage2_Embed": {"Type": "Task",
                              "Resource": "arn:aws:states:::lambda:invoke",
                              "Parameters": {"FunctionName": lambda_arns.get(LAMBDA_STAGE2, ""),
                                             "Payload.$": "$"},
                              "ResultPath": "$.embed_result",
                              "Next": "Stage3_Store"},
            "Stage3_Store": {"Type": "Task",
                              "Resource": "arn:aws:states:::lambda:invoke",
                              "Parameters": {"FunctionName": lambda_arns.get(LAMBDA_STAGE3, ""),
                                             "Payload.$": "$"},
                              "ResultPath": "$.store_result",
                              "End": True},
        },
    })

    try:
        resp = sf.create_state_machine(
            name=STATE_MACHINE_NAME,
            definition=definition,
            roleArn=role_arn,
            type="STANDARD",
        )
        sm_arn = resp["stateMachineArn"]
        console.print(f"  [green]✓[/green] State machine created: {STATE_MACHINE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "StateMachineAlreadyExists":
            sm_arn = f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{STATE_MACHINE_NAME}"
            try:
                sf.update_state_machine(stateMachineArn=sm_arn, definition=definition)
            except ClientError:
                pass
            console.print(f"  [yellow]~[/yellow] State machine exists: {STATE_MACHINE_NAME}")
        else:
            raise

    record("Step Functions", STATE_MACHINE_NAME, sm_arn)
    return sm_arn


# ── CloudWatch billing alarm ───────────────────────────────────────────────────

def setup_billing_alarm() -> None:
    console.rule("CloudWatch Billing Alarm")
    # BUDGET ALERT at $50 — well below $200 credit
    try:
        cw_billing = boto3.client("cloudwatch", region_name="us-east-1")  # billing metrics are us-east-1
        cw_billing.put_metric_alarm(
            AlarmName="documind-billing-warning",
            MetricName="EstimatedCharges",
            Namespace="AWS/Billing",
            Statistic="Maximum",
            Dimensions=[{"Name": "Currency", "Value": "USD"}],
            Period=86400,
            EvaluationPeriods=1,
            Threshold=50.0,
            ComparisonOperator="GreaterThanThreshold",
            AlarmDescription="DocuMind billing exceeds $50 — investigate immediately",
            TreatMissingData="notBreaching",
        )
        console.print("  [green]✓[/green] Billing alarm at $50")
    except ClientError as e:
        console.print(f"  [yellow]~[/yellow] Billing alarm: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print("\n[bold cyan]DocuMind AWS Setup[/bold cyan]\n")

    s3_arn = setup_s3()
    queue_url, queue_arn = setup_sqs()
    ddb_arn = setup_dynamodb()
    role_arn = setup_iam()
    setup_ssm(queue_url)
    rds_arn = setup_rds()
    setup_glue(role_arn)
    lambda_arns = setup_lambdas(role_arn, queue_url)
    sm_arn = setup_step_functions(role_arn, lambda_arns)
    setup_billing_alarm()

    # Print summary table
    console.print("\n")
    table = Table(title="DocuMind AWS Resources", show_lines=True)
    table.add_column("Resource", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("ARN / URL", style="dim")
    for resource, name, arn in created:
        table.add_row(resource, name, arn[:80])
    console.print(table)

    console.print(f"\n[bold green]Setup complete.[/bold green]")
    console.print(f"  State machine: {sm_arn}")
    console.print(f"  SQS queue:     {queue_url}")
    console.print(f"  S3 bucket:     s3://{S3_BUCKET}")
    console.print(f"\n[yellow]Next: run 'python aws/pgvector/migrate.py' after RDS is ready.[/yellow]")


if __name__ == "__main__":
    main()
