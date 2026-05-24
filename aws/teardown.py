"""DocuMind AWS teardown — destroys all resources to prevent surprise bills.

Run: python aws/teardown.py

WARNING: This is destructive and irreversible. All data will be lost.
Prompts for confirmation before proceeding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from rich.console import Console
from rich.prompt import Confirm

sys.path.insert(0, str(Path(__file__).parent.parent))
from aws.config import (
    ACCOUNT_ID, REGION, S3_BUCKET, SQS_QUEUE_NAME, SQS_DLQ_NAME,
    DYNAMODB_TABLE, STATE_MACHINE_NAME, IAM_ROLE_NAME,
    LAMBDA_S3_TRIGGER, LAMBDA_STAGE1, LAMBDA_STAGE2, LAMBDA_STAGE3,
    RDS_INSTANCE_ID, GLUE_DB_NAME, GLUE_CRAWLER_NAME,
)

console = Console()

s3 = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)
iam = boto3.client("iam", region_name=REGION)
rds = boto3.client("rds", region_name=REGION)
sf = boto3.client("stepfunctions", region_name=REGION)
glue = boto3.client("glue", region_name=REGION)
lmb = boto3.client("lambda", region_name=REGION)
ddb = boto3.client("dynamodb", region_name=REGION)


def safe(func):
    try:
        func()
    except ClientError as e:
        console.print(f"  [yellow]skip[/yellow] {e.response['Error']['Code']}")
    except Exception as e:
        console.print(f"  [yellow]skip[/yellow] {e}")


def delete_s3() -> None:
    console.rule("S3")
    # Empty bucket first
    paginator = s3.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=S3_BUCKET):
        objects = []
        for v in page.get("Versions", []):
            objects.append({"Key": v["Key"], "VersionId": v["VersionId"]})
        for m in page.get("DeleteMarkers", []):
            objects.append({"Key": m["Key"], "VersionId": m["VersionId"]})
        if objects:
            s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": objects})
    safe(lambda: s3.delete_bucket(Bucket=S3_BUCKET))
    console.print(f"  [red]✗[/red] S3 bucket deleted: {S3_BUCKET}")


def delete_sqs() -> None:
    console.rule("SQS")
    for name in [SQS_QUEUE_NAME, SQS_DLQ_NAME]:
        try:
            url = sqs.get_queue_url(QueueName=name)["QueueUrl"]
            sqs.delete_queue(QueueUrl=url)
            console.print(f"  [red]✗[/red] SQS deleted: {name}")
        except ClientError:
            pass


def delete_lambdas() -> None:
    console.rule("Lambda")
    for name in [LAMBDA_S3_TRIGGER, LAMBDA_STAGE1, LAMBDA_STAGE2, LAMBDA_STAGE3]:
        safe(lambda n=name: lmb.delete_function(FunctionName=n))
        console.print(f"  [red]✗[/red] Lambda deleted: {name}")


def delete_step_functions() -> None:
    console.rule("Step Functions")
    sm_arn = f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{STATE_MACHINE_NAME}"
    safe(lambda: sf.delete_state_machine(stateMachineArn=sm_arn))
    console.print(f"  [red]✗[/red] State machine deleted: {STATE_MACHINE_NAME}")


def delete_dynamodb() -> None:
    console.rule("DynamoDB")
    safe(lambda: ddb.delete_table(TableName=DYNAMODB_TABLE))
    console.print(f"  [red]✗[/red] DynamoDB table deleted: {DYNAMODB_TABLE}")


def delete_rds() -> None:
    console.rule("RDS")
    safe(lambda: rds.delete_db_instance(
        DBInstanceIdentifier=RDS_INSTANCE_ID,
        SkipFinalSnapshot=True,
        DeleteAutomatedBackups=True,
    ))
    console.print(f"  [red]✗[/red] RDS instance deleting: {RDS_INSTANCE_ID}")


def delete_iam() -> None:
    console.rule("IAM")
    safe(lambda: iam.delete_role_policy(
        RoleName=IAM_ROLE_NAME, PolicyName="documind-inline-policy"
    ))
    safe(lambda: iam.delete_role(RoleName=IAM_ROLE_NAME))
    console.print(f"  [red]✗[/red] IAM role deleted: {IAM_ROLE_NAME}")


def delete_glue() -> None:
    console.rule("Glue")
    safe(lambda: glue.delete_crawler(Name=GLUE_CRAWLER_NAME))
    safe(lambda: glue.delete_database(Name=GLUE_DB_NAME))
    console.print(f"  [red]✗[/red] Glue resources deleted")


def main() -> None:
    console.print("\n[bold red]DocuMind AWS Teardown[/bold red]")
    console.print("[red]WARNING: This will delete ALL DocuMind AWS resources.[/red]\n")

    if not Confirm.ask("Are you sure you want to destroy all DocuMind AWS resources?"):
        console.print("Aborted.")
        return

    delete_lambdas()
    delete_step_functions()
    delete_sqs()
    delete_dynamodb()
    delete_s3()
    delete_rds()
    delete_glue()
    delete_iam()

    console.print("\n[bold red]Teardown complete. All DocuMind resources deleted.[/bold red]")


if __name__ == "__main__":
    main()
