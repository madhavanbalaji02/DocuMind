"""Lambda: S3 PUT → SQS message.

Triggered by S3 ObjectCreated events on uploads/* prefix.
Sends a message to SQS which the SQS worker picks up and starts Step Functions.
"""

import boto3
import json
import os


def lambda_handler(event, context):
    sqs = boto3.client("sqs")
    queue_url = os.environ["SQS_QUEUE_URL"]

    for record in event.get("Records", []):
        s3_key = record["s3"]["object"]["key"]
        bucket = record["s3"]["bucket"]["name"]
        size_bytes = record["s3"]["object"].get("size", 0)
        filename = s3_key.split("/")[-1]

        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        doc_type = {"pdf": "pdf", "txt": "text", "html": "web", "htm": "web"}.get(ext, "unknown")

        if doc_type == "unknown":
            print(f"Skipping non-document: {s3_key}")
            continue

        message = {
            "s3_key": s3_key,
            "filename": filename,
            "doc_type": doc_type,
            "bucket": bucket,
            "size_bytes": size_bytes,
        }

        resp = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message),
        )
        print(f"Queued: {s3_key} → MessageId={resp['MessageId']}")

    return {"statusCode": 200, "body": "OK"}
