"""Lambda: S3 PUT → SQS message.

Triggered by S3 ObjectCreated events on uploads/* prefix.
Sends a message to SQS which the SQS worker picks up and starts Step Functions.
"""

import boto3
import json
import os
import urllib.parse

EXT_TYPES = {"pdf": "pdf", "txt": "text", "html": "web", "htm": "web"}
CONTENT_TYPES = {"application/pdf": "pdf", "text/plain": "text", "text/html": "web"}


def _detect_doc_type(s3, bucket: str, s3_key: str, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext in EXT_TYPES:
        return EXT_TYPES[ext]
    # Keys like uploads/web/<id>_1706.03762 have no real extension — use Content-Type
    content_type = s3.head_object(Bucket=bucket, Key=s3_key).get("ContentType", "")
    return CONTENT_TYPES.get(content_type.split(";")[0].strip().lower(), "unknown")


def lambda_handler(event, context):
    sqs = boto3.client("sqs")
    s3 = boto3.client("s3")
    queue_url = os.environ["SQS_QUEUE_URL"]

    for record in event.get("Records", []):
        s3_key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        bucket = record["s3"]["bucket"]["name"]
        size_bytes = record["s3"]["object"].get("size", 0)
        filename = s3_key.split("/")[-1]

        doc_type = _detect_doc_type(s3, bucket, s3_key, filename)

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
