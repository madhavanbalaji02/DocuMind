"""Lambda Stage 3: Comprehend entity extraction + store the chunk index.

Input:  {s3_key, embed_result_key, chunk_count, ...}
Output: {s3_key, status: "ready", total_chunks, indexed_key, ...}

The app runs on SQLite (RDS was removed), so the final index lives in S3:
  indexed/{s3_key}.json  — document metadata, entities, parent/child chunks + embeddings
and document metadata is written to the DynamoDB jobs table (keyed by s3_key).
MarkReady in the state machine then sets status = "ready".

Also pushes CloudWatch metrics:
  DocuMind/DocumentsProcessed += 1
  DocuMind/ChunksIndexed += chunk_count
  DocuMind/ProcessingTime = elapsed_ms
"""

import boto3
import json
import os
import time
from datetime import datetime, timezone


def lambda_handler(event, context):
    s3_key = event["s3_key"]
    embed_result_key = event["embed_result_key"]
    bucket = event.get("bucket", os.environ.get("S3_BUCKET", ""))
    table_name = os.environ.get("DYNAMODB_TABLE", "documind-jobs")
    t0 = time.time()

    print(f"Stage3: storing {s3_key}")

    s3 = boto3.client("s3")
    ddb = boto3.client("dynamodb")
    cw = boto3.client("cloudwatch")

    # Load embeddings from S3
    resp = s3.get_object(Bucket=bucket, Key=embed_result_key)
    embed_result = json.loads(resp["Body"].read())
    pairs = embed_result.get("pairs", [])

    # Run Comprehend entity extraction on first 5000 chars
    full_text = " ".join(
        child.get("content", "")
        for pair in pairs[:5]
        for child in pair.get("children", [])
    )[:5000]
    entities_meta = {}
    if os.environ.get("COMPREHEND_ENABLED", "true").lower() == "true" and full_text:
        entities_meta = _run_comprehend(full_text)

    total_chunks = sum(
        1 for pair in pairs for child in pair.get("children", []) if child.get("embedding")
    )
    filename = event.get("filename") or s3_key.split("/")[-1]
    processed_at = datetime.now(timezone.utc).isoformat()

    index = {
        "s3_key": s3_key,
        "filename": filename,
        "doc_type": event.get("doc_type", "pdf"),
        "page_count": event.get("page_count", 0),
        "embed_mode": embed_result.get("embed_mode"),
        "embed_dim": embed_result.get("embed_dim"),
        "entities": entities_meta,
        "parent_count": len(pairs),
        "chunk_count": total_chunks,
        "processed_at": processed_at,
        "pairs": pairs,
    }
    indexed_key = f"indexed/{s3_key}.json"
    s3.put_object(
        Bucket=bucket,
        Key=indexed_key,
        Body=json.dumps(index, default=str).encode(),
        ContentType="application/json",
    )

    # Metadata only — chunk bodies/embeddings would blow the 400 KB item limit
    ddb.update_item(
        TableName=table_name,
        Key={"s3_key": {"S": s3_key}},
        UpdateExpression=(
            "SET filename = :fn, doc_type = :dt, page_count = :pc, parent_count = :pp, "
            "indexed_key = :ik, entities = :en, processed_at = :ts"
        ),
        ExpressionAttributeValues={
            ":fn": {"S": filename},
            ":dt": {"S": index["doc_type"]},
            ":pc": {"N": str(index["page_count"] or 0)},
            ":pp": {"N": str(len(pairs))},
            ":ik": {"S": indexed_key},
            ":en": {"S": json.dumps(entities_meta)},
            ":ts": {"S": processed_at},
        },
    )
    print(f"Stored {total_chunks} chunks for {s3_key} → s3://{bucket}/{indexed_key}")

    elapsed_ms = int((time.time() - t0) * 1000)

    # CloudWatch metrics
    try:
        cw.put_metric_data(
            Namespace="DocuMind",
            MetricData=[
                {"MetricName": "DocumentsProcessed", "Value": 1, "Unit": "Count"},
                {"MetricName": "ChunksIndexed", "Value": total_chunks, "Unit": "Count"},
                {"MetricName": "ProcessingTime", "Value": elapsed_ms, "Unit": "Milliseconds"},
            ],
        )
    except Exception as e:
        print(f"CloudWatch metrics error (non-fatal): {e}")

    return {
        **event,
        "status": "ready",
        "total_chunks": total_chunks,
        "indexed_key": indexed_key,
        "elapsed_ms": elapsed_ms,
    }


def _run_comprehend(text: str) -> dict:
    comprehend = boto3.client("comprehend")
    result = {"entities": [], "key_phrases": []}
    try:
        ents = comprehend.detect_entities(Text=text[:5000], LanguageCode="en")
        result["entities"] = [
            {"text": e["Text"], "type": e["Type"], "confidence": round(e["Score"], 3)}
            for e in ents.get("Entities", [])
            if e["Score"] >= 0.8
        ]
    except Exception as e:
        print(f"Comprehend entities error: {e}")
    try:
        kps = comprehend.detect_key_phrases(Text=text[:5000], LanguageCode="en")
        result["key_phrases"] = [
            {"text": kp["Text"], "confidence": round(kp["Score"], 3)}
            for kp in kps.get("KeyPhrases", [])
            if kp["Score"] >= 0.8
        ][:20]
    except Exception as e:
        print(f"Comprehend key phrases error: {e}")
    return result
