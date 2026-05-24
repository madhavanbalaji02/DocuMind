"""Lambda Stage 3: Store embeddings to pgvector + Comprehend entity extraction.

Input:  {s3_key, embed_result_key, chunk_count, ...}
Output: {s3_key, status: "ready", total_chunks, ...}

Also pushes CloudWatch metrics:
  DocuMind/DocumentsProcessed += 1
  DocuMind/ChunksIndexed += chunk_count
  DocuMind/ProcessingTime = elapsed_ms
"""

import boto3
import json
import os
import time


def lambda_handler(event, context):
    s3_key = event["s3_key"]
    embed_result_key = event["embed_result_key"]
    bucket = event.get("bucket", os.environ.get("S3_BUCKET", ""))
    t0 = time.time()

    print(f"Stage3: storing {s3_key}")

    s3 = boto3.client("s3")
    cw = boto3.client("cloudwatch")

    # Load embeddings from S3
    resp = s3.get_object(Bucket=bucket, Key=embed_result_key)
    embed_result = json.loads(resp["Body"].read())
    pairs = embed_result.get("pairs", [])

    # Connect to pgvector
    conn = _get_db_conn()
    total_chunks = 0
    try:
        cursor = conn.cursor()

        # Get or create document record
        filename = s3_key.split("/")[-1]
        cursor.execute("""
            INSERT INTO documents (s3_key, filename, doc_type, page_count, status)
            VALUES (%s, %s, %s, %s, 'processing')
            ON CONFLICT (s3_key) DO UPDATE SET status = 'processing'
            RETURNING id::text
        """, (s3_key, filename, event.get("doc_type", "pdf"),
              event.get("page_count", 0)))
        doc_id = cursor.fetchone()[0]
        conn.commit()

        # Run Comprehend entity extraction on first 5000 chars
        full_text = " ".join(
            child.get("content", "")
            for pair in pairs[:5]
            for child in pair.get("children", [])
        )[:5000]
        entities_meta = {}
        if os.environ.get("COMPREHEND_ENABLED", "true").lower() == "true" and full_text:
            entities_meta = _run_comprehend(full_text)

        # Store all parent + child chunks
        for pair in pairs:
            cursor.execute("""
                INSERT INTO parent_chunks
                    (document_id, section_title, element_type, page_start, page_end,
                     content, char_count, chunk_index, metadata)
                VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING id::text
            """, (
                doc_id,
                pair.get("parent_section", ""),
                pair.get("parent_element_type", "text"),
                pair.get("parent_page_start", 1),
                pair.get("parent_page_end", 1),
                pair.get("parent_content", ""),
                len(pair.get("parent_content", "")),
                pairs.index(pair),
                json.dumps({}),
            ))
            parent_id = cursor.fetchone()[0]

            for child in pair.get("children", []):
                embedding = child.get("embedding", [])
                if not embedding:
                    continue
                vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
                cursor.execute("""
                    INSERT INTO child_chunks
                        (parent_id, document_id, content, embedding,
                         element_type, page_number, chunk_index, entities, metadata)
                    VALUES (%s::uuid, %s::uuid, %s, %s::vector,
                            %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT DO NOTHING
                """, (
                    parent_id, doc_id, child.get("content", ""), vec_str,
                    child.get("element_type", "text"),
                    child.get("page_number", 1),
                    child.get("chunk_index", 0),
                    json.dumps(entities_meta),
                    json.dumps({}),
                ))
                total_chunks += 1

        # Mark document ready
        cursor.execute("""
            UPDATE documents
            SET status='ready', entities=%s::jsonb, page_count=%s
            WHERE id=%s::uuid
        """, (json.dumps(entities_meta), event.get("page_count", 0), doc_id))
        conn.commit()
        print(f"Stored {total_chunks} chunks for document {doc_id}")

    except Exception as e:
        print(f"Stage3 DB error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

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
        "elapsed_ms": elapsed_ms,
    }


def _get_db_conn():
    import psycopg2
    import boto3
    # Try env var first
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        ssm = boto3.client("ssm", region_name=os.environ.get("REGION", "us-east-2"))
        resp = ssm.get_parameter(Name="/documind/rds-connection-string", WithDecryption=True)
        dsn = resp["Parameter"]["Value"]
    return psycopg2.connect(dsn)


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
