"""Lambda Stage 2: Hierarchical chunking + embedding.

Input:  {s3_key, parse_result_key, page_count, element_counts, ...}
Output: {s3_key, embed_result_key, chunk_count, parent_count, child_count, ...}

Loads parse results from S3, runs HierarchicalChunker, embeds children with
Bedrock Titan v2 (the Lambda zip ships no ML libraries, so "local" mode only
works when run outside Lambda).
Saves embeddings to S3: embeddings/{s3_key}.json
"""

import boto3
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor


def lambda_handler(event, context):
    s3_key = event["s3_key"]
    parse_result_key = event["parse_result_key"]
    bucket = event.get("bucket", os.environ.get("S3_BUCKET", ""))
    embedding_mode = os.environ.get("EMBEDDING_MODE", "bedrock")

    print(f"Stage2: chunking + embedding {s3_key}")
    t0 = time.time()

    s3 = boto3.client("s3")

    # Load parse results
    resp = s3.get_object(Bucket=bucket, Key=parse_result_key)
    parse_result = json.loads(resp["Body"].read())
    elements = parse_result.get("elements", [])

    # Build simple LayoutDocument-like structure for chunker
    pairs = _chunk_elements(elements)

    # Embed child chunks
    children_with_embeddings = []
    all_texts = []
    for pair in pairs:
        for child in pair["children"]:
            all_texts.append(child["content"])

    print(f"Embedding {len(all_texts)} chunks (mode={embedding_mode})")
    embeddings = _embed_texts(all_texts, embedding_mode)

    idx = 0
    parent_count = len(pairs)
    child_count = 0
    for pair in pairs:
        for child in pair["children"]:
            child["embedding"] = embeddings[idx]
            idx += 1
            child_count += 1

    # Save to S3
    embed_result = {
        "s3_key": s3_key,
        "pairs": pairs,
        "parent_count": parent_count,
        "child_count": child_count,
        "embed_mode": embedding_mode,
        "embed_dim": len(embeddings[0]) if embeddings else 0,
        "embed_time_s": round(time.time() - t0, 1),
    }
    embed_result_key = f"embeddings/{s3_key}.json"
    s3.put_object(
        Bucket=bucket,
        Key=embed_result_key,
        Body=json.dumps(embed_result, default=str).encode(),
        ContentType="application/json",
    )
    print(f"Embeddings saved: {embed_result_key} ({parent_count} parents, {child_count} children)")

    return {
        **event,
        "embed_result_key": embed_result_key,
        "chunk_count": child_count,
        "parent_count": parent_count,
        "child_count": child_count,
    }


def _chunk_elements(elements: list[dict]) -> list[dict]:
    """Simple chunking for Lambda — no dependency on aws.textract classes."""
    pairs: list[dict] = []
    current_section = ""
    current_texts: list[str] = []
    current_page = 1

    def _flush():
        if not current_texts:
            return
        text = " ".join(current_texts)
        children = _make_text_children(text, current_page)
        pairs.append({
            "parent_content": text,
            "parent_section": current_section,
            "parent_element_type": "text",
            "parent_page_start": current_page,
            "parent_page_end": current_page,
            "children": children,
        })
        current_texts.clear()

    for el in elements:
        etype = el.get("type", "")
        content = el.get("content", "")
        page = el.get("page", 1)

        if etype in ("TITLE", "SECTION_HEADER"):
            _flush()
            current_section = content
            current_page = page
        elif etype == "TEXT":
            current_texts.append(content)
            if sum(len(t.split()) for t in current_texts) > 900:
                _flush()
        elif etype == "TABLE":
            _flush()
            rows = [l for l in content.splitlines() if l.strip() and not all(c in "-|: " for c in l)]
            header = rows[0] if rows else ""
            sep = next((l for l in content.splitlines() if all(c in "-|: " for c in l)), "")
            children = []
            for i, row in enumerate(rows[1:]):
                children.append({
                    "content": f"{header}\n{sep}\n{row}" if header else row,
                    "element_type": "table_row",
                    "page_number": page,
                    "chunk_index": i,
                })
            if not children:
                children = [{"content": content, "element_type": "table_row",
                             "page_number": page, "chunk_index": 0}]
            pairs.append({
                "parent_content": content,
                "parent_section": current_section,
                "parent_element_type": "table",
                "parent_page_start": page,
                "parent_page_end": page,
                "children": children,
            })
        elif etype == "FIGURE":
            _flush()
            pairs.append({
                "parent_content": content,
                "parent_section": current_section,
                "parent_element_type": "figure",
                "parent_page_start": page,
                "parent_page_end": page,
                "children": [{"content": content, "element_type": "figure_desc",
                               "page_number": page, "chunk_index": 0}],
            })

    _flush()
    return pairs


def _make_text_children(text: str, page: int, max_tokens: int = 300) -> list[dict]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    children = []
    current = []
    chunk_idx = 0
    for sent in sentences:
        current.append(sent)
        if sum(len(s.split()) for s in current) >= int(max_tokens * 0.75):
            child_text = " ".join(current)
            if child_text.strip():
                children.append({
                    "content": child_text,
                    "element_type": "text",
                    "page_number": page,
                    "chunk_index": chunk_idx,
                })
                chunk_idx += 1
                overlap = max(1, len(current) // 5)
                current = current[-overlap:]
    if current:
        child_text = " ".join(current)
        if child_text.strip():
            children.append({
                "content": child_text,
                "element_type": "text",
                "page_number": page,
                "chunk_index": chunk_idx,
            })
    return children or [{"content": text, "element_type": "text",
                          "page_number": page, "chunk_index": 0}]


def _embed_texts(texts: list[str], mode: str) -> list[list[float]]:
    if not texts:
        return []
    if mode == "local":
        return _embed_local(texts)
    return _embed_bedrock(texts)


def _embed_local(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    # batch_size=32, parallel with ThreadPoolExecutor for speed
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()


def _embed_bedrock(texts: list[str]) -> list[list[float]]:
    import boto3
    client = boto3.client("bedrock-runtime")
    model_id = "amazon.titan-embed-text-v2:0"

    def _one(text: str) -> list[float]:
        # Titan v2 accepts dimensions 256 / 512 / 1024 only
        body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
        resp = client.invoke_model(modelId=model_id, body=body,
                                   contentType="application/json", accept="application/json")
        return json.loads(resp["body"].read())["embedding"]

    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(_one, texts))
