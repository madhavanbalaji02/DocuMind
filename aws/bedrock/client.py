"""Bedrock LLM client for DocuMind.

Drop-in replacement for direct Anthropic SDK calls.
Fallback chain: Bedrock → direct Anthropic SDK (if throttled).

COST: Haiku = $0.00025/1K input, $0.00125/1K output
500 research runs/month × ~4K tokens = ~$3/month
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import boto3
from botocore.exceptions import ClientError

from aws.config import REGION
from aws.bedrock.model_registry import get_model_id, estimate_cost, DEFAULT_RESEARCH_MODEL


class BedrockLLM:
    """Unified LLM interface using Bedrock with Anthropic fallback."""

    def __init__(self, model: str = DEFAULT_RESEARCH_MODEL) -> None:
        self.model_name = model
        self.model_id = get_model_id(model)
        self._bedrock = boto3.client("bedrock-runtime", region_name=REGION)
        self._provider = os.environ.get("LLM_PROVIDER", "bedrock")
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        """Generate a completion. Returns the full text response."""
        if self._provider == "bedrock":
            try:
                return self._bedrock_complete(prompt, system, max_tokens, temperature)
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("ThrottlingException", "ServiceUnavailableException"):
                    print(f"[Bedrock] throttled ({code}), falling back to Anthropic SDK")
                    return self._anthropic_complete(prompt, system, max_tokens)
                raise
        return self._anthropic_complete(prompt, system, max_tokens)

    def _bedrock_complete(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            body["system"] = system

        resp = self._bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        text = result["content"][0]["text"]

        # Log cost
        usage = result.get("usage", {})
        cost = estimate_cost(
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            self.model_name,
        )
        print(f"[Bedrock] {self.model_name}: {usage} → ${cost:.5f}")
        return text

    def _anthropic_complete(self, prompt: str, system: str | None, max_tokens: int) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self._anthropic_key)
        kwargs: dict = {
            "model": "claude-haiku-4-5",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        print(f"[Anthropic fallback] used: {resp.usage}")
        return resp.content[0].text

    def vision(self, image_bytes: bytes, prompt: str, max_tokens: int = 512) -> str:
        """Describe an image using Claude Haiku vision."""
        import base64
        image_b64 = base64.standard_b64encode(image_bytes).decode()
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": image_b64,
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        resp = self._bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        return result["content"][0]["text"]
