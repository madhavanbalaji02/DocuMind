"""Bedrock model IDs for DocuMind.

COST (per 1K tokens):
  Haiku:  $0.00025 input / $0.00125 output
  Sonnet: $0.003 input / $0.015 output
  Titan Embed v2: $0.00002 / 1K tokens

Budget allocation (500 research runs/month):
  Each run ~4K tokens Haiku → ~$0.006 → 500 runs = ~$3/month
"""

MODELS = {
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "opus": "us.anthropic.claude-opus-4-5-20251101-v1:0",
}

EMBEDDING_MODELS = {
    "titan": "amazon.titan-embed-text-v2:0",  # 1536-dim, $0.00002/1K tokens
    "local": "BAAI/bge-small-en-v1.5",        # 384-dim, free, CPU
}

DEFAULT_RESEARCH_MODEL = "haiku"
DEFAULT_VISION_MODEL = "haiku"  # supports vision input


def get_model_id(name: str = DEFAULT_RESEARCH_MODEL) -> str:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from: {list(MODELS)}")
    return MODELS[name]


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "haiku") -> float:
    """Estimate cost in USD for a Bedrock call."""
    pricing = {
        "haiku": (0.00025, 0.00125),
        "sonnet": (0.003, 0.015),
        "opus": (0.015, 0.075),
    }
    in_price, out_price = pricing.get(model, pricing["haiku"])
    return (input_tokens * in_price + output_tokens * out_price) / 1000
