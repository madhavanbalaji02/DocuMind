"""Amazon Comprehend entity and key-phrase extraction.

COST: $0.0001 per unit (100 chars). For a 1000-char chunk: $0.001.
~500 docs/month = ~$0.05/month total.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import boto3

from aws.config import REGION


@dataclass
class Entity:
    text: str
    entity_type: str  # PERSON|ORGANIZATION|DATE|QUANTITY|COMMERCIAL_ITEM|TITLE|EVENT|LOCATION
    confidence: float


@dataclass
class EntityResult:
    entities: list[Entity]
    key_phrases: list[dict]  # [{text, confidence}]
    dominant_language: str


class DocumentEntityExtractor:
    """Extract entities and key phrases from text using Amazon Comprehend."""

    SUPPORTED_ENTITY_TYPES = {
        "PERSON", "ORGANIZATION", "DATE", "QUANTITY",
        "COMMERCIAL_ITEM", "TITLE", "EVENT", "LOCATION",
    }
    MIN_CONFIDENCE = 0.8
    MAX_BYTES = 5000  # Comprehend limit per call

    def __init__(self) -> None:
        self._comprehend = boto3.client("comprehend", region_name=REGION)
        self._enabled = os.environ.get("COMPREHEND_ENABLED", "true").lower() == "true"

    def extract(self, text: str, language_code: str = "en") -> EntityResult:
        """Extract entities and key phrases. Truncates to 5KB per Comprehend limit."""
        if not self._enabled or not text.strip():
            return EntityResult(entities=[], key_phrases=[], dominant_language="en")

        # Truncate to byte limit
        text_bytes = text.encode("utf-8")[: self.MAX_BYTES]
        text = text_bytes.decode("utf-8", errors="ignore")

        entities: list[Entity] = []
        key_phrases: list[dict] = []

        try:
            ent_resp = self._comprehend.detect_entities(Text=text, LanguageCode=language_code)
            for ent in ent_resp.get("Entities", []):
                if ent["Score"] >= self.MIN_CONFIDENCE:
                    entities.append(Entity(
                        text=ent["Text"],
                        entity_type=ent["Type"],
                        confidence=ent["Score"],
                    ))
        except Exception as e:
            print(f"[Comprehend] entity extraction error: {e}")

        try:
            kp_resp = self._comprehend.detect_key_phrases(Text=text, LanguageCode=language_code)
            key_phrases = [
                {"text": kp["Text"], "confidence": kp["Score"]}
                for kp in kp_resp.get("KeyPhrases", [])
                if kp["Score"] >= self.MIN_CONFIDENCE
            ]
        except Exception as e:
            print(f"[Comprehend] key phrase error: {e}")

        return EntityResult(
            entities=entities,
            key_phrases=key_phrases[:20],  # top 20
            dominant_language=language_code,
        )

    def to_metadata(self, result: EntityResult) -> dict:
        """Convert EntityResult to JSON-serializable metadata dict."""
        return {
            "entities": [
                {"text": e.text, "type": e.entity_type, "confidence": round(e.confidence, 3)}
                for e in result.entities
            ],
            "key_phrases": result.key_phrases,
            "dominant_language": result.dominant_language,
        }
