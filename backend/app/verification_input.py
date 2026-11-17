"""Input contracts for the future India-first verification pipeline."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

InputType = Literal[
    "plain_text",
    "social_media_text",
    "article_text",
    "url",
    "image",
    "video",
]
InputProcessingStatus = Literal["ready", "not_implemented", "verification_unavailable"]


class VerificationInput(BaseModel):
    input_type: InputType
    text: str | None = None
    url: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class InputProcessingResult(BaseModel):
    input_type: InputType
    status: InputProcessingStatus
    text: str | None = None
    reason: str | None = None


URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)


def classify_input(
    value: str, input_type: InputType | None = None
) -> VerificationInput:
    """Classify a string without fetching it or inspecting external content."""
    if input_type is not None:
        return VerificationInput(input_type=input_type, text=value)
    if URL_RE.fullmatch(value.strip()):
        return VerificationInput(input_type="url", url=value.strip())
    return VerificationInput(input_type="plain_text", text=value)


def process_input(value: VerificationInput) -> InputProcessingResult:
    if value.input_type in {"image", "video", "url"}:
        return InputProcessingResult(
            input_type=value.input_type,
            status="not_implemented",
            reason=(
                "Image, video, and URL fetching are not implemented in the offline baseline."
            ),
        )
    if not value.text or not value.text.strip():
        return InputProcessingResult(
            input_type=value.input_type,
            status="verification_unavailable",
            reason="No text content was provided.",
        )
    return InputProcessingResult(
        input_type=value.input_type,
        status="ready",
        text=value.text,
    )
