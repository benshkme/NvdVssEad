# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data models for the Extended Audio Description (EAD) pipeline."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


# ---------------------------------------------------------------------------
# Caption / subtitle models
# ---------------------------------------------------------------------------


class CaptionCue(BaseModel):
    """A single subtitle or caption cue parsed from a VTT or SRT file."""

    index: int = Field(..., description="Cue index (1-based, as in the source file)")
    start_seconds: float = Field(..., ge=0.0, description="Cue start time in seconds from the beginning of the video")
    end_seconds: float = Field(..., ge=0.0, description="Cue end time in seconds from the beginning of the video")
    text: str = Field(..., description="Plain text of the caption (HTML tags and cue settings stripped)")

    @model_validator(mode="after")
    def validate_times(self) -> "CaptionCue":
        if self.end_seconds <= self.start_seconds:
            raise ValueError(f"Cue {self.index}: end_seconds must be after start_seconds")
        return self


class ParsedCaptions(BaseModel):
    """A fully parsed caption / subtitle file."""

    format: Literal["srt", "vtt"] = Field(..., description="Detected source format")
    cue_count: int = Field(..., ge=0, description="Total number of cues parsed")
    cues: list[CaptionCue] = Field(default_factory=list, description="Ordered list of caption cues")

    def cues_in_range(self, start_seconds: float, end_seconds: float) -> list[CaptionCue]:
        """Return all cues that overlap with [start_seconds, end_seconds)."""
        return [c for c in self.cues if c.end_seconds > start_seconds and c.start_seconds < end_seconds]

    def text_in_range(self, start_seconds: float, end_seconds: float) -> str:
        """Return the concatenated text of all cues overlapping the given range."""
        return " ".join(c.text for c in self.cues_in_range(start_seconds, end_seconds)).strip()


# ---------------------------------------------------------------------------
# Scene segmentation models
# ---------------------------------------------------------------------------


class SceneSegment(BaseModel):
    """A time-bounded segment of video identified for individual EAD processing."""

    index: int = Field(..., ge=0, description="Zero-based segment index")
    start_seconds: float = Field(..., ge=0.0, description="Segment start time in seconds")
    end_seconds: float = Field(..., ge=0.0, description="Segment end time in seconds")
    duration: float = Field(..., gt=0.0, description="Segment duration in seconds")
    caption_context: str = Field(
        default="",
        description=(
            "Subtitle/caption text whose timing overlaps this segment. "
            "Provided to the VLM as dialogue/speech context when generating the description."
        ),
    )

    @model_validator(mode="after")
    def validate_times(self) -> "SceneSegment":
        if self.end_seconds <= self.start_seconds:
            raise ValueError(f"Segment {self.index}: end_seconds must be after start_seconds")
        return self


# ---------------------------------------------------------------------------
# EAD cue model
# ---------------------------------------------------------------------------


class EADCue(BaseModel):
    """A single Extended Audio Description cue, covering one video segment."""

    index: int = Field(..., ge=0, description="Zero-based cue index matching the source SceneSegment")
    start_seconds: float = Field(..., ge=0.0, description="Cue start time in seconds")
    end_seconds: float = Field(..., ge=0.0, description="Cue end time in seconds")
    description: str = Field(
        ...,
        description=(
            "Visual description written for blind / visually impaired viewers. "
            "Present tense, active voice, visual-only — no dialogue or sound references."
        ),
    )

    @model_validator(mode="after")
    def validate_times(self) -> "EADCue":
        if self.end_seconds <= self.start_seconds:
            raise ValueError(f"EADCue {self.index}: end_seconds must be after start_seconds")
        return self


# ---------------------------------------------------------------------------
# Chapter model
# ---------------------------------------------------------------------------


class Chapter(BaseModel):
    """A chapter grouping a contiguous range of EAD cues into a named section."""

    index: int = Field(..., ge=0, description="Zero-based chapter index")
    title: str = Field(..., description="Short descriptive chapter title (5–10 words)")
    start_seconds: float = Field(..., ge=0.0, description="Chapter start time in seconds")
    end_seconds: float = Field(..., ge=0.0, description="Chapter end time in seconds")
    summary: str = Field(
        ...,
        description="2–3 sentence visual summary of the chapter content for blind viewers",
    )
    cue_indices: list[int] = Field(
        ...,
        description="Zero-based indices of the EADCues contained in this chapter",
    )

    @model_validator(mode="after")
    def validate_times(self) -> "Chapter":
        if self.end_seconds <= self.start_seconds:
            raise ValueError(f"Chapter {self.index}: end_seconds must be after start_seconds")
        return self


# ---------------------------------------------------------------------------
# Metadata enrichment document (JSON-LD schema.org VideoObject)
# ---------------------------------------------------------------------------


class VideoMetadataDocument(BaseModel):
    """
    JSON-LD metadata enrichment document following schema.org VideoObject.
    Intended to be embedded in video player pages or attached to the video asset.
    """

    schema_context: str = Field(
        default="https://schema.org",
        alias="@context",
        description="JSON-LD context URI",
    )
    schema_type: str = Field(
        default="VideoObject",
        alias="@type",
        description="Schema.org type",
    )
    name: str = Field(..., description="Video title")
    description: str = Field(..., description="Overall visual summary of the full video")
    duration_iso: str = Field(
        ...,
        alias="duration",
        description="ISO 8601 duration string (e.g., PT1H30M15S)",
    )
    transcript: str = Field(
        default="",
        description="Full concatenated EAD text — a complete visual transcript of the video",
    )
    chapter_list: list[dict] = Field(
        default_factory=list,
        description="schema.org Clip objects representing each chapter",
    )
    has_part: list[dict] = Field(
        default_factory=list,
        alias="hasPart",
        description="schema.org Clip objects for each individual EAD segment",
    )
    accessibility_feature: list[str] = Field(
        default_factory=lambda: [
            "audioDescription",
            "extendedAudioDescription",
            "structuralNavigation",
            "tableOfContents",
        ],
        alias="accessibilityFeature",
        description="WCAG / schema.org accessibility features present in the described video",
    )
    accessibility_hazard: str = Field(
        default="none",
        alias="accessibilityHazard",
        description="Accessibility hazard classification (schema.org)",
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Sensitivity helpers (used by both scene_segmenter and visual_describer)
# ---------------------------------------------------------------------------


def sensitivity_to_chunk_duration(sensitivity: float) -> float:
    """
    Map a sensitivity value (0.0–1.0) to a video chunk duration in seconds.

    Higher sensitivity → shorter chunks → more granular EAD descriptions.
    Lower sensitivity → longer chunks → fewer, broader descriptions.

    Mapping (exponential curve):
      1.0  →   5 s
      0.75 →  10 s
      0.5  →  24 s  (default)
      0.25 →  55 s
      0.0  → 120 s
    """
    s = max(0.0, min(1.0, float(sensitivity)))
    log_min = math.log(5.0)    # minimum chunk at max sensitivity
    log_max = math.log(120.0)  # maximum chunk at min sensitivity
    return math.exp(log_max - s * (log_max - log_min))


NO_DESCRIPTION_NEEDED = "[NO_DESCRIPTION_NEEDED]"
"""
Sentinel returned by the VLM when a segment contains no visual information that
warrants EAD treatment (e.g., a talking-head with no slides, text, or visual events,
where all information is already in the spoken audio). The EAD formatter skips
these cues and they are excluded from output files.
"""


def sensitivity_to_change_description(sensitivity: float) -> str:
    """
    Map a sensitivity value to a natural-language description of what constitutes
    a 'scene change', used in VLM prompts within the scene_segmenter.
    """
    s = max(0.0, min(1.0, float(sensitivity)))
    if s >= 0.75:
        return (
            "any visual change, however subtle — including camera angle shifts, "
            "lighting changes, character repositioning, background changes, or pacing differences"
        )
    elif s >= 0.5:
        return (
            "clear visual changes such as scene cuts, new characters entering the frame, "
            "significant action transitions, or obvious setting changes"
        )
    elif s >= 0.25:
        return (
            "significant scene changes such as new locations, major action transitions, "
            "or clear narrative sequence breaks"
        )
    else:
        return (
            "major scene changes only — entirely new locations, significant time jumps, "
            "or chapter-level narrative divisions"
        )


def sensitivity_to_priority_guidance(sensitivity: float) -> str:
    """
    Map a sensitivity value (0.0–1.0) to a description of which EAD priority
    levels to describe in a given segment, aligned with the EAD priority hierarchy:

      1. Actions that change the narrative or outcome
      2. On-screen text (titles, labels, data, slides — read verbatim)
      3. Identity of people on screen
      4. Setting and context establishment
      5. Supporting visual detail (color, texture, aesthetics — only when meaningful)

    Higher sensitivity → describe more levels.
    Lower sensitivity → describe only the highest-priority content.
    """
    s = max(0.0, min(1.0, float(sensitivity)))
    if s >= 0.75:
        return (
            "Describe content at all five priority levels:\n"
            "  1. Actions that change the narrative or outcome\n"
            "  2. On-screen text — read verbatim (titles, slides, labels, lower-thirds,\n"
            "     subtitles of foreign speech, credits)\n"
            "  3. Identity: named individuals by name; unnamed by a consistent observable\n"
            "     attribute (e.g., 'the woman in the blue jacket')\n"
            "  4. Setting and context — when it establishes essential understanding\n"
            "  5. Supporting visual detail: color, texture, shape — only when the attribute\n"
            "     carries meaning (e.g., a color-coded chart). Use basic color terms (red,\n"
            "     light blue) — not brand names or subjective descriptors."
        )
    elif s >= 0.5:
        return (
            "Describe content at priority levels 1–4:\n"
            "  1. Actions that change the narrative or outcome\n"
            "  2. On-screen text — read verbatim (titles, slides, labels, lower-thirds,\n"
            "     subtitles of foreign speech, credits)\n"
            "  3. Identity: named individuals by name; unnamed by a consistent observable\n"
            "     attribute (e.g., 'the woman in the blue jacket')\n"
            "  4. Setting and context — when it establishes essential understanding\n"
            "  Skip level 5 (supporting visual detail) unless the attribute directly changes\n"
            "  meaning for the viewer."
        )
    elif s >= 0.25:
        return (
            "Describe content at priority levels 1–3 only:\n"
            "  1. Actions that change the narrative or outcome\n"
            "  2. On-screen text — read verbatim (titles, slides, labels, lower-thirds,\n"
            "     subtitles of foreign speech, credits)\n"
            "  3. Identity: named individuals by name; unnamed by a consistent observable\n"
            "     attribute (e.g., 'the woman in the blue jacket')\n"
            "  Skip levels 4–5 (setting and visual detail) unless essential to meaning."
        )
    else:
        return (
            "Describe only the two highest-priority content types:\n"
            "  1. Actions that change the narrative or outcome\n"
            "  2. On-screen text — read verbatim (titles, slides, labels, lower-thirds,\n"
            "     subtitles of foreign speech, credits)\n"
            "  Omit identity, setting, and visual details unless they directly change meaning."
        )


def seconds_to_iso_duration(total_seconds: float) -> str:
    """Convert a duration in seconds to an ISO 8601 duration string (PTxHxMxS)."""
    total_seconds = max(0.0, total_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    parts = ["PT"]
    if hours:
        parts.append(f"{hours}H")
    if minutes:
        parts.append(f"{minutes}M")
    if seconds or not (hours or minutes):
        parts.append(f"{seconds:.3f}S")
    return "".join(parts)


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to a WebVTT timestamp string (HH:MM:SS.mmm)."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def seconds_to_ttml_timestamp(seconds: float) -> str:
    """Convert seconds to a TTML timestamp string (HH:MM:SS.mmm)."""
    return seconds_to_vtt_timestamp(seconds)  # same format
