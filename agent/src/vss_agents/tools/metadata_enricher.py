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

"""
Metadata Enricher Tool — assembles a structured JSON-LD metadata document.

Aggregates all pipeline outputs (EAD cues, chapters, video identity) into a
schema.org VideoObject JSON-LD document. This document can be:
  - Embedded in a video player page (<script type="application/ld+json">)
  - Stored alongside the video asset in a CMS
  - Indexed by search engines to improve video accessibility and discoverability

The document includes:
  - Video identity: name, description, ISO 8601 duration
  - Full EAD transcript: all cue descriptions concatenated
  - Chapter list: schema.org Clip objects with titles and time codes
  - Per-segment clips: schema.org Clip objects for every EAD cue
  - Accessibility metadata: WCAG features, hazard classification
"""


import json
import logging
from collections.abc import AsyncGenerator

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel
from pydantic import Field

from vss_agents.data_models.ead import Chapter
from vss_agents.data_models.ead import EADCue
from vss_agents.data_models.ead import VideoMetadataDocument
from vss_agents.data_models.ead import seconds_to_iso_duration
from vss_agents.data_models.ead import seconds_to_vtt_timestamp

logger = logging.getLogger(__name__)


def _build_chapter_clips(chapters: list[Chapter]) -> list[dict]:
    """Convert chapters to schema.org Clip objects."""
    clips = []
    for ch in chapters:
        clips.append(
            {
                "@type": "Clip",
                "name": ch.title,
                "description": ch.summary,
                "startOffset": ch.start_seconds,
                "endOffset": ch.end_seconds,
                "startTime": seconds_to_vtt_timestamp(ch.start_seconds),
                "endTime": seconds_to_vtt_timestamp(ch.end_seconds),
            }
        )
    return clips


def _build_segment_clips(cues: list[EADCue]) -> list[dict]:
    """Convert EAD cues to schema.org Clip objects (hasPart)."""
    clips = []
    for cue in cues:
        clips.append(
            {
                "@type": "Clip",
                "name": f"Segment {cue.index + 1}",
                "description": cue.description,
                "startOffset": cue.start_seconds,
                "endOffset": cue.end_seconds,
                "startTime": seconds_to_vtt_timestamp(cue.start_seconds),
                "endTime": seconds_to_vtt_timestamp(cue.end_seconds),
            }
        )
    return clips


def _build_transcript(cues: list[EADCue]) -> str:
    """Concatenate all EAD cue descriptions into a full visual transcript."""
    return " ".join(c.description for c in cues if c.description.strip())


def _overall_description(chapters: list[Chapter], cues: list[EADCue]) -> str:
    """Build an overall video description from chapter summaries or first/last cues."""
    if chapters:
        # Use all chapter summaries
        return " ".join(ch.summary for ch in chapters)
    elif cues:
        # Fallback: combine first and last cue descriptions
        parts = [cues[0].description]
        if len(cues) > 1:
            parts.append(cues[-1].description)
        return " ".join(parts)
    return ""


class MetadataEnricherConfig(FunctionBaseConfig, name="metadata_enricher"):
    """Configuration for the Metadata Enricher tool."""

    model_config = {"extra": "forbid"}


class MetadataEnricherInput(BaseModel):
    """Input for the Metadata Enricher tool."""

    sensor_id: str = Field(
        ...,
        min_length=1,
        description="Video sensor ID or filename — used as a fallback identifier if no title is provided.",
    )
    ead_cues_json: str = Field(
        ...,
        description="JSON array of EADCue objects — the output of visual_describer.",
    )
    chapters_json: str = Field(
        ...,
        description="JSON array of Chapter objects — the output of chapter_generator.",
    )
    video_title: str | None = Field(
        default=None,
        description="Human-readable video title. Defaults to sensor_id when omitted.",
    )
    video_duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Total video duration in seconds. When omitted, it is derived from the "
            "end_seconds of the last EAD cue."
        ),
    )

    model_config = {"extra": "forbid"}


@register_function(config_type=MetadataEnricherConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def metadata_enricher(config: MetadataEnricherConfig, builder: Builder) -> AsyncGenerator[FunctionInfo]:
    """Metadata Enricher — assembles a JSON-LD VideoObject metadata document."""

    async def _metadata_enricher(input: MetadataEnricherInput) -> str:
        """
        Assemble a JSON-LD schema.org VideoObject metadata enrichment document.

        Combines EAD cues, chapter manifest, and video identity into a structured
        document suitable for embedding in web pages or storing alongside the video
        asset. The document includes a full visual transcript, chapter navigation
        structure, and WCAG accessibility feature declarations.

        Returns:
            JSON-LD VideoObject document (JSON string).
        """
        # Parse inputs
        try:
            raw_cues = json.loads(input.ead_cues_json)
            cues = [EADCue.model_validate(c) for c in raw_cues]
        except Exception as e:
            raise ValueError(f"Failed to parse ead_cues_json: {e}") from e

        try:
            raw_chapters = json.loads(input.chapters_json)
            chapters = [Chapter.model_validate(ch) for ch in raw_chapters]
        except Exception as e:
            raise ValueError(f"Failed to parse chapters_json: {e}") from e

        title = input.video_title or input.sensor_id

        # Derive duration
        if input.video_duration_seconds is not None:
            total_duration = input.video_duration_seconds
        elif cues:
            total_duration = cues[-1].end_seconds
        elif chapters:
            total_duration = chapters[-1].end_seconds
        else:
            total_duration = 0.0

        logger.info(
            f"Building metadata: title='{title}', {len(cues)} cues, "
            f"{len(chapters)} chapters, duration={total_duration:.1f}s"
        )

        transcript = _build_transcript(cues)
        overall_desc = _overall_description(chapters, cues)
        chapter_clips = _build_chapter_clips(chapters)
        segment_clips = _build_segment_clips(cues)
        duration_iso = seconds_to_iso_duration(total_duration)

        doc = VideoMetadataDocument(
            name=title,
            description=overall_desc,
            duration_iso=duration_iso,
            transcript=transcript,
            chapter_list=chapter_clips,
            has_part=segment_clips,
        )

        return doc.model_dump_json(by_alias=True, indent=2)

    yield FunctionInfo.create(
        single_fn=_metadata_enricher,
        description=_metadata_enricher.__doc__,
        input_schema=MetadataEnricherInput,
        single_output_schema=str,
    )
