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
Chapter Generator Tool — groups EAD cues into named, summarised chapters.

Given the full list of EADCue objects (output of visual_describer), this tool
calls the LLM to:
  1. Group consecutive cues into logically coherent chapters based on visual content.
  2. Generate a concise, descriptive title for each chapter (5–10 words).
  3. Write a 2–3 sentence visual summary of each chapter for blind viewers.

The target number of chapters is derived from total video duration and a
configurable chapters_per_hour rate. For a 30-minute video at 4 chapters/hour
that gives ~2 chapters; at 8 chapters/hour, ~4 chapters.

The LLM is instructed to return a JSON array. A fallback divider is applied if
the LLM response cannot be parsed — it creates equal-size chapters automatically.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import AsyncGenerator

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel
from pydantic import Field

from vss_agents.data_models.ead import Chapter
from vss_agents.data_models.ead import EADCue

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an accessibility specialist creating chapter structure for a video audio description.
You will receive a list of Extended Audio Description (EAD) cues with timestamps.
Your task is to group consecutive cues into meaningful chapters that help blind and
visually impaired viewers navigate the video content.

Rules for chapters:
- Each chapter must be a contiguous range of cues (no gaps, no overlaps).
- Chapter titles should be 5–10 words, descriptive, and accessible (no jargon).
- Chapter summaries should be 2–3 sentences describing the primary visual content.
- Chapters should reflect genuine narrative or visual transitions — not just equal time splits.
- Every cue must belong to exactly one chapter.
- The first chapter must start with cue index 0.
- The last chapter must end with the last cue index.

Return ONLY a valid JSON array — no markdown, no explanation, no wrapping object.
Each element must have exactly these fields:
  index        (integer, 0-based)
  title        (string)
  start_seconds (float)
  end_seconds   (float)
  summary      (string)
  cue_indices  (array of integers, the zero-based cue indices in this chapter)
"""


def _format_cues_for_llm(cues: list[EADCue]) -> str:
    lines = []
    for cue in cues:
        lines.append(
            f"[{cue.index}] {cue.start_seconds:.1f}s – {cue.end_seconds:.1f}s: {cue.description}"
        )
    return "\n".join(lines)


def _target_chapter_count(total_duration_seconds: float, chapters_per_hour: float) -> int:
    """Calculate the suggested number of chapters for the video."""
    hours = total_duration_seconds / 3600.0
    target = max(1, round(hours * chapters_per_hour))
    return target


def _fallback_chapters(cues: list[EADCue], n_chapters: int) -> list[Chapter]:
    """Create equal-size chapters when LLM output cannot be parsed."""
    if not cues:
        return []
    n_chapters = max(1, min(n_chapters, len(cues)))
    cues_per_chapter = math.ceil(len(cues) / n_chapters)
    chapters = []
    for ch_idx in range(n_chapters):
        start_i = ch_idx * cues_per_chapter
        end_i = min(start_i + cues_per_chapter, len(cues))
        if start_i >= len(cues):
            break
        chunk = cues[start_i:end_i]
        chapters.append(
            Chapter(
                index=ch_idx,
                title=f"Chapter {ch_idx + 1}",
                start_seconds=chunk[0].start_seconds,
                end_seconds=chunk[-1].end_seconds,
                summary=" ".join(c.description[:80] for c in chunk[:2]) + "...",
                cue_indices=[c.index for c in chunk],
            )
        )
    return chapters


class ChapterGeneratorConfig(FunctionBaseConfig, name="chapter_generator"):
    """Configuration for the Chapter Generator tool."""

    llm_name: LLMRef = Field(
        ...,
        description="LLM used to generate chapter titles and summaries.",
    )
    chapters_per_hour: float = Field(
        default=6.0,
        gt=0.0,
        description=(
            "Target number of chapters per hour of video. "
            "For a 30-min video: 6/hr → 3 chapters, 12/hr → 6 chapters."
        ),
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for the LLM chapter generation response.",
    )

    model_config = {"extra": "forbid"}


class ChapterGeneratorInput(BaseModel):
    """Input for the Chapter Generator tool."""

    ead_cues_json: str = Field(
        ...,
        description=(
            "JSON array of EADCue objects — the output of visual_describer. "
            "Each cue must have: index, start_seconds, end_seconds, description."
        ),
    )
    video_title: str | None = Field(
        default=None,
        description="Optional video title to include as context for chapter naming.",
    )

    model_config = {"extra": "forbid"}


@register_function(config_type=ChapterGeneratorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def chapter_generator(config: ChapterGeneratorConfig, builder: Builder) -> AsyncGenerator[FunctionInfo]:
    """Chapter Generator — groups EAD cues into titled, summarised chapters via LLM."""

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _chapter_generator(input: ChapterGeneratorInput) -> str:
        """
        Group Extended Audio Description cues into named chapters for video navigation.

        The LLM analyses all EAD cue descriptions and groups consecutive cues into
        logically coherent chapters, generating a title and 2–3 sentence summary for
        each. The resulting chapter manifest supports accessible video navigation
        and is used by metadata_enricher and ead_formatter.

        Returns:
            JSON array of Chapter objects, each with:
              index, title, start_seconds, end_seconds, summary, cue_indices
        """
        try:
            raw_cues = json.loads(input.ead_cues_json)
            cues = [EADCue.model_validate(c) for c in raw_cues]
        except Exception as e:
            raise ValueError(f"Failed to parse ead_cues_json: {e}") from e

        if not cues:
            logger.warning("chapter_generator received empty cues list")
            return json.dumps([])

        total_duration = cues[-1].end_seconds
        n_chapters = _target_chapter_count(total_duration, config.chapters_per_hour)

        logger.info(
            f"Generating chapters: {len(cues)} cues, {total_duration:.0f}s total, "
            f"target={n_chapters} chapters"
        )

        cues_text = _format_cues_for_llm(cues)
        title_context = f'Video title: "{input.video_title}"\n\n' if input.video_title else ""

        user_message = (
            f"{title_context}"
            f"Total video duration: {total_duration:.1f} seconds\n"
            f"Target number of chapters: approximately {n_chapters}\n\n"
            f"EAD CUES:\n{cues_text}\n\n"
            f"Group these cues into {n_chapters} chapters (adjust count slightly if the "
            f"content calls for it). Return ONLY the JSON array."
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        try:
            response = await llm.ainvoke(messages)
            raw_content = str(response.content).strip()

            # Strip markdown code fences if present
            if raw_content.startswith("```"):
                raw_content = raw_content.split("```")[1]
                if raw_content.startswith("json"):
                    raw_content = raw_content[4:]
                raw_content = raw_content.strip()

            chapters_data = json.loads(raw_content)
            chapters = [Chapter.model_validate(ch) for ch in chapters_data]
            logger.info(f"LLM generated {len(chapters)} chapters")

        except Exception as e:
            logger.warning(f"LLM chapter generation failed ({e}) — using fallback equal-split chapters")
            chapters = _fallback_chapters(cues, n_chapters)

        return json.dumps([ch.model_dump() for ch in chapters], indent=2)

    yield FunctionInfo.create(
        single_fn=_chapter_generator,
        description=_chapter_generator.__doc__,
        input_schema=ChapterGeneratorInput,
        single_output_schema=str,
    )
