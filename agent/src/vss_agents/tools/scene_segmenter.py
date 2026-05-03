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
Scene Segmenter Tool — divides a video into timed segments for EAD processing.

Sensitivity parameter (0.0–1.0) controls segment granularity:
  1.0 (maximum) →  ~5s chunks  — one description every ~5 seconds
  0.75 (high)   → ~10s chunks
  0.5  (medium) → ~24s chunks  — default
  0.25 (low)    → ~55s chunks
  0.0  (minimum)→ ~120s chunks — one description every 2 minutes

When a parsed captions JSON is provided (output of caption_parser), each
segment is annotated with the overlapping subtitle text, giving the VLM
dialogue context during visual description.

The tool calls vst_video_duration to determine total video length, then
computes uniform segments of chunk_duration seconds. The final segment
absorbs any remainder shorter than half a chunk to avoid very short tail
segments.
"""


import json
import logging
import math
from collections.abc import AsyncGenerator

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel
from pydantic import Field

from vss_agents.data_models.ead import ParsedCaptions
from vss_agents.data_models.ead import SceneSegment
from vss_agents.data_models.ead import sensitivity_to_chunk_duration

logger = logging.getLogger(__name__)

_SENSITIVITY_DEFAULT = 0.5


class SceneSegmenterConfig(FunctionBaseConfig, name="scene_segmenter"):
    """Configuration for the Scene Segmenter tool."""

    default_sensitivity: float = Field(
        default=_SENSITIVITY_DEFAULT,
        ge=0.0,
        le=1.0,
        description=(
            "Default sensitivity when no override is provided in the input. "
            "0.0 = minimum (large segments, only major changes), "
            "1.0 = maximum (small segments, every subtle change)."
        ),
    )
    duration_tool: str = Field(
        default="vst_video_duration",
        description="Name of the VST duration tool used to retrieve total video length.",
    )

    model_config = {"extra": "forbid"}


class SceneSegmenterInput(BaseModel):
    """Input for the Scene Segmenter tool."""

    sensor_id: str = Field(
        ...,
        min_length=1,
        description="The sensor ID or video filename in VST to segment.",
    )
    sensitivity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Segmentation sensitivity (0.0–1.0). Overrides the config default when provided. "
            "Higher values produce shorter, more granular segments; "
            "lower values produce longer, broader segments. "
            "Omit to use the profile default (0.5 = medium)."
        ),
    )
    captions_json: str | None = Field(
        default=None,
        description=(
            "Optional JSON output from the caption_parser tool (a ParsedCaptions object). "
            "When provided, each segment is annotated with the overlapping subtitle text "
            "to give the VLM dialogue context during description generation."
        ),
    )

    model_config = {"extra": "forbid"}


def _build_segments(
    total_duration: float,
    chunk_duration: float,
    captions: ParsedCaptions | None,
) -> list[SceneSegment]:
    """
    Divide total_duration into fixed-size segments of chunk_duration seconds.

    The final segment absorbs any remainder < 0.5 * chunk_duration so we
    don't emit very short trailing segments.
    """
    if total_duration <= 0:
        return []

    segments: list[SceneSegment] = []
    n_full = int(total_duration // chunk_duration)
    remainder = total_duration - n_full * chunk_duration

    # If the remainder is at least half a chunk, keep it as its own segment
    include_tail = remainder >= chunk_duration * 0.5

    boundaries: list[float] = [i * chunk_duration for i in range(n_full + 1)]
    if include_tail and remainder > 0:
        boundaries.append(total_duration)
    elif n_full > 0:
        # Extend last boundary to cover the short remainder
        boundaries[-1] = total_duration

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        duration = end - start

        caption_context = ""
        if captions:
            caption_context = captions.text_in_range(start, end)

        segments.append(
            SceneSegment(
                index=i,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                duration=round(duration, 3),
                caption_context=caption_context,
            )
        )

    return segments


@register_function(config_type=SceneSegmenterConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def scene_segmenter(config: SceneSegmenterConfig, builder: Builder) -> AsyncGenerator[FunctionInfo]:
    """Scene Segmenter — divides a video into timed segments for EAD processing."""

    async def _scene_segmenter(input: SceneSegmenterInput) -> str:
        """
        Divide a video into timed segments suitable for Extended Audio Description generation.

        The sensitivity parameter (0.0–1.0) controls how granular the segmentation is:
          - 1.0 (maximum): ~5s segments, suitable for content with rapid visual changes
          - 0.75 (high):  ~10s segments
          - 0.5  (medium): ~24s segments  — good default for most content
          - 0.25 (low):   ~55s segments
          - 0.0  (minimum): ~120s segments, suitable for slow-paced documentary content

        If captions_json is provided (output of caption_parser), each segment is
        annotated with any subtitle text that overlaps its time range.

        Returns:
            JSON array of SceneSegment objects, each with:
              index, start_seconds, end_seconds, duration, caption_context
        """
        sensitivity = input.sensitivity if input.sensitivity is not None else config.default_sensitivity
        chunk_duration = sensitivity_to_chunk_duration(sensitivity)

        logger.info(
            f"Scene segmenter: sensor='{input.sensor_id}', sensitivity={sensitivity:.2f}, "
            f"chunk_duration={chunk_duration:.1f}s"
        )

        # Get total video duration
        duration_tool = await builder.get_tool(config.duration_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        duration_result = await duration_tool.ainvoke(input={"sensor_id": input.sensor_id})

        # VST duration tool returns an object with a duration_seconds or similar field
        if hasattr(duration_result, "duration_seconds"):
            total_duration = float(duration_result.duration_seconds)
        elif hasattr(duration_result, "duration"):
            total_duration = float(duration_result.duration)
        elif isinstance(duration_result, (int, float)):
            total_duration = float(duration_result)
        elif isinstance(duration_result, str):
            # May be a JSON string or plain number string
            try:
                parsed = json.loads(duration_result)
                if isinstance(parsed, dict):
                    total_duration = float(
                        parsed.get("duration_seconds") or parsed.get("duration") or parsed.get("total_seconds", 0)
                    )
                else:
                    total_duration = float(parsed)
            except (json.JSONDecodeError, ValueError):
                total_duration = float(duration_result)
        else:
            total_duration = float(str(duration_result))

        logger.info(f"Total video duration: {total_duration:.1f}s")

        if total_duration <= 0:
            logger.warning(f"Video '{input.sensor_id}' has zero or negative duration — returning empty segment list")
            return json.dumps([])

        # Parse captions if provided
        captions: ParsedCaptions | None = None
        if input.captions_json:
            try:
                captions = ParsedCaptions.model_validate_json(input.captions_json)
                logger.info(f"Using {captions.cue_count} caption cues for context injection")
            except Exception as e:
                logger.warning(f"Failed to parse captions_json — proceeding without captions: {e}")

        segments = _build_segments(total_duration, chunk_duration, captions)
        logger.info(
            f"Produced {len(segments)} segments "
            f"(chunk={chunk_duration:.1f}s, sensitivity={sensitivity:.2f})"
        )

        return json.dumps([seg.model_dump() for seg in segments], indent=2)

    yield FunctionInfo.create(
        single_fn=_scene_segmenter,
        description=_scene_segmenter.__doc__,
        input_schema=SceneSegmenterInput,
        single_output_schema=str,
    )
