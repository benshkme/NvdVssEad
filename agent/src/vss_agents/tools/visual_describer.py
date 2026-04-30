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
Visual Describer Tool — generates Extended Audio Description cues for video segments.

For each SceneSegment (output of scene_segmenter), this tool calls the VLM via
video_understanding with an EAD-specific prompt. When caption context is present
in a segment, it is injected into the prompt so the VLM can use dialogue/speech
as additional context without reproducing it verbatim.

Sensitivity-tuned prompting:
  - High sensitivity: asks for detailed, sentence-level descriptions of subtle
    visual changes, expressions, gestures, and transitions.
  - Low sensitivity:  asks for broader scene-level descriptions covering the
    main visual content without over-specifying transient details.

Segments are processed concurrently (up to max_concurrent tasks) to keep
processing time proportional to video length rather than segment count.
"""

from __future__ import annotations

import asyncio
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

from vss_agents.data_models.ead import EADCue
from vss_agents.data_models.ead import SceneSegment
from vss_agents.data_models.ead import sensitivity_to_change_description

logger = logging.getLogger(__name__)

_SENSITIVITY_DEFAULT = 0.5
_MAX_CONCURRENT_DEFAULT = 4


def _build_ead_prompt(segment: SceneSegment, sensitivity: float) -> str:
    """Construct the VLM user prompt for a single EAD segment."""
    change_desc = sensitivity_to_change_description(sensitivity)

    # Caption context block — injected only when captions exist for this segment
    caption_block = ""
    if segment.caption_context.strip():
        caption_block = (
            f"\nDIALOGUE / SPEECH CONTEXT (for reference only — "
            f"do NOT reproduce verbatim in your description):\n"
            f'"{segment.caption_context.strip()}"\n'
        )

    # Detail level guidance based on sensitivity
    if sensitivity >= 0.75:
        detail_guidance = (
            "Provide a detailed, sentence-level description. Note subtle visual elements: "
            "facial expressions, hand gestures, minor background details, camera movement, "
            "and moment-to-moment action. Each visual change should be described."
        )
    elif sensitivity >= 0.5:
        detail_guidance = (
            "Provide a clear, moderately detailed description. Cover the main visual action, "
            "characters and their activities, the setting, and any significant visual transitions."
        )
    elif sensitivity >= 0.25:
        detail_guidance = (
            "Provide a concise description focused on the primary visual content: "
            "where we are, who is present, and what the main action is. "
            "You do not need to describe every subtle detail."
        )
    else:
        detail_guidance = (
            "Provide a brief, high-level description of the scene: the setting, "
            "the main subject or action, and any significant visual events. "
            "Avoid over-describing transient details."
        )

    return (
        f"You are generating an Extended Audio Description (EAD) cue for blind and visually "
        f"impaired viewers. This is segment {segment.index + 1}, covering "
        f"{segment.start_seconds:.1f}s – {segment.end_seconds:.1f}s of the video.\n"
        f"{caption_block}\n"
        f"DESCRIPTION RULES:\n"
        f"- Describe ONLY what is visible on screen. Do not mention audio, music, or dialogue.\n"
        f"- Use present tense and active voice (e.g. 'A man walks towards the camera').\n"
        f"- Describe people by appearance, clothing, and action — not by assumed name unless "
        f"  a name appears on screen.\n"
        f"- Include: setting/location, people present, their actions and expressions, "
        f"  on-screen text or graphics, and camera movement or transitions.\n"
        f"- Be objective and factual. Avoid interpretation or emotional editorialising.\n"
        f"- This segment is sensitive to: {change_desc}.\n\n"
        f"{detail_guidance}\n\n"
        f"OUTPUT: Write one concise paragraph of visual description for this segment only."
    )


class VisualDescriberConfig(FunctionBaseConfig, name="visual_describer"):
    """Configuration for the Visual Describer tool."""

    video_understanding_tool: str = Field(
        default="video_understanding",
        description="Name of the video_understanding tool used to call the VLM per segment.",
    )
    default_sensitivity: float = Field(
        default=_SENSITIVITY_DEFAULT,
        ge=0.0,
        le=1.0,
        description="Default sensitivity when no override is provided in the input.",
    )
    max_concurrent: int = Field(
        default=_MAX_CONCURRENT_DEFAULT,
        ge=1,
        le=16,
        description="Maximum number of segments to describe concurrently.",
    )
    vlm_reasoning: bool = Field(
        default=False,
        description="Enable VLM reasoning mode (cosmos-reason models) for richer descriptions.",
    )

    model_config = {"extra": "forbid"}


class VisualDescriberInput(BaseModel):
    """Input for the Visual Describer tool."""

    sensor_id: str = Field(
        ...,
        min_length=1,
        description="The sensor ID or video filename in VST — passed to video_understanding.",
    )
    segments_json: str = Field(
        ...,
        description=(
            "JSON array of SceneSegment objects — the output of scene_segmenter. "
            "Each segment defines start_seconds, end_seconds, and optional caption_context."
        ),
    )
    sensitivity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Sensitivity used to tune prompt detail level. Should match the value used in "
            "scene_segmenter. Defaults to the config value when omitted."
        ),
    )
    vlm_reasoning: bool | None = Field(
        default=None,
        description=(
            "Override VLM reasoning mode. When True, the VLM thinks step-by-step before "
            "answering, producing more accurate descriptions at the cost of speed. "
            "Defaults to the config value when omitted."
        ),
    )

    model_config = {"extra": "forbid"}


@register_function(config_type=VisualDescriberConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def visual_describer(config: VisualDescriberConfig, builder: Builder) -> AsyncGenerator[FunctionInfo]:
    """Visual Describer — generates EAD cues for each video segment via VLM."""

    async def _visual_describer(input: VisualDescriberInput) -> str:
        """
        Generate Extended Audio Description (EAD) cues for a list of video segments.

        For each SceneSegment (from scene_segmenter), the VLM is called with an
        EAD-specific prompt tuned to the requested sensitivity level. If the segment
        contains caption_context, that dialogue text is injected as reference context
        (the VLM is instructed not to reproduce it verbatim).

        Segments are processed concurrently to keep latency proportional to video
        length rather than segment count.

        Returns:
            JSON array of EADCue objects, each with:
              index, start_seconds, end_seconds, description
        """
        sensitivity = input.sensitivity if input.sensitivity is not None else config.default_sensitivity
        use_reasoning = input.vlm_reasoning if input.vlm_reasoning is not None else config.vlm_reasoning

        # Parse segments
        try:
            raw_segments = json.loads(input.segments_json)
            segments = [SceneSegment.model_validate(s) for s in raw_segments]
        except Exception as e:
            raise ValueError(f"Failed to parse segments_json: {e}") from e

        if not segments:
            logger.warning("visual_describer received empty segments list")
            return json.dumps([])

        logger.info(
            f"Describing {len(segments)} segments for '{input.sensor_id}', "
            f"sensitivity={sensitivity:.2f}, reasoning={use_reasoning}, "
            f"max_concurrent={config.max_concurrent}"
        )

        vu_tool = await builder.get_tool(config.video_understanding_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        semaphore = asyncio.Semaphore(config.max_concurrent)

        async def _describe_segment(segment: SceneSegment) -> EADCue:
            user_prompt = _build_ead_prompt(segment, sensitivity)
            async with semaphore:
                try:
                    result = await vu_tool.ainvoke(
                        input={
                            "sensor_id": input.sensor_id,
                            "start_timestamp": segment.start_seconds,
                            "end_timestamp": segment.end_seconds,
                            "user_prompt": user_prompt,
                            "vlm_reasoning": use_reasoning,
                        }
                    )
                    description = str(result).strip() if result else ""
                    if not description:
                        description = f"[Segment {segment.index + 1}: no visual description returned]"
                    logger.debug(f"Segment {segment.index}: described ({len(description)} chars)")
                except Exception as e:
                    logger.error(f"VLM call failed for segment {segment.index}: {e}")
                    description = f"[Segment {segment.index + 1}: description unavailable — {e}]"

            return EADCue(
                index=segment.index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                description=description,
            )

        cues = await asyncio.gather(*[_describe_segment(seg) for seg in segments])
        cues_sorted = sorted(cues, key=lambda c: c.index)

        logger.info(f"Generated {len(cues_sorted)} EAD cues for '{input.sensor_id}'")
        return json.dumps([c.model_dump() for c in cues_sorted], indent=2)

    yield FunctionInfo.create(
        single_fn=_visual_describer,
        description=_visual_describer.__doc__,
        input_schema=VisualDescriberInput,
        single_output_schema=str,
    )
