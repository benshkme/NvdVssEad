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
from vss_agents.data_models.ead import NO_DESCRIPTION_NEEDED
from vss_agents.data_models.ead import SceneSegment
from vss_agents.data_models.ead import sensitivity_to_priority_guidance

logger = logging.getLogger(__name__)

_SENSITIVITY_DEFAULT = 0.5
_MAX_CONCURRENT_DEFAULT = 4


def _build_ead_prompt(segment: SceneSegment, sensitivity: float) -> str:
    """
    Construct a guidelines-aligned VLM prompt for a single EAD segment.

    Implements the full EAD describe/do-not-describe ruleset and maps the
    sensitivity parameter to the EAD priority hierarchy levels.
    """
    priority_guidance = sensitivity_to_priority_guidance(sensitivity)

    # Audio context block — present only when captions overlap this segment.
    # Used to prevent re-describing audio content and to avoid verbatim repetition.
    audio_block = ""
    if segment.caption_context.strip():
        audio_block = (
            f"\n── AUDIO CONTEXT (spoken dialogue/narration during this segment) ──\n"
            f'"{segment.caption_context.strip()}"\n'
            f"Do NOT describe anything already clearly communicated by the above audio.\n"
            f"Do NOT reproduce this text verbatim in your description.\n"
        )

    return f"""\
You are generating an Extended Audio Description (EAD) cue for blind and visually impaired viewers.
Segment {segment.index + 1} of the video — {segment.start_seconds:.1f}s to {segment.end_seconds:.1f}s.
{audio_block}
━━━ WHAT TO DESCRIBE ━━━

Describe only visual content that meets one or more of these criteria:

1. ESSENTIAL VISUALS — anything that, if omitted, would leave a blind viewer unable to follow,
   understand, or achieve the intended learning outcome of this content.

2. ACTIONS AND EVENTS not already communicated by the audio above — physical actions, scene events,
   and cause-and-effect sequences that occur silently or are unreferenced in the narration.

3. ON-SCREEN TEXT — read verbatim:
   • Titles, headings, labels, links, presenter names, lower-thirds
   • Subtitled or translated foreign-language speech (read verbatim)
   • Text in charts, diagrams, slides, or graphics
   • Opening/closing credits when they convey meaningful information

4. SETTING AND CONTEXT — only when establishing essential understanding:
   • Time period, location, environment when relevant to meaning
   • Scene changes if they affect understanding of the narrative
   • Time passages only when there is objective visual evidence (not inference)

5. CHARACTERS AND PEOPLE:
   • Named individuals: identify by name
   • Unnamed individuals: use a consistent, observable attribute (e.g., "the woman in the blue jacket")
   • Describe race/ethnicity only when meaningful to the content's intent — and apply equally to
     BIPOC and white individuals
   • Describe significant physical characteristics only when relevant to the content
     (e.g., a patient's presentation in a medical training video)

6. VISUAL PROPERTIES — only when comprehension depends on them:
   • Shape, size, texture, color — only when the attribute carries meaning (e.g., a color-coded chart,
     two differently shaped tools being compared)
   • Use basic color terms (red, light blue) — not brand names or subjective descriptors

7. MEDIA TYPE — identify when content switches to: photograph, archival footage, animation,
   or re-enactment. Viewers need this to interpret credibility and tone.

8. MONTAGES — describe when time allows; summarizing is acceptable when individual elements
   cannot each be fully described.

9. UNRECOGNIZABLE SOUNDS WITH PERTINENT MEANING — describe the source if the sound is not
   commonly recognizable AND it matters to the content (e.g., an ambiguous mechanical sound
   in a safety training video).

━━━ WHAT NOT TO DESCRIBE ━━━

• Anything already clearly communicated by the audio above.
• Emotional states, motivations, or inferences — describe observable gestures and expressions
  only. Say "she crosses her arms and looks away" — NOT "she feels defensive."
• Cinematic/technical terms (close-up, pan, flashback) unless the technique itself is important
  to the viewer's understanding.
• Color or physical details when they carry no meaning (e.g., the color of a background wall
  in a talking-head interview).
• Commonly recognized sounds (applause, a phone ringing).
• Purely decorative background elements, stylistic choices, or aesthetic details that do not
  affect understanding.

━━━ SPECIAL CASE — NO DESCRIPTION NEEDED ━━━

If this segment contains NO visual information essential for understanding — for example, a
person talking directly to camera with no slides, text, demonstrations, or visual events, and
all relevant information is already present in the spoken audio — output EXACTLY this token
and nothing else:

{NO_DESCRIPTION_NEEDED}

━━━ PRIORITY HIERARCHY FOR THIS SEGMENT ━━━

{priority_guidance}

━━━ LANGUAGE RULES ━━━

• Present tense, active voice: "A man walks towards the camera" — not "A man is seen walking."
• Objective and factual — no interpretation, opinion, or emotional commentary.
• Be proportional: the description length should match the segment duration and information density.

OUTPUT: One concise paragraph of EAD description for this segment, or {NO_DESCRIPTION_NEEDED}.\
"""


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
                    raw = str(result).strip() if result else ""

                    # VLM signalled that this segment needs no description
                    # (talking-head with no slides/text/visual events, all info in audio).
                    # Store as empty string so the formatter silently skips it.
                    if NO_DESCRIPTION_NEEDED in raw:
                        description = ""
                        logger.info(f"Segment {segment.index}: no description needed (VLM signal)")
                    elif not raw:
                        description = ""
                        logger.warning(f"Segment {segment.index}: VLM returned empty response")
                    else:
                        description = raw
                        logger.debug(f"Segment {segment.index}: described ({len(description)} chars)")

                except Exception as e:
                    logger.error(f"VLM call failed for segment {segment.index}: {e}")
                    description = ""

            return EADCue(
                index=segment.index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                description=description,
            )

        cues = await asyncio.gather(*[_describe_segment(seg) for seg in segments])
        cues_sorted = sorted(cues, key=lambda c: c.index)

        described = sum(1 for c in cues_sorted if c.description)
        skipped = len(cues_sorted) - described
        logger.info(
            f"Visual description complete for '{input.sensor_id}': "
            f"{described} cues with descriptions, {skipped} skipped (no visual info needed)"
        )
        return json.dumps([c.model_dump() for c in cues_sorted], indent=2)

    yield FunctionInfo.create(
        single_fn=_visual_describer,
        description=_visual_describer.__doc__,
        input_schema=VisualDescriberInput,
        single_output_schema=str,
    )
