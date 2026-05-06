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
Scene Segmenter Tool — content-driven segmentation for EAD processing.

Rather than dividing the video into equal time chunks, this tool uses the
VLM to detect *where* significant visual changes occur and creates segments
only at those change points. This produces far fewer, more meaningful
segments than time-based chunking.

The video is probed in windows (probe_window_seconds, default 60s).
For each window, the VLM returns timestamps of detected changes. These
become segment boundaries. The opening scene (t=0) is always included.

Sensitivity (0.0–1.0) controls what counts as "significant":
  1.0 (maximum): any visual change — new text, label, camera shift
  0.75 (high):   clear changes — new slide, person, scene cut
  0.5  (medium): meaningful transitions — new content, subject change  ← default
  0.25 (low):    major transitions only — scene cuts, new presenter
  0.0  (minimum): hard cuts and large content changes only

Result: a 30-minute video typically produces 10–40 segments rather than
75 equal-time chunks, with descriptions only when something actually changes.
"""

import json
import logging
import re
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
from vss_agents.data_models.ead import sensitivity_to_change_description

logger = logging.getLogger(__name__)

_SENSITIVITY_DEFAULT = 0.5
_PROBE_WINDOW_DEFAULT = 60.0   # seconds per VLM change-detection call
_DESC_WINDOW_DEFAULT = 20.0    # max seconds passed to VLM for description
_MIN_SEGMENT_GAP = 2.0         # ignore detected changes closer than this (seconds)


def _build_change_detection_prompt(
    clip_duration: float,
    sensitivity: float,
    caption_context: str = "",
) -> str:
    """Prompt asking the VLM to return JSON timestamps of visual changes."""
    change_desc = sensitivity_to_change_description(sensitivity)
    caption_note = ""
    if caption_context.strip():
        preview = caption_context.strip()[:200]
        caption_note = (
            f"\nAudio/dialogue in this clip (for context — do NOT flag speech as visual changes): "
            f'"{preview}"\n'
        )

    return f"""\
You are analyzing a {clip_duration:.0f}-second video clip to identify visual change points \
for Extended Audio Description (EAD).
{caption_note}
Identify every timestamp (in seconds from the START of this clip) where \
{change_desc}.

SIGNIFICANT CHANGES that warrant a new EAD cue:
• Hard scene cut or transition to a different location or setting
• New slide, diagram, or presentation content appearing on screen
• Text labels, title cards, lower-thirds, or on-screen graphics appearing
• Presenter switching to a new demonstration or subject
• New person appearing or significant change in who is on screen
• Switch between speaker view and shared-screen / presentation content

NOT significant — do NOT report:
• The same scene continuing with the speaker talking
• Normal gestures, head movement, or minor repositioning
• Brief pauses or cuts that return immediately to the same scene
• Same slide content with no new visual information

Return ONLY a valid JSON array of decimal timestamps in seconds from the start of this clip.
Return [] if no significant changes occur.

Examples:
  [4.5, 18.0, 31.2]   ← changes detected at those offsets
  []                   ← no significant changes in this clip\
"""


def _parse_change_timestamps(raw: str) -> list[float]:
    """Extract a list of floats from a possibly noisy VLM response."""
    if not raw:
        return []
    # Try to find a JSON array anywhere in the response
    match = re.search(r"\[([^\]]*)\]", raw)
    if not match:
        return []
    try:
        values = json.loads("[" + match.group(1) + "]")
        return sorted({float(v) for v in values if isinstance(v, (int, float)) and v >= 0})
    except (json.JSONDecodeError, ValueError):
        return []


def _merge_close_boundaries(boundaries: list[float], min_gap: float) -> list[float]:
    """Remove change points that are too close together."""
    if not boundaries:
        return []
    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] >= min_gap:
            merged.append(b)
    return merged


class SceneSegmenterConfig(FunctionBaseConfig, name="scene_segmenter"):
    """Configuration for the content-driven Scene Segmenter tool."""

    default_sensitivity: float = Field(
        default=_SENSITIVITY_DEFAULT,
        ge=0.0,
        le=1.0,
        description="Default sensitivity (0.0–1.0) controlling what changes trigger a new segment.",
    )
    duration_tool: str = Field(
        default="vst_video_duration",
        description="Tool to retrieve total video duration.",
    )
    video_understanding_tool: str = Field(
        default="video_understanding",
        description="VLM tool used to detect change timestamps within each probe window.",
    )
    probe_window_seconds: float = Field(
        default=_PROBE_WINDOW_DEFAULT,
        gt=10.0,
        description=(
            "Duration of each probe window (seconds). The VLM analyzes the video in "
            "chunks of this length to find change points. Larger = fewer VLM calls "
            "but may miss rapid changes."
        ),
    )
    max_description_window_seconds: float = Field(
        default=_DESC_WINDOW_DEFAULT,
        gt=5.0,
        description=(
            "Maximum seconds of video passed to the VLM per description call. "
            "Caps the analysis window for very long segments so the VLM only "
            "sees the start of the new scene, not the entire segment."
        ),
    )

    model_config = {"extra": "forbid"}


class SceneSegmenterInput(BaseModel):
    """Input for the Scene Segmenter tool."""

    sensor_id: str = Field(..., min_length=1, description="The sensor ID or video filename in VST.")
    sensitivity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Change sensitivity (0.0–1.0). Controls what visual differences trigger "
            "a new EAD segment. Higher = more sensitive, more segments. "
            "Defaults to the profile config value when omitted."
        ),
    )
    captions_json: str | None = Field(
        default=None,
        description="Optional JSON output from caption_parser for dialogue context.",
    )

    model_config = {"extra": "forbid"}


@register_function(config_type=SceneSegmenterConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def scene_segmenter(config: SceneSegmenterConfig, builder: Builder) -> AsyncGenerator[FunctionInfo]:
    """Scene Segmenter — content-driven segmentation using VLM change detection."""

    async def _scene_segmenter(input: SceneSegmenterInput) -> str:
        """
        Segment a video into EAD processing units by detecting actual visual changes.

        Unlike fixed-duration chunking, this tool probes the video using the VLM to
        find where significant visual changes occur (scene cuts, new slides, text
        appearing, presenter changes, etc.) and creates segment boundaries only at
        those points.

        The opening scene (t=0) always produces a segment. Subsequent segments are
        created only when the VLM detects a meaningful change.

        Sensitivity controls the threshold:
          1.0 → detect any change (subtle text, camera angle)
          0.5 → detect clear changes (new slide, person, scene cut) — default
          0.0 → detect major transitions only (hard cuts, large content changes)

        Returns:
            JSON array of SceneSegment objects, each with:
              index, start_seconds, end_seconds, description_end_seconds, duration,
              caption_context
        """
        sensitivity = input.sensitivity if input.sensitivity is not None else config.default_sensitivity

        # Get total duration
        duration_tool = await builder.get_tool(config.duration_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        duration_result = await duration_tool.ainvoke(input={"sensor_id": input.sensor_id})
        if hasattr(duration_result, "duration_seconds"):
            total_duration = float(duration_result.duration_seconds)
        elif hasattr(duration_result, "duration"):
            total_duration = float(duration_result.duration)
        elif isinstance(duration_result, (int, float)):
            total_duration = float(duration_result)
        else:
            try:
                parsed = json.loads(str(duration_result))
                total_duration = float(
                    parsed.get("duration_seconds") or parsed.get("duration") or parsed.get("total_seconds", 0)
                    if isinstance(parsed, dict) else parsed
                )
            except (json.JSONDecodeError, ValueError):
                total_duration = float(str(duration_result))

        if total_duration <= 0:
            logger.warning(f"Video '{input.sensor_id}' has zero or negative duration")
            return json.dumps([])

        logger.info(
            f"Scene segmenter: sensor='{input.sensor_id}', duration={total_duration:.1f}s, "
            f"sensitivity={sensitivity:.2f}, probe_window={config.probe_window_seconds:.0f}s"
        )

        # Parse captions if provided
        captions: ParsedCaptions | None = None
        if input.captions_json:
            try:
                captions = ParsedCaptions.model_validate_json(input.captions_json)
                logger.info(f"Using {captions.cue_count} caption cues for change-detection context")
            except Exception as e:
                logger.warning(f"Failed to parse captions_json: {e}")

        # --- Step 1: VLM-based change detection ---
        vu_tool = await builder.get_tool(config.video_understanding_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        all_boundaries: list[float] = [0.0]  # always start with the opening

        probe_start = 0.0
        while probe_start < total_duration:
            probe_end = min(probe_start + config.probe_window_seconds, total_duration)
            clip_duration = probe_end - probe_start

            caption_context = captions.text_in_range(probe_start, probe_end) if captions else ""
            prompt = _build_change_detection_prompt(clip_duration, sensitivity, caption_context)

            try:
                result = await vu_tool.ainvoke(
                    input={
                        "sensor_id": input.sensor_id,
                        "start_timestamp": probe_start,
                        "end_timestamp": probe_end,
                        "user_prompt": prompt,
                        "vlm_reasoning": False,
                    }
                )
                raw = str(result).strip() if result else ""
                relative_ts = _parse_change_timestamps(raw)
                # Convert relative timestamps to absolute
                absolute_ts = [probe_start + t for t in relative_ts if t < clip_duration]
                all_boundaries.extend(absolute_ts)
                logger.info(
                    f"Probe [{probe_start:.0f}s–{probe_end:.0f}s]: "
                    f"detected {len(absolute_ts)} change(s) → {[f'{t:.1f}s' for t in absolute_ts]}"
                )
            except Exception as e:
                logger.warning(f"Change detection failed for [{probe_start:.0f}–{probe_end:.0f}s]: {e}")

            probe_start = probe_end

        # --- Step 2: Build segments from boundaries ---
        # Sort, deduplicate, remove too-close points, add video end
        all_boundaries = sorted(set(all_boundaries))
        all_boundaries = _merge_close_boundaries(all_boundaries, _MIN_SEGMENT_GAP)
        if not all_boundaries or all_boundaries[0] > 0.1:
            all_boundaries.insert(0, 0.0)
        all_boundaries.append(total_duration)

        segments: list[SceneSegment] = []
        for i in range(len(all_boundaries) - 1):
            start = round(all_boundaries[i], 3)
            end = round(all_boundaries[i + 1], 3)
            if end <= start:
                continue
            duration = round(end - start, 3)
            # Cap the description window — VLM only sees the start of each new scene
            desc_end = round(min(end, start + config.max_description_window_seconds), 3)
            cap_ctx = captions.text_in_range(start, desc_end) if captions else ""

            segments.append(SceneSegment(
                index=i,
                start_seconds=start,
                end_seconds=end,
                duration=duration,
                description_end_seconds=desc_end,
                caption_context=cap_ctx,
            ))

        logger.info(
            f"Content-driven segmentation complete: {len(segments)} segments "
            f"from {len(all_boundaries) - 1} boundaries "
            f"(sensitivity={sensitivity:.2f})"
        )
        return json.dumps([seg.model_dump() for seg in segments], indent=2)

    yield FunctionInfo.create(
        single_fn=_scene_segmenter,
        description=_scene_segmenter.__doc__,
        input_schema=SceneSegmenterInput,
        single_output_schema=str,
    )
