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
Caption Parser Tool — parses VTT or SRT subtitle/caption files into structured cues.

Accepts raw caption file content (as a string) and returns a JSON-serialised
ParsedCaptions object. The output is consumed by scene_segmenter and
visual_describer to inject dialogue context into VLM prompts.

Supports:
  - WebVTT (.vtt): WEBVTT header, optional NOTE/STYLE/REGION blocks, dot milliseconds
  - SubRip (.srt): numeric cue indices, comma milliseconds, no header
  - Auto-detection based on content
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Literal

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel
from pydantic import Field

from vss_agents.data_models.ead import CaptionCue
from vss_agents.data_models.ead import ParsedCaptions

logger = logging.getLogger(__name__)

# Regex patterns
_SRT_TIMESTAMP = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2}),(\d{3})"
)
_VTT_TIMESTAMP = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})"
)
_VTT_TIMESTAMP_SHORT = re.compile(
    r"(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2})\.(\d{3})"
)
_HTML_TAG = re.compile(r"<[^>]+>")
_VTT_CUE_SETTING = re.compile(r"\s+(align|line|position|region|size|vertical):[^\s]+")


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _strip_formatting(text: str) -> str:
    """Remove HTML tags and WebVTT cue settings from caption text."""
    text = _HTML_TAG.sub("", text)
    return text.strip()


def _parse_vtt(content: str) -> list[CaptionCue]:
    """Parse WebVTT content into a list of CaptionCue objects."""
    cues: list[CaptionCue] = []
    cue_index = 1

    # Normalise line endings and split into blocks
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    i = 0
    # Skip WEBVTT header line and any header metadata
    while i < len(lines) and not lines[i].startswith("WEBVTT"):
        i += 1
    i += 1  # skip "WEBVTT" line

    while i < len(lines):
        line = lines[i].strip()

        # Skip blank lines, NOTE blocks, STYLE blocks, REGION blocks
        if not line:
            i += 1
            continue
        if line.startswith("NOTE") or line.startswith("STYLE") or line.startswith("REGION"):
            # Skip until next blank line
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue

        # Try to match a timestamp line (with or without preceding cue id)
        m = _VTT_TIMESTAMP.match(line)
        if not m:
            # Might be a cue identifier — advance and try next line
            i += 1
            if i < len(lines):
                m = _VTT_TIMESTAMP.match(lines[i].strip())
            if not m:
                continue

        # Strip cue settings from the timestamp line (everything after the --> part)
        start_s = _ts_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end_s = _ts_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        i += 1

        # Collect text lines until blank line or EOF
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(_strip_formatting(lines[i]))
            i += 1

        text = " ".join(t for t in text_lines if t)
        if text and end_s > start_s:
            cues.append(CaptionCue(index=cue_index, start_seconds=start_s, end_seconds=end_s, text=text))
            cue_index += 1

    return cues


def _parse_srt(content: str) -> list[CaptionCue]:
    """Parse SubRip (SRT) content into a list of CaptionCue objects."""
    cues: list[CaptionCue] = []

    # Split into blocks separated by blank lines
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n").strip())

    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            continue

        # Find the timestamp line (may be preceded by a numeric cue index)
        ts_line_idx = None
        cue_num = None
        for idx, line in enumerate(lines):
            m = _SRT_TIMESTAMP.match(line)
            if m:
                ts_line_idx = idx
                if idx > 0 and lines[idx - 1].isdigit():
                    cue_num = int(lines[idx - 1])
                break

        if ts_line_idx is None:
            continue

        m = _SRT_TIMESTAMP.match(lines[ts_line_idx])
        if not m:
            continue

        start_s = _ts_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end_s = _ts_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        text_lines = lines[ts_line_idx + 1 :]
        text = " ".join(_strip_formatting(l) for l in text_lines if l)

        if text and end_s > start_s:
            index = cue_num if cue_num is not None else len(cues) + 1
            cues.append(CaptionCue(index=index, start_seconds=start_s, end_seconds=end_s, text=text))

    return cues


def _detect_format(content: str) -> Literal["srt", "vtt"]:
    """Detect whether content is WebVTT or SRT."""
    stripped = content.lstrip()
    if stripped.startswith("WEBVTT"):
        return "vtt"
    # VTT files sometimes have a BOM or whitespace before WEBVTT
    if "WEBVTT" in stripped[:20]:
        return "vtt"
    return "srt"


def parse_captions(content: str, fmt: Literal["srt", "vtt", "auto"] = "auto") -> ParsedCaptions:
    """Parse caption content and return a ParsedCaptions object."""
    detected = _detect_format(content) if fmt == "auto" else fmt
    if detected == "vtt":
        cues = _parse_vtt(content)
    else:
        cues = _parse_srt(content)

    logger.info(f"Parsed {len(cues)} cues from {detected.upper()} content")
    return ParsedCaptions(format=detected, cue_count=len(cues), cues=cues)


# ---------------------------------------------------------------------------
# NAT tool registration
# ---------------------------------------------------------------------------


class CaptionParserConfig(FunctionBaseConfig, name="caption_parser"):
    """Configuration for the Caption Parser tool (no configurable parameters)."""
    pass


class CaptionParserInput(BaseModel):
    """Input for the Caption Parser tool."""

    caption_content: str = Field(
        ...,
        description=(
            "Raw content of a WebVTT (.vtt) or SubRip (.srt) caption/subtitle file. "
            "Paste the full file text here."
        ),
        min_length=10,
    )
    format: Literal["srt", "vtt", "auto"] = Field(
        default="auto",
        description=(
            "Caption format. Use 'auto' (default) to detect automatically from the content. "
            "Specify 'srt' or 'vtt' to force a specific parser."
        ),
    )

    model_config = {"extra": "forbid"}


@register_function(config_type=CaptionParserConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def caption_parser(config: CaptionParserConfig, builder: Builder) -> AsyncGenerator[FunctionInfo]:
    """Caption Parser — converts VTT/SRT file content to structured caption cues."""

    async def _caption_parser(input: CaptionParserInput) -> str:
        """
        Parse a WebVTT or SRT caption/subtitle file into structured cues.

        Use this tool when the user provides a caption or subtitle file alongside
        a video to improve the quality of Extended Audio Descriptions. The parsed
        output (JSON) should be passed as captions_json to scene_segmenter and
        visual_describer.

        Returns:
            JSON string containing a ParsedCaptions object with format, cue_count,
            and a list of CaptionCue objects (each with index, start_seconds,
            end_seconds, text).
        """
        try:
            result = parse_captions(input.caption_content, input.format)
            return result.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Caption parsing failed: {e}")
            raise RuntimeError(f"Failed to parse caption file: {e}") from e

    yield FunctionInfo.create(
        single_fn=_caption_parser,
        description=_caption_parser.__doc__,
        input_schema=CaptionParserInput,
        single_output_schema=str,
    )
