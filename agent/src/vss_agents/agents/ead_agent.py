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
EAD Agent — end-to-end Extended Audio Description pipeline orchestrator.

This deterministic agent drives the full EAD generation pipeline for a single
uploaded video.  It does not use an LLM for routing decisions — every step is
pre-defined and executed in order:

  1. HITL   — confirm processing parameters with the user before the long run
  2. Parse  — if caption content is provided, parse it via caption_parser
  3. Segment — divide the video into timed segments via scene_segmenter
  4. Describe — generate a VLM-based EAD cue per segment via visual_describer
  5. Chapter — group cues into titled, summarised chapters via chapter_generator
  6. Enrich  — assemble a JSON-LD VideoObject metadata document via metadata_enricher
  7. Format  — render WebVTT descriptions, WebVTT chapters, and/or TTML via ead_formatter

Progress is streamed to the UI via AgentMessageChunk after each step so the
user sees live status rather than a silent wait.

Configuration mirrors the EAD tool suite in config.yml; all tool names are
FunctionRef fields so they can be remapped per deployment.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import AsyncGenerator
from typing import Literal

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.interactive import HumanPromptText
from nat.data_models.interactive import InteractionResponse
from pydantic import BaseModel
from pydantic import Field

from vss_agents.agents.data_models import AgentMessageChunk
from vss_agents.agents.data_models import AgentMessageChunkType
from vss_agents.agents.data_models import AgentOutput
from vss_agents.data_models.ead import sensitivity_to_chunk_duration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------

_DEFAULT_HITL_CONFIRMATION = """\
**EAD Generation Configuration**

Please review the parameters below before starting. Processing long videos can
take several minutes.

**Video:** `{sensor_id}`
**Sensitivity:** `{sensitivity}` ({chunk_duration:.0f}s segments → ~{n_segments} segments estimated)
**Output formats:** `{output_format}`
**VLM Reasoning:** `{vlm_reasoning}`
**Captions provided:** `{has_captions}`
{title_line}

**Options:**
- Press **Submit** (empty) → Confirm and start EAD generation
- Type `/edit` → Change parameters
- Type `/cancel` → Cancel

Enter your choice or press Submit to proceed:"""


async def _prompt_user(prompt_text: str, required: bool = False, placeholder: str = "") -> str:
    nat_context = Context.get()
    human_prompt = HumanPromptText(text=prompt_text, required=required, placeholder=placeholder)
    response: InteractionResponse = await nat_context.user_interaction_manager.prompt_user_input(human_prompt)
    return str(response.content.text).strip()


# ---------------------------------------------------------------------------
# Config and input models
# ---------------------------------------------------------------------------


class EADAgentConfig(FunctionBaseConfig, name="ead_agent"):
    """Configuration for the EAD Agent."""

    # Tool references — names must match functions registered in config.yml
    caption_parser_tool: FunctionRef = Field(
        default="caption_parser",
        description="Tool to parse VTT/SRT caption files.",
    )
    scene_segmenter_tool: FunctionRef = Field(
        default="scene_segmenter",
        description="Tool to divide the video into timed segments.",
    )
    visual_describer_tool: FunctionRef = Field(
        default="visual_describer",
        description="Tool to generate EAD cues per segment via VLM.",
    )
    chapter_generator_tool: FunctionRef = Field(
        default="chapter_generator",
        description="Tool to group EAD cues into chapters via LLM.",
    )
    metadata_enricher_tool: FunctionRef = Field(
        default="metadata_enricher",
        description="Tool to assemble the JSON-LD metadata document.",
    )
    ead_formatter_tool: FunctionRef = Field(
        default="ead_formatter",
        description="Tool to render WebVTT and TTML output files.",
    )

    # Defaults
    default_sensitivity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default segmentation sensitivity (0.0–1.0).",
    )
    default_output_format: Literal["webvtt", "ttml", "all"] = Field(
        default="all",
        description="Default output format for ead_formatter.",
    )
    default_vlm_reasoning: bool = Field(
        default=False,
        description="Default VLM reasoning mode for visual_describer.",
    )

    # HITL
    hitl_enabled: bool = Field(
        default=True,
        description="Whether to show a HITL confirmation step before processing.",
    )
    hitl_confirmation_template: str = Field(
        default=_DEFAULT_HITL_CONFIRMATION,
        description="HITL confirmation prompt template. "
        "Supports: {sensor_id}, {sensitivity}, {chunk_duration}, {n_segments}, "
        "{output_format}, {vlm_reasoning}, {has_captions}, {title_line}.",
    )

    model_config = {"extra": "forbid"}


class EADAgentInput(BaseModel):
    """Input for the EAD Agent."""

    sensor_id: str = Field(
        ...,
        min_length=1,
        description=(
            "VST sensor ID or video filename. This is the video for which EAD files "
            "will be generated. Must already be uploaded to VST."
        ),
    )
    sensitivity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Segmentation sensitivity (0.0–1.0). Controls how granular the scene "
            "segmentation and description detail level are:\n"
            "  1.0 (maximum) →  ~5s segments — every subtle visual change\n"
            "  0.75 (high)   → ~10s segments\n"
            "  0.5  (medium) → ~24s segments  (default)\n"
            "  0.25 (low)    → ~55s segments\n"
            "  0.0  (minimum)→ ~120s segments — chapter-level only\n"
            "Defaults to the profile config value when omitted."
        ),
    )
    captions_content: str | None = Field(
        default=None,
        description=(
            "Raw content of a WebVTT (.vtt) or SubRip (.srt) caption/subtitle file "
            "to use as dialogue context for the VLM during description generation. "
            "Providing captions generally improves description quality and accuracy. "
            "Paste the full file text here."
        ),
    )
    captions_format: Literal["srt", "vtt", "auto"] = Field(
        default="auto",
        description="Caption file format. Use 'auto' to detect automatically.",
    )
    output_format: Literal["webvtt", "ttml", "all"] | None = Field(
        default=None,
        description=(
            "Output file format(s) to generate:\n"
            "  'webvtt' — WebVTT descriptions + chapters files\n"
            "  'ttml'   — TTML file only\n"
            "  'all'    — all formats (default)\n"
            "Defaults to the profile config value when omitted."
        ),
    )
    video_title: str | None = Field(
        default=None,
        description=(
            "Human-readable video title used in file headers and the metadata document. "
            "Defaults to the sensor_id when omitted."
        ),
    )
    vlm_reasoning: bool | None = Field(
        default=None,
        description=(
            "Enable VLM reasoning mode (cosmos-reason models). "
            "Produces richer descriptions at the cost of increased processing time. "
            "Defaults to the profile config value when omitted."
        ),
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------


@register_function(config_type=EADAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def ead_agent(config: EADAgentConfig, builder: Builder) -> AsyncGenerator[FunctionInfo]:
    """EAD Agent — orchestrates the full Extended Audio Description generation pipeline."""

    # Pre-load all tools at startup to fail fast on misconfiguration
    caption_parser_tool = await builder.get_tool(config.caption_parser_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    scene_segmenter_tool = await builder.get_tool(config.scene_segmenter_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    visual_describer_tool = await builder.get_tool(
        config.visual_describer_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    chapter_generator_tool = await builder.get_tool(
        config.chapter_generator_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    metadata_enricher_tool = await builder.get_tool(
        config.metadata_enricher_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    ead_formatter_tool = await builder.get_tool(config.ead_formatter_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    logger.info("EAD Agent initialised — all pipeline tools loaded")

    async def _run_pipeline(input: EADAgentInput) -> AsyncGenerator[AgentMessageChunk]:
        """
        Generate Extended Audio Description files for an uploaded video.

        This agent runs the full EAD pipeline end-to-end:
          1. Optionally parse a provided VTT/SRT caption file for context
          2. Segment the video into timed chunks (sensitivity-controlled)
          3. Generate a VLM-based visual description per segment
          4. Group segments into titled chapters
          5. Assemble a JSON-LD VideoObject metadata document
          6. Render WebVTT and/or TTML output files

        Returns streamed progress updates and a final AgentOutput containing
        all generated file contents and the metadata document.
        """
        start_time = time.time()
        thread_id = ContextState.get().conversation_id.get()

        # Resolve parameters from input or config defaults
        sensitivity = input.sensitivity if input.sensitivity is not None else config.default_sensitivity
        output_format = input.output_format or config.default_output_format
        vlm_reasoning = input.vlm_reasoning if input.vlm_reasoning is not None else config.default_vlm_reasoning
        title = input.video_title or input.sensor_id
        has_captions = bool(input.captions_content and input.captions_content.strip())

        logger.info(
            f"[EAD Agent] thread={thread_id}, sensor='{input.sensor_id}', "
            f"sensitivity={sensitivity:.2f}, format={output_format}, reasoning={vlm_reasoning}"
        )

        # ------------------------------------------------------------------ #
        # Step 0: HITL confirmation
        # ------------------------------------------------------------------ #
        if config.hitl_enabled:
            chunk_duration = sensitivity_to_chunk_duration(sensitivity)
            # We don't know duration yet, so estimate based on a typical long video (3600s)
            # This is just for the confirmation display — actual segmentation is accurate
            n_segments_est = "?"
            title_line = f"**Title:** `{title}`" if input.video_title else ""

            prompt = config.hitl_confirmation_template.format(
                sensor_id=input.sensor_id,
                sensitivity=f"{sensitivity:.2f}",
                chunk_duration=chunk_duration,
                n_segments=n_segments_est,
                output_format=output_format,
                vlm_reasoning=str(vlm_reasoning),
                has_captions="Yes" if has_captions else "No",
                title_line=title_line,
            )

            user_choice = await _prompt_user(prompt, required=False, placeholder="/edit, /cancel, or Submit")
            choice = user_choice.lower()

            if choice == "/cancel":
                logger.info("EAD generation cancelled by user")
                cancelled = AgentOutput(
                    messages=["EAD generation cancelled."],
                    status="success",
                    metadata={"sensor_id": input.sensor_id},
                )
                yield AgentMessageChunk(type=AgentMessageChunkType.FINAL, content=cancelled.model_dump_json())
                return

            if choice == "/edit":
                # Collect sensitivity override
                sens_input = await _prompt_user(
                    "Enter new sensitivity (0.0–1.0) or press Submit to keep current value:",
                    required=False,
                    placeholder=str(sensitivity),
                )
                if sens_input:
                    try:
                        sensitivity = max(0.0, min(1.0, float(sens_input)))
                    except ValueError:
                        pass

                # Collect output format override
                fmt_input = await _prompt_user(
                    "Enter output format (webvtt / ttml / all) or press Submit to keep current value:",
                    required=False,
                    placeholder=output_format,
                )
                if fmt_input and fmt_input in ("webvtt", "ttml", "all"):
                    output_format = fmt_input  # type: ignore[assignment]

                # Collect title override
                title_input = await _prompt_user(
                    "Enter video title or press Submit to keep current value:",
                    required=False,
                    placeholder=title,
                )
                if title_input:
                    title = title_input

                logger.info(f"Parameters updated: sensitivity={sensitivity}, format={output_format}, title={title}")

        # ------------------------------------------------------------------ #
        # Step 1: Parse captions
        # ------------------------------------------------------------------ #
        captions_json: str | None = None
        if has_captions:
            yield AgentMessageChunk(
                type=AgentMessageChunkType.TOOL_CALL,
                content="Tool: caption_parser\nParsing caption file…",
            )
            try:
                captions_result = await caption_parser_tool.ainvoke(
                    input={
                        "caption_content": input.captions_content,
                        "format": input.captions_format,
                    }
                )
                captions_json = str(captions_result)
                parsed = json.loads(captions_json)
                cue_count = parsed.get("cue_count", 0)
                yield AgentMessageChunk(
                    type=AgentMessageChunkType.THOUGHT,
                    content=f"✓ Parsed {cue_count} caption cues ({parsed.get('format', '').upper()}) — "
                    f"will be used as dialogue context for VLM descriptions.",
                )
                logger.info(f"Caption parsing complete: {cue_count} cues")
            except Exception as e:
                logger.warning(f"Caption parsing failed — continuing without captions: {e}")
                yield AgentMessageChunk(
                    type=AgentMessageChunkType.THOUGHT,
                    content=f"⚠ Caption parsing failed ({e}) — proceeding without caption context.",
                )
                captions_json = None

        # ------------------------------------------------------------------ #
        # Step 2: Scene segmentation
        # ------------------------------------------------------------------ #
        chunk_duration = sensitivity_to_chunk_duration(sensitivity)
        yield AgentMessageChunk(
            type=AgentMessageChunkType.TOOL_CALL,
            content=(
                f"Tool: scene_segmenter\n"
                f"Segmenting '{input.sensor_id}' "
                f"(sensitivity={sensitivity:.2f} → ~{chunk_duration:.0f}s chunks)…"
            ),
        )

        seg_input: dict = {
            "sensor_id": input.sensor_id,
            "sensitivity": sensitivity,
        }
        if captions_json:
            seg_input["captions_json"] = captions_json

        try:
            segments_result = await scene_segmenter_tool.ainvoke(input=seg_input)
            segments_json = str(segments_result)
            segments_list = json.loads(segments_json)
            n_segments = len(segments_list)

            if n_segments == 0:
                raise RuntimeError("No segments returned — is the video empty or duration zero?")

            total_duration = segments_list[-1]["end_seconds"] if segments_list else 0
            yield AgentMessageChunk(
                type=AgentMessageChunkType.THOUGHT,
                content=(
                    f"✓ Video divided into **{n_segments} segments** "
                    f"({total_duration:.0f}s total, ~{chunk_duration:.0f}s each)."
                ),
            )
            logger.info(f"Segmentation complete: {n_segments} segments, {total_duration:.0f}s")
        except Exception as e:
            logger.error(f"Scene segmentation failed: {e}")
            error_out = AgentOutput(
                messages=[f"EAD generation failed during scene segmentation: {e}"],
                status="error",
                error_message=str(e),
                metadata={"sensor_id": input.sensor_id, "step": "scene_segmenter"},
            )
            yield AgentMessageChunk(type=AgentMessageChunkType.FINAL, content=error_out.model_dump_json())
            return

        # ------------------------------------------------------------------ #
        # Step 3: Visual description (VLM per segment)
        # ------------------------------------------------------------------ #
        yield AgentMessageChunk(
            type=AgentMessageChunkType.TOOL_CALL,
            content=(
                f"Tool: visual_describer\n"
                f"Generating EAD descriptions for {n_segments} segments "
                f"(VLM reasoning={'on' if vlm_reasoning else 'off'})…\n"
                f"This may take a few minutes for long videos."
            ),
        )

        try:
            desc_result = await visual_describer_tool.ainvoke(
                input={
                    "sensor_id": input.sensor_id,
                    "segments_json": segments_json,
                    "sensitivity": sensitivity,
                    "vlm_reasoning": vlm_reasoning,
                }
            )
            ead_cues_json = str(desc_result)
            cues_list = json.loads(ead_cues_json)
            n_cues = len(cues_list)

            yield AgentMessageChunk(
                type=AgentMessageChunkType.THOUGHT,
                content=f"✓ Generated **{n_cues} EAD description cues** via VLM.",
            )
            logger.info(f"Visual description complete: {n_cues} cues")
        except Exception as e:
            logger.error(f"Visual description failed: {e}")
            error_out = AgentOutput(
                messages=[f"EAD generation failed during visual description: {e}"],
                status="error",
                error_message=str(e),
                metadata={"sensor_id": input.sensor_id, "step": "visual_describer", "segments": n_segments},
            )
            yield AgentMessageChunk(type=AgentMessageChunkType.FINAL, content=error_out.model_dump_json())
            return

        # ------------------------------------------------------------------ #
        # Step 4: Chapter generation
        # ------------------------------------------------------------------ #
        yield AgentMessageChunk(
            type=AgentMessageChunkType.TOOL_CALL,
            content="Tool: chapter_generator\nGrouping cues into chapters…",
        )

        try:
            ch_result = await chapter_generator_tool.ainvoke(
                input={
                    "ead_cues_json": ead_cues_json,
                    "video_title": title,
                }
            )
            chapters_json = str(ch_result)
            chapters_list = json.loads(chapters_json)
            n_chapters = len(chapters_list)

            chapter_titles = [f"  {i+1}. {ch['title']}" for i, ch in enumerate(chapters_list)]
            yield AgentMessageChunk(
                type=AgentMessageChunkType.THOUGHT,
                content=(
                    f"✓ Organised into **{n_chapters} chapters**:\n"
                    + "\n".join(chapter_titles)
                ),
            )
            logger.info(f"Chapter generation complete: {n_chapters} chapters")
        except Exception as e:
            logger.warning(f"Chapter generation failed — continuing with empty chapter list: {e}")
            chapters_json = "[]"
            chapters_list = []
            n_chapters = 0
            yield AgentMessageChunk(
                type=AgentMessageChunkType.THOUGHT,
                content=f"⚠ Chapter generation failed ({e}) — EAD files will be produced without chapter navigation.",
            )

        # ------------------------------------------------------------------ #
        # Step 5: Metadata enrichment
        # ------------------------------------------------------------------ #
        yield AgentMessageChunk(
            type=AgentMessageChunkType.TOOL_CALL,
            content="Tool: metadata_enricher\nAssembling JSON-LD metadata document…",
        )

        try:
            meta_result = await metadata_enricher_tool.ainvoke(
                input={
                    "sensor_id": input.sensor_id,
                    "ead_cues_json": ead_cues_json,
                    "chapters_json": chapters_json,
                    "video_title": title,
                    "video_duration_seconds": total_duration,
                }
            )
            metadata_json = str(meta_result)
            yield AgentMessageChunk(
                type=AgentMessageChunkType.THOUGHT,
                content="✓ JSON-LD VideoObject metadata document assembled.",
            )
            logger.info("Metadata enrichment complete")
        except Exception as e:
            logger.warning(f"Metadata enrichment failed — continuing: {e}")
            metadata_json = "{}"
            yield AgentMessageChunk(
                type=AgentMessageChunkType.THOUGHT,
                content=f"⚠ Metadata enrichment failed ({e}).",
            )

        # ------------------------------------------------------------------ #
        # Step 6: EAD formatting
        # ------------------------------------------------------------------ #
        yield AgentMessageChunk(
            type=AgentMessageChunkType.TOOL_CALL,
            content=f"Tool: ead_formatter\nRendering {output_format.upper()} output files…",
        )

        try:
            fmt_result = await ead_formatter_tool.ainvoke(
                input={
                    "ead_cues_json": ead_cues_json,
                    "chapters_json": chapters_json,
                    "video_title": title,
                    "output_format": output_format,
                }
            )
            fmt_data = json.loads(str(fmt_result))
            yield AgentMessageChunk(
                type=AgentMessageChunkType.THOUGHT,
                content="✓ Output files rendered.",
            )
            logger.info(f"EAD formatting complete: format={output_format}")
        except Exception as e:
            logger.error(f"EAD formatting failed: {e}")
            error_out = AgentOutput(
                messages=[f"EAD generation failed during file formatting: {e}"],
                status="error",
                error_message=str(e),
                metadata={"sensor_id": input.sensor_id, "step": "ead_formatter"},
            )
            yield AgentMessageChunk(type=AgentMessageChunkType.FINAL, content=error_out.model_dump_json())
            return

        # ------------------------------------------------------------------ #
        # Final output
        # ------------------------------------------------------------------ #
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Build summary message
        summary_lines = [
            f"## EAD Generation Complete — *{title}*\n",
            f"- **Segments:** {n_cues} EAD cues across {total_duration:.0f}s",
            f"- **Chapters:** {n_chapters}" + (
                " (" + ", ".join(ch["title"] for ch in chapters_list[:3])
                + ("…" if n_chapters > 3 else "") + ")"
                if n_chapters else " (none)"
            ),
            f"- **Sensitivity:** {sensitivity:.2f} (~{chunk_duration:.0f}s segments)",
            f"- **Captions used:** {'Yes' if has_captions else 'No'}",
            f"- **Processing time:** {elapsed_ms / 1000:.1f}s\n",
        ]

        # List generated files
        files_lines = ["### Generated Files\n"]
        if fmt_data.get("webvtt_descriptions"):
            files_lines.append(f"- `{fmt_data['webvtt_descriptions_filename']}` — WebVTT descriptions (EAD cues)")
        if fmt_data.get("webvtt_chapters"):
            files_lines.append(f"- `{fmt_data['webvtt_chapters_filename']}` — WebVTT chapters (navigation)")
        if fmt_data.get("ttml"):
            files_lines.append(f"- `{fmt_data['ttml_filename']}` — TTML (broadcast/streaming)")
        files_lines.append("- JSON-LD VideoObject metadata document\n")

        # First few EAD cues as preview
        preview_lines = ["### EAD Preview (first 3 cues)\n"]
        for cue in cues_list[:3]:
            start = cue["start_seconds"]
            end = cue["end_seconds"]
            m_s, s_s = divmod(int(start), 60)
            m_e, s_e = divmod(int(end), 60)
            preview_lines.append(
                f"**[{m_s:02d}:{s_s:02d} → {m_e:02d}:{s_e:02d}]** {cue['description']}\n"
            )

        full_summary = "\n".join(summary_lines + files_lines + preview_lines)

        # Side effects — embed file content for download / display
        side_effects: dict = {}
        if fmt_data.get("webvtt_descriptions"):
            side_effects["webvtt_descriptions"] = fmt_data["webvtt_descriptions"]
            side_effects["webvtt_descriptions_filename"] = fmt_data["webvtt_descriptions_filename"]
        if fmt_data.get("webvtt_chapters"):
            side_effects["webvtt_chapters"] = fmt_data["webvtt_chapters"]
            side_effects["webvtt_chapters_filename"] = fmt_data["webvtt_chapters_filename"]
        if fmt_data.get("ttml"):
            side_effects["ttml"] = fmt_data["ttml"]
            side_effects["ttml_filename"] = fmt_data["ttml_filename"]
        side_effects["metadata_json_ld"] = metadata_json

        agent_output = AgentOutput(
            messages=[full_summary],
            side_effects=side_effects,
            status="success",
            metadata={
                "sensor_id": input.sensor_id,
                "video_title": title,
                "sensitivity": sensitivity,
                "chunk_duration_seconds": chunk_duration,
                "n_segments": n_segments,
                "n_cues": n_cues,
                "n_chapters": n_chapters,
                "output_format": output_format,
                "vlm_reasoning": vlm_reasoning,
                "has_captions": has_captions,
                "total_duration_seconds": total_duration,
                "processing_time_ms": elapsed_ms,
            },
        )

        yield AgentMessageChunk(type=AgentMessageChunkType.FINAL, content=agent_output.model_dump_json())

    yield FunctionInfo.create(
        stream_fn=_run_pipeline,
        description=(
            "Generate Extended Audio Description (EAD) files for an uploaded video. "
            "Runs the full EAD pipeline: scene segmentation → VLM visual description per segment → "
            "chapter generation → JSON-LD metadata enrichment → WebVTT/TTML file rendering. "
            "Call this agent when the user asks to generate audio descriptions, EAD files, "
            "accessibility descriptions, or a chapter manifest for a video. "
            "Optionally accepts a VTT/SRT caption file for improved description context. "
            "Sensitivity (0.0–1.0) controls description granularity."
        ),
        input_schema=EADAgentInput,
        stream_output_schema=AgentMessageChunk,
    )
