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
Caption file upload support for the EAD pipeline.

Provides an in-memory store and FastAPI routes so the UI can upload
.srt / .vtt caption files directly to the agent (bypassing VST, which
only handles video files).

Flow:
  1. UI calls POST /api/v1/videos with filename="subs.srt"
  2. video_upload_url detects the caption extension and returns
     http://AGENT:8000/api/v1/caption-upload/{key}
  3. Browser PUTs the file bytes to that URL
  4. This module stores the text content keyed by {key}
  5. When EAD runs, the agent fetches the caption via get_caption_content(key)
"""

import hashlib
import logging

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response

logger = logging.getLogger(__name__)

# Simple in-memory store: caption_key -> text content
# Survives for the lifetime of the agent process.
_caption_store: dict[str, str] = {}

CAPTION_EXTENSIONS = frozenset({".srt", ".vtt", ".SRT", ".VTT"})


def is_caption_file(filename: str) -> bool:
    """Return True if the filename has a known caption/subtitle extension."""
    if "." not in filename:
        return False
    ext = "." + filename.rsplit(".", 1)[-1]
    return ext in CAPTION_EXTENSIONS


def caption_key_for(filename: str) -> str:
    """Derive a stable, short key from the original filename."""
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def get_caption_content(key: str) -> str:
    """Return the stored caption text for the given key, or empty string."""
    return _caption_store.get(key, "")


def register_caption_routes(app: FastAPI) -> None:
    """Register caption upload / retrieval routes on the given FastAPI app."""

    @app.put("/api/v1/caption-upload/{caption_key}", include_in_schema=False)
    async def upload_caption(caption_key: str, request: Request) -> dict:
        """Receive a caption file from the browser and store its text content."""
        body = await request.body()
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            text = body.decode("latin-1", errors="replace")
        _caption_store[caption_key] = text
        logger.info(f"Caption stored: key={caption_key}, size={len(text)} chars")
        return {"status": "ok", "key": caption_key}

    @app.get("/api/v1/caption-content/{caption_key}", include_in_schema=False)
    async def get_caption(caption_key: str) -> Response:
        """Return the stored caption text for the given key."""
        content = _caption_store.get(caption_key, "")
        if not content:
            return Response(status_code=404, content="Caption not found")
        return Response(content=content, media_type="text/plain; charset=utf-8")

    logger.info("Registered caption upload/retrieval routes")
