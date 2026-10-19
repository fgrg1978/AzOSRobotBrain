# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""VLM interface — connects to LM Studio for scene understanding."""

import base64
from openai import OpenAI

# Request timeout (seconds) for the synchronous OpenAI client. Callers that
# invoke `VisionPerception.describe()` from an asyncio context MUST wrap the
# call in `await asyncio.wait_for(asyncio.to_thread(...), timeout=
# VLM_TIMEOUT_S)` — this constructor-level timeout is the last-resort bound
# on the underlying HTTP call itself (LM Studio hang / dead backend), not a
# substitute for that wrapping. Image analysis is slower than a plain text
# completion, hence a larger budget than LLM_TIMEOUT_S.
VLM_TIMEOUT_S: float = 20.0


class VisionPerception:
    """Sends camera frames to a VLM via LM Studio's OpenAI-compatible API."""

    def __init__(self, host: str = "127.0.0.1", port: int = 1234, model: str = "smolvlm"):
        self.client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="not-needed",
            timeout=VLM_TIMEOUT_S,
        )
        self.model = model

    def describe(self, image_bytes: bytes, context: str = "") -> str:
        """Send an image to the VLM and get a scene description.

        Args:
            image_bytes: Raw image bytes (JPEG or PNG).
            context: Optional context about the robot's current task.

        Returns:
            Text description of what the VLM sees.
        """
        b64 = base64.b64encode(image_bytes).decode("ascii")

        prompt = (
            "You are the eyes of an autonomous robot. "
            "Describe what you see concisely. "
            "Focus on: obstacles, paths, people, doors, walls, floor type. "
            "Include estimated distances when possible. "
            "Be brief — max 3 sentences."
        )
        if context:
            prompt += f"\nCurrent task: {context}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
