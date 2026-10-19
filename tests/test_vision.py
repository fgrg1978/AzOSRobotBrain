# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for perception/vision.py — VLM interface.

Note: actual VLM calls require LM Studio running.
These tests verify the interface, not the model output.
"""

from perception.vision import VisionPerception


class TestVisionPerceptionInit:
    def test_default_init(self):
        v = VisionPerception()
        assert v.model == "smolvlm"

    def test_custom_init(self):
        v = VisionPerception(host="10.0.0.1", port=5678, model="llava-34b")
        assert v.model == "llava-34b"

    def test_client_created(self):
        v = VisionPerception()
        assert v.client is not None

    def test_base_url(self):
        v = VisionPerception(host="192.168.1.1", port=1234)
        assert "192.168.1.1" in str(v.client.base_url)
        assert "1234" in str(v.client.base_url)


class TestVisionPerceptionDescribe:
    def test_describe_signature(self):
        v = VisionPerception()
        assert callable(v.describe)

    def test_describe_has_context_param(self):
        import inspect

        sig = inspect.signature(VisionPerception.describe)
        params = list(sig.parameters.keys())
        assert "image_bytes" in params
        assert "context" in params
