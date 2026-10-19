# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Bearer-token authentication for the control-plane HTTP API.

Design:
  - A single shared secret (AZOS_BRAIN_CP_API_KEY env var) is expected.
  - Public endpoints (see _PUBLIC_PATHS) bypass auth.
  - All other endpoints require "Authorization: Bearer <token>".
  - Comparison is done with ``hmac.compare_digest`` to avoid timing attacks.

If neither the env var nor an explicit key is provided AND the
AZOS_BRAIN_ALLOW_INSECURE env var is set to "1", auth is disabled and a
warning is emitted.  This matches the pattern in api.py so operators have a
consistent mental model across both tiers.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Optional

logger = logging.getLogger("brain.control_plane.auth")

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

#: Env var that holds the control-plane bearer token.
ENV_CP_API_KEY: str = "AZOS_BRAIN_CP_API_KEY"

#: Env var for explicit insecure-mode opt-in (shared with data plane).
ENV_ALLOW_INSECURE: str = "AZOS_BRAIN_ALLOW_INSECURE"

#: Value that enables unauthenticated mode.
ALLOW_INSECURE_VALUE: str = "1"

#: HTTP header prefix for bearer auth.
_BEARER_PREFIX: str = "bearer "

#: Paths reachable without a token.
_PUBLIC_PATHS: frozenset[str] = frozenset({"/health", "/", ""})


class BearerAuth:
    """Validates Authorization: Bearer headers for the control plane.

    Instantiate once at startup; share the instance across request handlers.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._key: Optional[str] = os.environ.get(ENV_CP_API_KEY) or api_key or None
        self._insecure: bool = os.environ.get(ENV_ALLOW_INSECURE) == ALLOW_INSECURE_VALUE
        if self._key is None and not self._insecure:
            raise RuntimeError(
                "[ControlPlane/Auth] No authentication configured.  "
                f"Set {ENV_CP_API_KEY} to a secret token, or set "
                f"{ENV_ALLOW_INSECURE}={ALLOW_INSECURE_VALUE} to allow "
                "unauthenticated access (development only)."
            )
        if self._key is None:
            logger.warning(
                "[ControlPlane/Auth] Insecure mode enabled — "
                "all requests are accepted without a token."
            )

    def is_insecure(self) -> bool:
        """True when no token is configured, i.e. every request is accepted.

        Callers that bind a listening socket MUST consult this before choosing
        an interface: an unauthenticated control plane on 0.0.0.0 hands
        ``POST /v1/ota`` — a fleet-wide firmware push — to anyone on the LAN.
        api.py already downgrades its own insecure bind to loopback; this
        accessor exists so ``ControlPlaneServer.run`` can do the same without
        reaching into ``_key``.
        """
        return self._key is None

    def is_authorised(self, path: str, headers: dict[str, str]) -> bool:
        """Return True if *path* + *headers* are authorised.

        Public paths always pass.  When a key is configured, every non-public
        path must supply a matching Bearer token.  In explicit insecure mode
        (no key) all paths pass.
        """
        bare = path.split("?")[0].rstrip("/") or "/"
        if bare in _PUBLIC_PATHS:
            return True
        if self._key is None:
            # Insecure mode: all paths pass.
            return True
        auth = headers.get("authorization", "")
        if not auth.lower().startswith(_BEARER_PREFIX):
            return False
        provided = auth[len(_BEARER_PREFIX) :].strip()
        return hmac.compare_digest(provided, self._key)
