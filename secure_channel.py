"""The brain↔kernel link's secure channel: a Noise session, nothing else.

Pairs byte for byte with `crates/core/crypto/src/secure_channel.rs` on the
robot. Every brain-protocol packet travels as one Noise transport message
(ChaCha20-Poly1305) of a session established per TCP connection, in both
directions. There is no second MAC inside the session, no MAC-only mode and
no plaintext mode with a key: replay is refused by the transport's message
counter inside a session and by the fresh keys of the next one across
sessions.

## Threat model

Without this layer the brain↔kernel TCP stream is plaintext on a presumed-
trusted LAN. Any peer on the segment can:
- Forge an ESTOP (`PKT_ESTOP=0x88`) and stop every robot.
- Replay an old "FORWARD 100" actuator command indefinitely.
- Inject malformed packets to crash the brain or kernel parsers.

## Key management

Pre-shared symmetric key (32 bytes raw) loaded from env var
`AZOS_BRAIN_LINK_KEY` as 64 hex chars (32 bytes); the robot holds the same
bytes in the reserved sector of its disk. Both static Noise keys derive from
it. If it is missing, `server.py` refuses to start. A different key on the
two sides fails the handshake and the brain is disconnected.

The brain's own static key (`AZOS_BRAIN_NOISE_SK`) is optional: the robot's
signed topology pins its public half in the `[peer.brain]` row (`noise_pk`)
and the link is then `Noise_IK` toward it; without a row it is `Noise_XX`
with the key derived from the link key.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import time
from typing import Callable, Optional

# ── Constants ─────────────────────────────────────────────────────────────

KEY_BYTES = 32


def load_link_key() -> Optional[bytes]:
    """Load the pre-shared link key.

    Returns 32 raw bytes, or `None` if no key is configured.
    """
    hex_key = os.environ.get("AZOS_BRAIN_LINK_KEY", "").strip()
    if not hex_key:
        return None
    try:
        raw = bytes.fromhex(hex_key)
    except ValueError:
        print("[SECCHAN] ERROR: AZOS_BRAIN_LINK_KEY is not valid hex")
        return None
    if len(raw) != KEY_BYTES:
        print(f"[SECCHAN] ERROR: link key must be {KEY_BYTES} bytes, got {len(raw)}")
        return None
    return raw


# ── Brain link secure channel (Noise) ─────────────────────────────────────
#
# Armed whenever a link key is configured (`protocol.enable_encrypt_link`).
# Pairs byte for byte with `crates/core/crypto/src/secure_channel.rs` on the robot; the fixed-key
# vectors in `tests/test_secure_channel.py` were produced by that Rust code.
#
# Roles: the robot dials the brain, so the robot is the TCP client and the
# Noise INITIATOR; this process is the TCP server and the Noise RESPONDER. (The
# AES-CTR + HMAC link this replaces had them the other way round.)
#
# Keys. The 32-byte link key is never used as a curve scalar; static keys derive
# from it (HKDF-SHA256, salt STATIC_SALT, one info string per side).
#   * companion static key pinned in the robot's signed topology (`[peer.companion]`,
#     `noise_pk`): Noise_IK. The brain's static private key is AZOS_BRAIN_NOISE_SK; the
#     robot's static (message 1) must be the key the link key derives.
#   * nothing pinned: Noise_XX with both statics derived from the link key; the
#     brain checks the robot's static after message 3 (the robot checks the
#     brain's after message 2).
#
# Wire:
#   robot -> brain   [pattern 'I' | 'X'] once, before frame 1
#   frame            [length u16 BE][body]
#   IK   1 robot -> brain  e, es, s, ss     2 brain -> robot  e, ee, se
#   XX   1 robot -> brain  e                2 brain -> robot  e, ee, s, es
#        3 robot -> brain  s, se
#   transport frame  [length u16 BE][ChaCha20-Poly1305 ciphertext || tag]
# Handshake payloads are empty; the prologue is b"azos-companion-link-v2" + the
# pattern byte. A transport frame's AD is its two length bytes, its nonce the
# CipherState counter (the wire carries none: TCP is in order). Any violation
# is terminal and closes the connection; there is no REJECT. An empty
# plaintext (a bare tag) is the rekey signal: sender and receiver then replace
# that direction's key by Noise REKEY.

try:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
        X25519PublicKey,
    )
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    _HAS_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover — only hit if requirements not installed
    _HAS_CRYPTOGRAPHY = False


PSK_BYTES = KEY_BYTES
PATTERN_IK = 0x49  # 'I'
PATTERN_XX = 0x58  # 'X'
LINK_VERSION = 2
PROLOGUE_LABEL = b"azos-companion-link-v2"
STATIC_SALT = b"azos-companion-link-v2 static"
KERNEL_STATIC_INFO = b"azos companion-link kernel static v2"
COMPANION_STATIC_INFO = b"azos companion-link companion static v2"

DH_LEN = 32
HASH_LEN = 32
TAG_LEN = 16
FRAME_HEAD = 2
FRAME_OVERHEAD = FRAME_HEAD + TAG_LEN
HANDSHAKE_MSG_MAX = 128
SESSION_ID_BYTES = 16
#: Largest message `seal_message` accepts (and the receive limit).
MAX_MESSAGE_BYTES = 16 * 1024

#: Messages per key generation, per direction, including the rekey message.
REKEY_MAX_RECORDS = 1 << 20
#: Plaintext bytes per key generation, per direction (RFC-0019: 1 GiB).
REKEY_MAX_BYTES = 1 << 30
#: Wall-clock rekey interval (RFC-0019: 1 hour).
REKEY_INTERVAL_S = 3600.0
REKEY_MIN_RECORDS = 2
REKEY_MIN_BYTES = MAX_MESSAGE_BYTES
REKEY_INTERVAL_MIN_S = 5.0

_PROTOCOL_NAMES = {
    PATTERN_IK: b"Noise_IK_25519_ChaChaPoly_SHA256",
    PATTERN_XX: b"Noise_XX_25519_ChaChaPoly_SHA256",
}
# message patterns; tokens per message (IK: <- s ...; the pre-message)
_PATTERNS: dict[int, tuple[tuple[str, ...], ...]] = {
    PATTERN_IK: (("e", "es", "s", "ss"), ("e", "ee", "se")),
    PATTERN_XX: (("e",), ("e", "ee", "s", "es"), ("s", "se")),
}
_MAX_NONCE = (1 << 64) - 1


class NoiseError(Exception):
    """A Noise primitive refused: bad tag, bad key, wrong state, bad length."""


class LinkViolation(PermissionError):
    """A terminal record-layer or handshake violation: the session is over and
    the caller closes the connection."""


def rekey_interval_s(requested: "float | None") -> float:
    """The wall-clock rekey interval for a configured request: a request may
    only SHORTEN the RFC-0019 hour (the robot's `rekey_interval_secs`)."""
    if requested is None:
        return REKEY_INTERVAL_S
    return min(max(float(requested), REKEY_INTERVAL_MIN_S), REKEY_INTERVAL_S)


# ── Primitives ──────────────────────────────────────────────────────────────


def _hmac(key: bytes, *parts: bytes) -> bytes:
    h = hmac.new(key, digestmod=hashlib.sha256)
    for p in parts:
        h.update(p)
    return h.digest()


def hkdf(salt: bytes, ikm: bytes, info: bytes, length: int = 32) -> bytes:
    """HKDF-SHA256 (RFC 5869), as `azos_crypto::hkdf::hkdf`."""
    prk = _hmac(salt, ikm)
    okm = b""
    t = b""
    i = 1
    while len(okm) < length:
        t = _hmac(prk, t, info, bytes([i]))
        okm += t
        i += 1
    return okm[:length]


def public_key(sk: bytes) -> bytes:
    """The X25519 public key of private key `sk`."""
    return X25519PrivateKey.from_private_bytes(sk).public_key().public_bytes_raw()


def _dh(sk: bytes, pk: bytes) -> bytes:
    """X25519, rejecting an all-zero result (a low-order peer key)."""
    try:
        return X25519PrivateKey.from_private_bytes(sk).exchange(
            X25519PublicKey.from_public_bytes(pk)
        )
    except ValueError as e:
        raise NoiseError("bad DH") from e


def kernel_static_key(psk: bytes) -> bytes:
    """The robot's static private key for the brain link."""
    return hkdf(STATIC_SALT, psk, KERNEL_STATIC_INFO)


def companion_static_key(psk: bytes) -> bytes:
    """The companion's derived static private key (the XX identity)."""
    return hkdf(STATIC_SALT, psk, COMPANION_STATIC_INFO)


def load_noise_static() -> "bytes | None":
    """The brain's own static private key (IK mode): env `AZOS_BRAIN_NOISE_SK`,
    64 hex digits. `None` when unset (XX with the derived static)."""
    text = os.environ.get("AZOS_BRAIN_NOISE_SK", "").strip()
    if not text:
        return None
    try:
        key = bytes.fromhex(text)
    except ValueError:
        raise ValueError("AZOS_BRAIN_NOISE_SK is not valid hex") from None
    if len(key) != DH_LEN or key == bytes(DH_LEN):
        raise ValueError("AZOS_BRAIN_NOISE_SK must be 32 non-zero bytes (64 hex digits)")
    return key


# ── Noise (revision 34): CipherState, SymmetricState, HandshakeState ───────


class CipherState:
    """A key and a 64-bit message counter."""

    def __init__(self, k: "bytes | None" = None) -> None:
        self.k = k
        self.n = 0

    @property
    def has_key(self) -> bool:
        return self.k is not None

    def _nonce(self) -> bytes:
        return bytes(4) + self.n.to_bytes(8, "little")

    def encrypt(self, ad: bytes, plaintext: bytes) -> bytes:
        if self.k is None:
            return plaintext
        if self.n == _MAX_NONCE:
            raise NoiseError("nonce exhausted")
        out = ChaCha20Poly1305(self.k).encrypt(self._nonce(), plaintext, ad)
        self.n += 1
        return out

    def decrypt(self, ad: bytes, ciphertext: bytes) -> bytes:
        """The counter advances only on success, so a forged or replayed
        packet does not desynchronise the stream."""
        if self.k is None:
            return ciphertext
        if self.n == _MAX_NONCE:
            raise NoiseError("nonce exhausted")
        try:
            out = ChaCha20Poly1305(self.k).decrypt(self._nonce(), ciphertext, ad)
        except InvalidTag:
            raise NoiseError("decrypt") from None
        self.n += 1
        return out

    def rekey(self) -> None:
        """Noise REKEY: k = first 32 bytes of ENCRYPT(k, 2^64 - 1, "", zeros).
        The counter keeps running."""
        assert self.k is not None
        nonce = bytes(4) + _MAX_NONCE.to_bytes(8, "little")
        self.k = ChaCha20Poly1305(self.k).encrypt(nonce, bytes(32), b"")[:32]

    def wipe(self) -> None:
        self.k = None


def _hkdf2(ck: bytes, ikm: bytes) -> tuple[bytes, bytes]:
    temp = _hmac(ck, ikm)
    o1 = _hmac(temp, b"\x01")
    o2 = _hmac(temp, o1, b"\x02")
    return o1, o2


class HandshakeState:
    """A Noise handshake (IK or XX), as `azos_crypto::noise::HandshakeState`."""

    def __init__(
        self,
        pattern: int,
        initiator: bool,
        s: bytes,
        rs: "bytes | None",
        prologue: bytes,
    ) -> None:
        if pattern not in _PATTERNS:
            raise NoiseError("unknown pattern")
        # IK needs the responder's static up front for the initiator only.
        if (rs is not None) != (pattern == PATTERN_IK and initiator):
            raise NoiseError("remote static does not fit the pattern")
        name = _PROTOCOL_NAMES[pattern]
        self._h = name.ljust(HASH_LEN, b"\x00") if len(name) <= HASH_LEN else hashlib.sha256(name).digest()
        self._ck = self._h
        self._cs = CipherState()
        self._pattern = pattern
        self._initiator = initiator
        self._s = s
        self._s_pub = public_key(s)
        self._e: "bytes | None" = None
        self._rs = rs
        self._re: "bytes | None" = None
        self._idx = 0
        self._poisoned = False
        self._mix_hash(prologue)
        if pattern == PATTERN_IK:
            # Pre-message `<- s`: the responder's static, known to both.
            if initiator:
                assert rs is not None
                self._mix_hash(rs)
            else:
                self._mix_hash(self._s_pub)

    # -- symmetric state --
    def _mix_hash(self, data: bytes) -> None:
        self._h = hashlib.sha256(self._h + data).digest()

    def _mix_key(self, ikm: bytes) -> None:
        self._ck, tk = _hkdf2(self._ck, ikm)
        self._cs = CipherState(tk)

    def _encrypt_and_hash(self, plaintext: bytes) -> bytes:
        ct = self._cs.encrypt(self._h, plaintext)
        self._mix_hash(ct)
        return ct

    def _decrypt_and_hash(self, ciphertext: bytes) -> bytes:
        pt = self._cs.decrypt(self._h, ciphertext)
        self._mix_hash(ciphertext)
        return pt

    # -- accessors --
    @property
    def local_static(self) -> bytes:
        return self._s_pub

    @property
    def remote_static(self) -> "bytes | None":
        return self._rs

    @property
    def handshake_hash(self) -> bytes:
        return self._h

    @property
    def is_finished(self) -> bool:
        return not self._poisoned and self._idx == len(_PATTERNS[self._pattern])

    @property
    def is_my_turn(self) -> bool:
        return (
            not self._poisoned
            and self._idx < len(_PATTERNS[self._pattern])
            and ((self._idx % 2 == 0) == self._initiator)
        )

    def _dh_token(self, tok: str) -> bytes:
        # `es` is DH(initiator e, responder s); `se` is DH(initiator s, responder e).
        init = self._initiator
        if tok == "ee":
            k, p = self._e, self._re
        elif tok == "ss":
            k, p = self._s, self._rs
        elif tok == "es":
            k, p = (self._e, self._rs) if init else (self._s, self._re)
        elif tok == "se":
            k, p = (self._s, self._re) if init else (self._e, self._rs)
        else:
            raise NoiseError("bad token")
        if k is None or p is None:
            raise NoiseError("DH before the keys exist")
        return _dh(k, p)

    def write_message(self, eph: bytes, payload: bytes = b"") -> bytes:
        """The next handshake message. `eph` is a fresh ephemeral private key."""
        if not self.is_my_turn:
            raise NoiseError("not our turn")
        try:
            out = bytearray()
            for tok in _PATTERNS[self._pattern][self._idx]:
                if tok == "e":
                    self._e = eph
                    pub = public_key(eph)
                    out += pub
                    self._mix_hash(pub)
                elif tok == "s":
                    out += self._encrypt_and_hash(self._s_pub)
                else:
                    self._mix_key(self._dh_token(tok))
            out += self._encrypt_and_hash(payload)
        except Exception:
            self._poisoned = True
            raise
        self._idx += 1
        return bytes(out)

    def read_message(self, msg: bytes) -> bytes:
        """Process the peer's next message; returns its decrypted payload."""
        if self._poisoned or self._idx >= len(_PATTERNS[self._pattern]) or self.is_my_turn:
            raise NoiseError("not the peer's turn")
        try:
            off = 0
            for tok in _PATTERNS[self._pattern][self._idx]:
                if tok == "e":
                    if len(msg) < off + DH_LEN:
                        raise NoiseError("truncated")
                    self._re = msg[off : off + DH_LEN]
                    self._mix_hash(self._re)
                    off += DH_LEN
                elif tok == "s":
                    n = DH_LEN + (TAG_LEN if self._cs.has_key else 0)
                    if len(msg) < off + n:
                        raise NoiseError("truncated")
                    self._rs = self._decrypt_and_hash(msg[off : off + n])
                    off += n
                else:
                    self._mix_key(self._dh_token(tok))
            payload = self._decrypt_and_hash(msg[off:])
        except Exception:
            self._poisoned = True
            raise
        self._idx += 1
        return payload

    def finish(self) -> tuple[CipherState, CipherState, bytes, bytes]:
        """(send, recv, handshake hash, remote static) once the pattern is done."""
        if not self.is_finished or self._rs is None:
            raise NoiseError("handshake not finished")
        k1, k2 = _hkdf2(self._ck, b"")
        c1, c2 = CipherState(k1), CipherState(k2)
        send, recv = (c1, c2) if self._initiator else (c2, c1)
        return send, recv, self._h, self._rs


def _prologue(pattern: int) -> bytes:
    return PROLOGUE_LABEL + bytes([pattern])


def handshake_frame_size(head: bytes) -> int:
    """Wire size (prefix included) of the handshake frame whose first two
    bytes are `head`. Raises ValueError for a length the handshake cannot
    carry (read no body then)."""
    if len(head) < FRAME_HEAD:
        raise ValueError("short header")
    body = int.from_bytes(head[:FRAME_HEAD], "big")
    if body == 0 or body > HANDSHAKE_MSG_MAX:
        raise ValueError(f"handshake frame length {body}")
    return FRAME_HEAD + body


def _frame(body: bytes) -> bytes:
    return len(body).to_bytes(FRAME_HEAD, "big") + body


class SecureChannel:
    """The brain link's Noise channel (RFC-0019, Noise revision).

    Responder (the brain)::

        sc = SecureChannel(psk, is_initiator=False, static_key=load_noise_static())
        sc.begin(pattern_byte)                       # the byte before frame 1
        reply = sc.handle_handshake(frame1)          # a reply frame, or None
        ...                                          # XX: one more frame
        sc.is_established

    Initiator (the robot's role; used by tests and the peer tools)::

        sc = SecureChannel(psk, is_initiator=True, remote_pinned=brain_pk_or_None)
        first = sc.start_handshake()                 # pattern byte + frame 1
        out = sc.handle_handshake(frame2)            # frame 3 (XX) or None

    Established: `seal_message` / `open_frame` / `open_message`.
    """

    # Frame kinds returned by `open_frame`.
    RECORD_DATA = 0
    RECORD_REKEY = 2

    def __init__(
        self,
        psk: bytes,
        is_initiator: bool = False,
        *,
        static_key: "bytes | None" = None,
        remote_pinned: "bytes | None" = None,
        clock: Callable[[], float] = time.monotonic,
        _testing_eph_priv: "bytes | None" = None,
    ) -> None:
        if not _HAS_CRYPTOGRAPHY:
            raise RuntimeError(
                "secure_channel.SecureChannel needs the 'cryptography' package — "
                "run `pip install -r requirements.txt`"
            )
        if len(psk) != PSK_BYTES:
            raise ValueError(f"PSK must be {PSK_BYTES} bytes")
        if _testing_eph_priv is not None and len(_testing_eph_priv) != DH_LEN:
            raise ValueError("testing eph priv must be 32 bytes")
        self._psk = psk
        self._is_initiator = is_initiator
        self._static_key = static_key
        self._remote_pinned = remote_pinned
        self._clock = clock
        # `_testing_eph_priv` is for cross-side vectors ONLY.
        self._eph_override = _testing_eph_priv
        self._hs: "HandshakeState | None" = None
        self._pattern: "int | None" = None
        self._expect_remote: "bytes | None" = None
        self._sent_pattern = False
        self._send: "CipherState | None" = None
        self._recv: "CipherState | None" = None
        self._hash: "bytes | None" = None
        self._remote: "bytes | None" = None
        self._established = False
        self._failed = False

        self._tx_generation = 0
        self._rx_generation = 0
        self._tx_msgs = 0
        self._tx_bytes = 0
        self._rx_msgs = 0
        self._rx_bytes = 0
        self._rekey_records = REKEY_MAX_RECORDS
        self._rekey_bytes = REKEY_MAX_BYTES
        self._rekey_interval_s = REKEY_INTERVAL_S
        self._tx_gen_started = 0.0
        self._rekey_requested = False

        #: Frames refused by the AEAD (parallel to Receiver's drop counters).
        self.drops_decrypt = 0

        if is_initiator:
            self._init_initiator()

    # ── Handshake ──────────────────────────────────────────────────────

    def _eph(self) -> bytes:
        if self._eph_override is not None:
            return self._eph_override
        return os.urandom(DH_LEN)

    def _init_initiator(self) -> None:
        sk = kernel_static_key(self._psk)
        if self._remote_pinned is not None:
            self._pattern = PATTERN_IK
            rs: "bytes | None" = self._remote_pinned
        else:
            self._pattern = PATTERN_XX
            rs = None
            self._expect_remote = public_key(companion_static_key(self._psk))
        self._hs = HandshakeState(self._pattern, True, sk, rs, _prologue(self._pattern))

    def begin(self, pattern_byte: int) -> None:
        """Responder: the pattern byte the robot sent before its first frame.
        Raises `LinkViolation` for one this brain cannot serve (the caller
        closes the connection)."""
        if self._is_initiator or self._hs is not None or self._failed:
            raise RuntimeError("begin() on a channel that already started")
        if pattern_byte == PATTERN_IK:
            if self._static_key is None:
                self._fail()
                raise LinkViolation("Noise_IK announced but no AZOS_BRAIN_NOISE_SK is set")
            sk = self._static_key
        elif pattern_byte == PATTERN_XX:
            sk = companion_static_key(self._psk)
        else:
            self._fail()
            raise LinkViolation(f"unknown handshake pattern byte {pattern_byte:#x}")
        self._pattern = pattern_byte
        # The robot's static must be the one the link key derives; checked as
        # soon as it is known (IK: after message 1, before any reply).
        self._expect_remote = public_key(kernel_static_key(self._psk))
        self._hs = HandshakeState(pattern_byte, False, sk, None, _prologue(pattern_byte))

    def start_handshake(self) -> bytes:
        """Initiator step 1: the pattern byte and the first frame."""
        if not self._is_initiator or self._sent_pattern or self._hs is None:
            raise RuntimeError("start_handshake from invalid state")
        self._sent_pattern = True
        assert self._pattern is not None
        return bytes([self._pattern]) + self._write_frame()

    def _write_frame(self) -> bytes:
        assert self._hs is not None
        try:
            return _frame(self._hs.write_message(self._eph()))
        except NoiseError as e:
            self._fail()
            raise LinkViolation(f"handshake: {e} (link v{LINK_VERSION} expected)") from e

    def handle_handshake(self, frame: bytes) -> "bytes | None":
        """Process the peer's handshake frame (prefix included); returns the
        frame to send next, or `None` when there is nothing to send. Raises
        `LinkViolation` (terminal) on any failure."""
        if self._hs is None or self._failed or self._established:
            raise RuntimeError("handle_handshake from invalid state")
        try:
            if len(frame) < FRAME_HEAD or int.from_bytes(frame[:FRAME_HEAD], "big") != len(frame) - FRAME_HEAD:
                raise NoiseError("bad handshake frame length")
            if len(frame) - FRAME_HEAD > HANDSHAKE_MSG_MAX:
                raise NoiseError("handshake frame too long")
            if self._hs.read_message(frame[FRAME_HEAD:]):
                raise NoiseError("unexpected handshake payload")
            if self._expect_remote is not None and self._hs.remote_static is not None:
                if not hmac.compare_digest(self._expect_remote, self._hs.remote_static):
                    raise NoiseError("the peer's static key is not the one the link key derives")
            out: "bytes | None" = None
            if self._hs.is_my_turn:
                out = _frame(self._hs.write_message(self._eph()))
            if self._hs.is_finished:
                self._establish()
            return out
        except NoiseError as e:
            self._fail()
            raise LinkViolation(f"handshake: {e} (link v{LINK_VERSION} expected)") from e

    def _establish(self) -> None:
        assert self._hs is not None
        send, recv, h, rs = self._hs.finish()
        self._send, self._recv, self._hash, self._remote = send, recv, h, rs
        self._hs = None
        self._established = True
        self._tx_gen_started = self._clock()

    def _fail(self) -> None:
        self._failed = True
        self._established = False
        self._hs = None
        for cs in (self._send, self._recv):
            if cs is not None:
                cs.wipe()
        self._send = self._recv = None
        self._hash = None

    # ── State ──────────────────────────────────────────────────────────

    @property
    def is_established(self) -> bool:
        return self._established

    @property
    def is_rejected(self) -> bool:
        return self._failed

    @property
    def session_id(self) -> "bytes | None":
        """The first 16 bytes of the handshake hash (None before the end)."""
        return None if self._hash is None else self._hash[:SESSION_ID_BYTES]

    @property
    def remote_static(self) -> "bytes | None":
        return self._remote

    @property
    def tx_generation(self) -> int:
        return self._tx_generation

    @property
    def rx_generation(self) -> int:
        return self._rx_generation

    def request_rekey(self) -> None:
        """Make the next `seal_message` start with a rekey frame."""
        self._rekey_requested = True

    def set_rekey_limits(
        self, records: int, nbytes: int, interval_s: "float | None" = None
    ) -> None:
        """Lower the per-generation limits (clamped like the Rust side). Both
        ends of a link must use the same message/byte values: the receive side
        refuses a peer that passes them."""
        self._rekey_records = min(max(records, REKEY_MIN_RECORDS), REKEY_MAX_RECORDS)
        self._rekey_bytes = min(max(nbytes, REKEY_MIN_BYTES), REKEY_MAX_BYTES)
        if interval_s is not None:
            self._rekey_interval_s = interval_s

    def close(self) -> None:
        """End the session locally; every key reference is dropped."""
        self._fail()

    # ── Transport ──────────────────────────────────────────────────────

    @staticmethod
    def _seal_frame(cs: CipherState, plain: bytes) -> bytes:
        head = (len(plain) + TAG_LEN).to_bytes(FRAME_HEAD, "big")
        return head + cs.encrypt(head, plain)

    def seal_message(self, msg: bytes) -> bytes:
        """Seal one message as one frame, with a rekey frame in front when the
        generation limits, the wall-clock interval or `request_rekey()` call
        for one."""
        if not self._established or self._send is None:
            raise RuntimeError("seal before handshake established")
        if not msg:
            raise ValueError("an empty message is the rekey signal, never data")
        if len(msg) > MAX_MESSAGE_BYTES:
            raise ValueError(f"message too large ({len(msg)} > {MAX_MESSAGE_BYTES})")
        rekey = (
            self._rekey_requested
            or self._tx_msgs + 2 > self._rekey_records
            or self._tx_bytes + len(msg) > self._rekey_bytes
            or self._clock() - self._tx_gen_started >= self._rekey_interval_s
        )
        out = b""
        if rekey:
            out += self._seal_frame(self._send, b"")
            self._send.rekey()
            self._tx_generation += 1
            self._tx_msgs = 0
            self._tx_bytes = 0
            self._tx_gen_started = self._clock()
            self._rekey_requested = False
        out += self._seal_frame(self._send, msg)
        self._tx_msgs += 1
        self._tx_bytes += len(msg)
        return out

    def frame_size(self, head: bytes) -> int:
        """Wire size (prefix included) of the transport frame whose first two
        bytes are `head`, so a reader never reads a body that is too long or
        too short to be a frame. A bad length is terminal: `LinkViolation`."""
        if len(head) < FRAME_HEAD:
            raise ValueError("short header")
        body = int.from_bytes(head[:FRAME_HEAD], "big")
        if body < TAG_LEN or body > MAX_MESSAGE_BYTES + TAG_LEN:
            self._fail()
            raise LinkViolation(f"bad frame length {body}")
        return FRAME_HEAD + body

    def open_frame(self, frame: bytes) -> tuple[int, bytes]:
        """Open one whole frame; returns `(RECORD_DATA, plaintext)` or
        `(RECORD_REKEY, b"")`. Any failure is terminal: `LinkViolation`."""
        if not self._established or self._recv is None:
            raise LinkViolation("link not established")
        total = self.frame_size(frame[:FRAME_HEAD])
        if len(frame) != total:
            self._fail()
            raise LinkViolation("frame length does not match its prefix")
        assert self._recv is not None
        try:
            plain = self._recv.decrypt(frame[:FRAME_HEAD], frame[FRAME_HEAD:])
        except NoiseError:
            self.drops_decrypt += 1
            self._fail()
            raise LinkViolation("bad frame (forged, replayed, reordered or dropped)") from None
        if not plain:
            self._recv.rekey()
            self._rx_generation += 1
            self._rx_msgs = 0
            self._rx_bytes = 0
            return self.RECORD_REKEY, b""
        # The sender rekeys before its limit-th message of a generation, so a
        # data frame that would be that message is a peer that skipped it.
        if self._rx_msgs + 1 >= self._rekey_records or self._rx_bytes + len(plain) > self._rekey_bytes:
            self._fail()
            raise LinkViolation("peer passed the rekey limit without a rekey")
        self._rx_msgs += 1
        self._rx_bytes += len(plain)
        return self.RECORD_DATA, plain

    def open_message(self, data: bytes) -> bytes:
        """Open one sealed message: optional rekey frames, then exactly one
        data frame, nothing after it. (A multi-stream control frame carries
        one sealed message.) Raises `LinkViolation` otherwise."""
        off = 0
        message: "bytes | None" = None
        while off < len(data):
            if message is not None:
                self._fail()
                raise LinkViolation("bytes after the message's data frame")
            total = self.frame_size(data[off : off + FRAME_HEAD])
            if off + total > len(data):
                self._fail()
                raise LinkViolation("torn frame in a sealed message")
            kind, plain = self.open_frame(data[off : off + total])
            off += total
            if kind == self.RECORD_DATA:
                message = plain
        if message is None:
            self._fail()
            raise LinkViolation("no data frame in a sealed message")
        return message
