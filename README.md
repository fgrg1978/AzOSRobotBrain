# AzOSRobotBrain

AzOSRobotBrain is the host-side planner and data plane for
[AzOS](https://github.com/fgrg1978/AzOS), an operating system on a hybrid
kernel in Rust for riscv64 and aarch64. Robotics is one deployment of AzOS,
not its definition: the AzOS robot domain links a planner link, and this
repository is the program on the other end of it.

It is a Python server that runs on a macOS or Linux host. It receives sensor
and camera data from AzOS over the brain link, decides what the robot should
do with vision and language models, and sends actuator, mode and waypoint
commands back, checking every sensor packet against a per-robot safety
profile on the way.

## What it does

- **Brain link.** A compact binary protocol: a `BR` magic, a type byte, a
  little-endian length, up to 1400 bytes of payload and a CRC-8/MAXIM.
  Packet types and shared limits match the AzOS kernel.
- **Authenticated, encrypted link.** With a pre-shared key in
  `AZOS_COMPANION_LINK_KEY`, every connection runs a Noise handshake
  (`Noise_IK` or `Noise_XX`, X25519 / ChaCha20-Poly1305 / SHA-256) whose static
  keys derive from the same secret, then every brain-protocol frame travels as
  one Noise transport message with in-session rekeying. There is no second MAC
  and no MAC-only mode: a forged, replayed or reordered frame fails the AEAD
  and ends the session, and the next session has fresh keys. The robot dials
  the brain and is the Noise initiator; this process is the responder.
  `Noise_IK` when the robot's signed topology pins the brain's static public
  key (the `noise_pk` of its `[peer.brain]` row) and the brain holds the
  private half in `AZOS_BRAIN_NOISE_SK` (64 hex digits): a leaked link key
  alone then cannot impersonate the brain. Without a pinned key the handshake
  is `Noise_XX` with keys derived from the link key, authenticated by that key
  and not trust-on-first-use. Encryption is on whenever a key is configured
  and there is no switch to turn it off. The wire format is described in
  the header of `secure_channel.py`, and `tests/test_secure_channel.py` pins
  it byte for byte against the Rust implementation in AzOS.
- **No unauthenticated robot link.** The server refuses to start without a
  link key, and the simulated robots (`sitl.py`, `tools/sitl/`) speak the same
  Noise session with the same `AZOS_COMPANION_LINK_KEY`. The HTTP API refuses to
  start without `AZOS_BRAIN_API_KEY` unless `AZOS_BRAIN_ALLOW_INSECURE` is set.
- **Safety checks independent of the planner.** Each sensor packet goes
  through a safety profile for its robot type (wheeled, drone, humanoid,
  Ackermann) whatever the planner is doing; a violation interrupts the
  running plan and sends a stop command (a drone gets its own emergency
  response).
- **Perception and planning.** A VLM and an LLM served by LM Studio, or any
  OpenAI-compatible endpoint, describe the scene and pick the next action; a
  task planner turns free text into a plan of skills from a per-robot-type
  catalogue, and a skill runner executes it. Model answers can be cached,
  with canned fallbacks for when the model server is unreachable.
- **Cameras.** RTSP cameras are watched with a background-subtraction motion
  detector and a blob tracker, and only confirmed motion goes to the VLM.
- **Fleet.** A fleet registry with heartbeats, broadcast and targeted
  commands, and OTA pushes whose images are checked against an Ed25519
  signature before they are sent.
- **Interfaces.** An HTTP API (status, mode, tasks, fleet, OTA, Prometheus
  metrics), a web dashboard, a Telegram bot, alerts (Pushover, Telegram,
  webhook) and a MAVLink v1 bridge to PX4 or ArduPilot.
- **Simulation.** A software-in-the-loop robot (wheeled, drone, Ackermann,
  humanoid) that speaks the same protocol, so the brain runs without
  hardware.

## Status

**Early development. Not for production use, and not for safety-critical
deployments.** It has been run against simulated robots, and the link to
AzOS has been exercised under QEMU with the stub brain in `tools/`; nothing
has run on a real robot. No certification has been sought or
obtained.

## Run and test

Host: Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q        # the test suite
python -m mypy             # strict type check of the modules listed in pyproject.toml
```

Run the server (edit `config.yaml` first: robot type, model endpoint, ports):

```bash
export AZOS_COMPANION_LINK_KEY=<64 hex chars: the same 32 bytes as the kernel's LINK.KEY>
export AZOS_BRAIN_API_KEY=<token for the HTTP API>
export AZOS_BRAIN_NOISE_SK=<64 hex chars: the private key whose public half the robot pins; optional>
python server.py
```

Run a simulated robot instead of hardware:

```bash
python sitl.py --type wheeled --scenario forward_10m
```

## Layout

```
server.py, protocol.py, secure_channel.py   brain link, dispatch, link security
api.py, dashboard/, metrics.py              HTTP API, web UI, metrics
perception/       VLM client, motion detection, tracking, camera fusion
planner/          decisions, modes, skills, task planning, missions
policy/           skill-to-actuator translation and safety profiles
executor/         plan execution
fleet.py, fleet_ota.py                      fleet registry and OTA
control_plane/, data_plane/                 split deployment (tools/start_split.py)
sitl.py, tools/sitl/                        simulated robots
tools/            stub brain, benchmarks, command-line client
tests/            the test suite
```

## License and contributing

Dual-licensed, `Apache-2.0 OR GPL-2.0-only` at your option: see
[`LICENSE`](LICENSE), [`LICENSES/`](LICENSES) and [`NOTICE`](NOTICE) (two
Python dependencies are Apache-2.0 only). Contributions go under
[`CLA.md`](CLA.md). Versions published before this change were MIT-licensed.
