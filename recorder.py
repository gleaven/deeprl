"""Training session recording and replay for drone RL.

Records the exact WebSocket messages the frontend consumes, then replays them
at adjustable speed through the same /ws/course endpoint. No new dependencies —
uses stdlib json + zlib only.

Storage: zlib-compressed JSONL in Redis (deeprl:drone:recording:*).
"""

import json
import logging
import time
import zlib

logger = logging.getLogger("deeprl.recorder")

# Message kind codes (compact keys in the JSONL)
K_STATE = 0
K_STATS = 1
K_EVENT = 2
K_LAYOUT = 3

# Only record every Nth state frame (20 Hz → 2 Hz)
STATE_DECIMATION = 10


class TrainingRecorder:
    """Non-blocking recording of WS messages for later replay.

    Call record_*() from the training loop — each just appends a tuple to an
    in-memory list. Serialization only happens on stop().
    """

    def __init__(self):
        self._active = False
        self._timeline: list[tuple[int, int, dict]] = []  # (ms, kind, msg)
        self._start_time = 0.0
        self._frame_counter = 0
        self._name = ""

    @property
    def active(self) -> bool:
        return self._active

    def start(self, name: str):
        self._active = True
        self._timeline = []
        self._start_time = time.monotonic()
        self._frame_counter = 0
        self._name = name
        logger.info(f"Recording started: {name}")

    def record_state(self, msg: dict):
        """Called from _publish_snapshot (~20 Hz). Decimates to ~2 Hz."""
        if not self._active:
            return
        self._frame_counter += 1
        if self._frame_counter % STATE_DECIMATION != 0:
            return
        self._append(K_STATE, msg)

    def record_stats(self, msg: dict):
        if not self._active:
            return
        self._append(K_STATS, msg)

    def record_event(self, msg: dict):
        if not self._active:
            return
        self._append(K_EVENT, msg)

    def record_layout(self, msg: dict):
        if not self._active:
            return
        self._append(K_LAYOUT, msg)

    def _append(self, kind: int, msg: dict):
        elapsed_ms = int((time.monotonic() - self._start_time) * 1000)
        self._timeline.append((elapsed_ms, kind, msg))

    def stop(self) -> tuple[dict, bytes]:
        """Stop recording, return (metadata_dict, compressed_bytes)."""
        self._active = False
        duration = time.monotonic() - self._start_time

        n_states = sum(1 for t in self._timeline if t[1] == K_STATE)
        n_stats = sum(1 for t in self._timeline if t[1] == K_STATS)
        last_gen = 0
        last_level = 1
        for _, k, d in self._timeline:
            if k == K_STATS:
                last_gen = d.get("generation", last_gen)
                last_level = d.get("curriculum_level", last_level)

        metadata = {
            "name": self._name,
            "recorded_at": time.time(),
            "duration_seconds": round(duration, 1),
            "total_entries": len(self._timeline),
            "state_frames": n_states,
            "generations": last_gen,
            "final_level": last_level,
        }

        # Serialize: JSONL → zlib
        lines = []
        for t_ms, kind, msg in self._timeline:
            lines.append(json.dumps({"t": t_ms, "k": kind, "d": msg},
                                    separators=(',', ':')))
        raw = "\n".join(lines).encode("utf-8")
        compressed = zlib.compress(raw, level=6)

        logger.info(
            f"Recording stopped: {self._name} — "
            f"{len(self._timeline)} entries, {n_stats} gens, "
            f"{len(raw) / 1e6:.1f} MB raw → {len(compressed) / 1e6:.1f} MB compressed"
        )

        self._timeline = []
        return metadata, compressed


class TrainingReplayer:
    """Plays back a recorded training session at adjustable speed.

    Call get_pending_messages() at ~20 Hz from the broadcast loop — it returns
    all messages whose timeline timestamp has been reached at the current speed.
    """

    def __init__(self, compressed_blob: bytes, metadata: dict):
        raw = zlib.decompress(compressed_blob)
        self._timeline: list[tuple[int, int, dict]] = []
        for line in raw.decode("utf-8").split("\n"):
            if not line:
                continue
            entry = json.loads(line)
            self._timeline.append((entry["t"], entry["k"], entry["d"]))

        self.metadata = metadata
        self.total_entries = len(self._timeline)
        self._playing = False
        self._paused = False
        self._speed = 50.0
        self._position = 0
        self._start_time = 0.0
        self._start_ms = 0  # timeline ms at start of current play segment

        # Cache the last course_layout for immediate scene building on connect
        self._last_layout: dict | None = None
        for _, k, d in self._timeline:
            if k == K_LAYOUT:
                self._last_layout = d

        logger.info(
            f"Replayer loaded: {metadata.get('name', '?')} — "
            f"{self.total_entries} entries, "
            f"{metadata.get('generations', 0)} gens"
        )

    def play(self, speed: float = 50.0):
        self._speed = max(1.0, speed)
        self._playing = True
        self._paused = False
        self._start_time = time.monotonic()
        if self._position < self.total_entries:
            self._start_ms = self._timeline[self._position][0]

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False
        self._start_time = time.monotonic()
        if self._position < self.total_entries:
            self._start_ms = self._timeline[self._position][0]

    def seek(self, fraction: float):
        """Seek to a fraction (0.0–1.0) of the recording."""
        target_idx = int(fraction * self.total_entries)
        self._position = max(0, min(target_idx, self.total_entries - 1))
        self._start_time = time.monotonic()
        if self._position < self.total_entries:
            self._start_ms = self._timeline[self._position][0]

    def set_speed(self, speed: float):
        if self._position < self.total_entries:
            self._start_ms = self._timeline[self._position][0]
        self._start_time = time.monotonic()
        self._speed = max(1.0, speed)

    def get_pending_messages(self) -> list[dict]:
        """Called at ~20 Hz. Returns messages due at current playback speed."""
        if not self._playing or self._paused or self._position >= self.total_entries:
            return []

        elapsed_real = time.monotonic() - self._start_time
        elapsed_recording_ms = self._start_ms + (elapsed_real * 1000 * self._speed)

        messages = []
        while self._position < self.total_entries:
            ts_ms, kind, msg = self._timeline[self._position]
            if ts_ms > elapsed_recording_ms:
                break
            messages.append(msg)
            self._position += 1

        return messages

    @property
    def progress(self) -> float:
        return self._position / max(1, self.total_entries)

    @property
    def finished(self) -> bool:
        return self._position >= self.total_entries

    @property
    def current_generation(self) -> int:
        """Walk backward to find the last stats message before current position."""
        for i in range(min(self._position, self.total_entries - 1), -1, -1):
            _, k, d = self._timeline[i]
            if k == K_STATS:
                return d.get("generation", 0)
        return 0

    def get_status(self) -> dict:
        return {
            "type": "replay_status",
            "playing": self._playing,
            "paused": self._paused,
            "speed": self._speed,
            "progress": round(self.progress, 4),
            "finished": self.finished,
            "recording_name": self.metadata.get("name", ""),
            "total_generations": self.metadata.get("generations", 0),
            "current_generation": self.current_generation,
            "duration_seconds": self.metadata.get("duration_seconds", 0),
        }

    def get_last_layout(self) -> dict | None:
        """Return the last course_layout in the recording for scene init."""
        return self._last_layout
