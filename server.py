"""DEEPRL — Deep Reinforcement Learning: Soccer Arena + Drone Course."""

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager

import httpx
import numpy as np
import redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ppo import PPOTrainer

# ── Configuration ──────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL", "redis://demo-redis:6379/10")
SERVICEROUTER_URL = os.environ.get("SERVICEROUTER_URL", "http://demo-servicerouter:8080")
N_ENVS = int(os.environ.get("N_ENVS", "256"))
TRAINING_SPEED = int(os.environ.get("TRAINING_SPEED", "1"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("deeprl")

# ── Global State ───────────────────────────────────────────────
_soccer_trainer: PPOTrainer | None = None
_drone_trainer: PPOTrainer | None = None
_soccer_ws: set[WebSocket] = set()
_drone_ws: set[WebSocket] = set()
_redis_client: redis.Redis | None = None

# Replay state
from recorder import TrainingReplayer
_drone_replayer: TrainingReplayer | None = None


# ── Human Flight Motor Mixer ──────────────────────────────────
# Lazy-initialized on first use (needs drone_environment imports)
_HOVER_BASE: float | None = None


def _mix_human_controls(throttle: float, pitch: float, roll: float, yaw: float) -> np.ndarray:
    """Convert human-friendly controls to 4 motor actions in [-1, 1].

    X-config mixer: FR(CW), FL(CCW), BL(CW), BR(CCW).
    """
    global _HOVER_BASE
    if _HOVER_BASE is None:
        from drone_environment import HOVER_OMEGA, MAX_OMEGA
        _HOVER_BASE = (HOVER_OMEGA / MAX_OMEGA) * 2.0 - 1.0

    base = _HOVER_BASE
    m0 = base + throttle - pitch + roll - yaw   # FR CW
    m1 = base + throttle - pitch - roll + yaw   # FL CCW
    m2 = base + throttle + pitch - roll - yaw   # BL CW
    m3 = base + throttle + pitch + roll + yaw   # BR CCW
    return np.clip(np.array([m0, m1, m2, m3], dtype=np.float32), -1.0, 1.0)


# ── Lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _soccer_trainer, _drone_trainer, _redis_client

    logger.info("DEEPRL starting up (soccer + drone)...")

    # Connect Redis
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=False)
        _redis_client.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis unavailable: {e} — running without checkpoints")
        _redis_client = None

    # Start Soccer trainer
    _soccer_trainer = PPOTrainer(
        scenario="soccer", n_envs=N_ENVS,
        redis_client=_redis_client, players_per_team=1,
    )
    _soccer_trainer.set_speed(TRAINING_SPEED)
    _soccer_trainer.start()

    # Start Drone trainer
    _drone_trainer = PPOTrainer(
        scenario="drone", n_envs=N_ENVS,
        redis_client=_redis_client,
    )
    _drone_trainer.set_speed(TRAINING_SPEED)
    _drone_trainer.start()

    # Start broadcast loops
    soccer_task = asyncio.create_task(_broadcast_loop(_soccer_trainer, _soccer_ws, "soccer"))
    drone_task = asyncio.create_task(_broadcast_loop(_drone_trainer, _drone_ws, "drone"))

    logger.info(
        f"DEEPRL ready: soccer={N_ENVS} envs ({_soccer_trainer.device}), "
        f"drone={N_ENVS} envs ({_drone_trainer.device})"
    )

    yield

    logger.info("DEEPRL shutting down...")
    _soccer_trainer.stop()
    _drone_trainer.stop()
    soccer_task.cancel()
    drone_task.cancel()
    for t in (soccer_task, drone_task):
        try:
            await t
        except asyncio.CancelledError:
            pass


app = FastAPI(title="DEEPRL", lifespan=lifespan)


# ── Broadcast Loop (shared for both scenarios) ────────────────
async def _broadcast_loop(
    trainer: PPOTrainer,
    ws_clients: set[WebSocket],
    label: str,
):
    """Poll trainer for new snapshots and broadcast to WS clients at ~20 Hz."""
    last_stats_gen = -1
    last_training_state = None
    loop_count = 0
    replay_status_counter = 0

    while True:
        try:
            await asyncio.sleep(0.05)  # 20 Hz

            if not trainer:
                continue

            loop_count += 1

            # ── Replay mode: send recorded messages instead of live ──
            global _drone_replayer
            if label == "drone" and _drone_replayer is not None:
                messages = _drone_replayer.get_pending_messages()
                for msg in messages:
                    await _broadcast_to(ws_clients, json.dumps(msg))
                # Send replay status at ~4 Hz
                replay_status_counter += 1
                if replay_status_counter % 5 == 0 and ws_clients:
                    status = _drone_replayer.get_status()
                    await _broadcast_to(ws_clients, json.dumps(status))
                    if _drone_replayer.finished:
                        status["finished"] = True
                        await _broadcast_to(ws_clients, json.dumps(status))
                continue  # skip live training broadcast while replaying

            # Broadcast game state
            snap = trainer.get_snapshot()
            if snap and ws_clients:
                msg = json.dumps(snap)
                dead = set()
                for ws in list(ws_clients):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.add(ws)
                ws_clients.difference_update(dead)

            # Broadcast episode events
            events = trainer.get_episode_events()
            for event in events:
                event_msg = json.dumps(event)
                dead = set()
                for ws in list(ws_clients):
                    try:
                        await ws.send_text(event_msg)
                    except Exception:
                        dead.add(ws)
                ws_clients.difference_update(dead)

            # Broadcast training stats on new generation
            stats = trainer.get_stats()
            if stats and stats.get("generation", 0) != last_stats_gen:
                last_stats_gen = stats.get("generation", 0)
                stats_msg = json.dumps(stats)
                dead = set()
                for ws in list(ws_clients):
                    try:
                        await ws.send_text(stats_msg)
                    except Exception:
                        dead.add(ws)
                ws_clients.difference_update(dead)

            # Broadcast status when training/watch state changes
            current_state = (trainer._training_active, trainer._watch_mode)
            if current_state != last_training_state and ws_clients:
                last_training_state = current_state
                status_msg = json.dumps(trainer.get_status())
                dead = set()
                for ws in list(ws_clients):
                    try:
                        await ws.send_text(status_msg)
                    except Exception:
                        dead.add(ws)
                ws_clients.difference_update(dead)

            # Broadcast BC pretrain progress (drone only)
            if label == "drone" and trainer and ws_clients:
                prog = trainer.get_pretrain_progress()
                if prog:
                    prog_msg = json.dumps(prog)
                    dead = set()
                    for ws in list(ws_clients):
                        try:
                            await ws.send_text(prog_msg)
                        except Exception:
                            dead.add(ws)
                    ws_clients.difference_update(dead)

            # Broadcast new course layout when it changes (endless mode auto-advance)
            if label == "drone" and trainer and ws_clients:
                if hasattr(trainer.env, '_course_changed') and trainer.env._course_changed:
                    trainer.env._course_changed = False
                    layout_msg = json.dumps(trainer.env.get_course_layout())
                    await _broadcast_to(ws_clients, layout_msg)

            # Periodic heartbeat log
            if loop_count % 200 == 0:
                logger.info(
                    f"[{label}] broadcast alive: {loop_count} ticks, "
                    f"{len(ws_clients)} clients, snap={'yes' if snap else 'no'}"
                )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[{label}] broadcast error: {e}", exc_info=True)


async def _broadcast_to(ws_clients: set[WebSocket], msg: str):
    """Send a message to a specific set of WS clients."""
    dead = set()
    for ws in list(ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


# ── Health Check ───────────────────────────────────────────────
@app.get("/health")
async def health():
    status = {
        "status": "ok",
        "service": "deeprl",
        "scenarios": {},
    }
    if _soccer_trainer:
        status["scenarios"]["soccer"] = {
            "training": _soccer_trainer._training_active,
            "generation": _soccer_trainer.generation,
            "device": str(_soccer_trainer.device),
        }
    if _drone_trainer:
        status["scenarios"]["drone"] = {
            "training": _drone_trainer._training_active,
            "generation": _drone_trainer.generation,
            "device": str(_drone_trainer.device),
        }
    return status


# ── System Stats (proxy from servicerouter) ────────────────────
@app.get("/api/system/stats")
async def api_system_stats():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{SERVICEROUTER_URL}/api/system/stats")
            return r.json()
    except Exception:
        return {"gpu_percent": None, "gpu_memory_percent": None}


# ═══════════════════════════════════════════════════════════════
#  SOCCER SCENARIO (backward-compatible routes)
# ═══════════════════════════════════════════════════════════════

@app.get("/api/training/status")
async def soccer_training_status():
    if not _soccer_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    return _soccer_trainer.get_status()


@app.post("/api/training/start")
async def soccer_training_start():
    if _soccer_trainer:
        _soccer_trainer.set_train_mode()
    return {"status": "started"}


@app.post("/api/training/stop")
async def soccer_training_stop():
    if _soccer_trainer:
        _soccer_trainer.stop_training()
    return {"status": "stopped"}


@app.post("/api/training/reset")
async def soccer_training_reset():
    if _soccer_trainer:
        _soccer_trainer.reset_training()
    return {"status": "reset"}


@app.post("/api/training/speed")
async def soccer_training_speed(body: dict):
    speed = body.get("speed", 1)
    if _soccer_trainer:
        _soccer_trainer.set_speed(int(speed))
    return {"speed": speed}


@app.post("/api/training/config")
async def soccer_training_config(body: dict):
    if _soccer_trainer:
        _soccer_trainer.update_config(body)
    return {"status": "updated"}


@app.post("/api/training/team_size")
async def soccer_training_team_size(body: dict):
    ppt = body.get("players_per_team", 1)
    if ppt not in (1, 2):
        return JSONResponse({"error": "players_per_team must be 1 or 2"}, status_code=400)
    if _soccer_trainer:
        _soccer_trainer.set_team_size(int(ppt))
    return {"players_per_team": ppt}


# ── Soccer Checkpoints ────────────────────────────────────────

@app.get("/api/checkpoints")
async def soccer_checkpoint_list():
    if not _soccer_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    return _soccer_trainer.list_checkpoints()


@app.post("/api/checkpoints/save")
async def soccer_checkpoint_save(body: dict):
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)
    if not _soccer_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    result = _soccer_trainer.save_named_checkpoint(name)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    await _broadcast_to(_soccer_ws, json.dumps({
        "type": "checkpoint_list",
        "checkpoints": _soccer_trainer.list_checkpoints(),
    }))
    return result


@app.post("/api/checkpoints/load")
async def soccer_checkpoint_load(body: dict):
    name = body.get("name", "").strip()
    mode = body.get("mode", "watch")
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)
    if mode not in ("watch", "train"):
        return JSONResponse({"error": "Mode must be 'watch' or 'train'"}, status_code=400)
    if not _soccer_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    result = _soccer_trainer.load_named_checkpoint(name, mode)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    await _broadcast_to(_soccer_ws, json.dumps(_soccer_trainer.get_status()))
    await _broadcast_to(_soccer_ws, json.dumps({
        "type": "checkpoint_list",
        "checkpoints": _soccer_trainer.list_checkpoints(),
    }))
    return result


@app.post("/api/checkpoints/delete")
async def soccer_checkpoint_delete(body: dict):
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)
    if not _soccer_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    result = _soccer_trainer.delete_checkpoint(name)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    await _broadcast_to(_soccer_ws, json.dumps({
        "type": "checkpoint_list",
        "checkpoints": _soccer_trainer.list_checkpoints(),
    }))
    return result


# ── Soccer WebSocket ──────────────────────────────────────────
@app.websocket("/ws/arena")
async def ws_arena(ws: WebSocket):
    await ws.accept()
    _soccer_ws.add(ws)
    logger.info(f"[soccer] WS connected ({len(_soccer_ws)} total)")

    if _soccer_trainer:
        await ws.send_text(json.dumps(_soccer_trainer.get_status()))
        stats = _soccer_trainer.get_stats()
        if stats:
            await ws.send_text(json.dumps(stats))
        checkpoints = _soccer_trainer.list_checkpoints()
        if checkpoints:
            await ws.send_text(json.dumps({"type": "checkpoint_list", "checkpoints": checkpoints}))

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            cmd = msg.get("cmd")
            if cmd == "set_speed" and _soccer_trainer:
                _soccer_trainer.set_speed(int(msg.get("speed", 1)))
            elif cmd == "set_config" and _soccer_trainer:
                _soccer_trainer.update_config(msg)
            elif cmd == "set_team_size" and _soccer_trainer:
                _soccer_trainer.set_team_size(int(msg.get("players_per_team", 1)))
            elif cmd == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"[soccer] WS error: {e}")
    finally:
        _soccer_ws.discard(ws)
        logger.info(f"[soccer] WS disconnected ({len(_soccer_ws)} total)")


# ═══════════════════════════════════════════════════════════════
#  DRONE SCENARIO
# ═══════════════════════════════════════════════════════════════

@app.get("/api/drone/training/status")
async def drone_training_status():
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    return _drone_trainer.get_status()


@app.post("/api/drone/training/start")
async def drone_training_start():
    if _drone_trainer:
        _drone_trainer.set_train_mode()
    return {"status": "started"}


@app.post("/api/drone/training/stop")
async def drone_training_stop():
    if _drone_trainer:
        _drone_trainer.stop_training()
    return {"status": "stopped"}


@app.post("/api/drone/training/reset")
async def drone_training_reset():
    if _drone_trainer:
        _drone_trainer.reset_training()
    return {"status": "reset"}


@app.post("/api/drone/training/speed")
async def drone_training_speed(body: dict):
    speed = body.get("speed", 1)
    if _drone_trainer:
        _drone_trainer.set_speed(int(speed))
    return {"speed": speed}


@app.post("/api/drone/training/config")
async def drone_training_config(body: dict):
    if _drone_trainer:
        _drone_trainer.update_config(body)
    return {"status": "updated"}


@app.post("/api/drone/training/curriculum")
async def drone_set_curriculum(body: dict):
    level = body.get("level", 1)
    if level not in range(1, 16):
        return JSONResponse({"error": "Level must be 1-15"}, status_code=400)
    if _drone_trainer:
        _drone_trainer.env.set_curriculum_level(int(level))
        # Broadcast new course layout to all connected viewers
        await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.env.get_course_layout()))
    return {"level": level}


# ── Drone Checkpoints ─────────────────────────────────────────

@app.get("/api/drone/checkpoints")
async def drone_checkpoint_list():
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    return _drone_trainer.list_checkpoints()


@app.post("/api/drone/checkpoints/save")
async def drone_checkpoint_save(body: dict):
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    result = _drone_trainer.save_named_checkpoint(name)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    await _broadcast_to(_drone_ws, json.dumps({
        "type": "checkpoint_list",
        "checkpoints": _drone_trainer.list_checkpoints(),
    }))
    return result


@app.post("/api/drone/checkpoints/load")
async def drone_checkpoint_load(body: dict):
    name = body.get("name", "").strip()
    mode = body.get("mode", "watch")
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)
    if mode not in ("watch", "train"):
        return JSONResponse({"error": "Mode must be 'watch' or 'train'"}, status_code=400)
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    result = _drone_trainer.load_named_checkpoint(name, mode)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))
    await _broadcast_to(_drone_ws, json.dumps({
        "type": "checkpoint_list",
        "checkpoints": _drone_trainer.list_checkpoints(),
    }))
    return result


@app.post("/api/drone/checkpoints/delete")
async def drone_checkpoint_delete(body: dict):
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    result = _drone_trainer.delete_checkpoint(name)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    await _broadcast_to(_drone_ws, json.dumps({
        "type": "checkpoint_list",
        "checkpoints": _drone_trainer.list_checkpoints(),
    }))
    return result


# ── Drone Demos & Pretrain ────────────────────────────────────

@app.get("/api/drone/demos")
async def drone_demo_list():
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    return _drone_trainer.list_demos()


@app.post("/api/drone/demos/delete")
async def drone_demo_delete(body: dict):
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    result = _drone_trainer.delete_demo(name)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    await _broadcast_to(_drone_ws, json.dumps({
        "type": "demo_list",
        "demos": _drone_trainer.list_demos(),
    }))
    return result


@app.post("/api/drone/pretrain")
async def drone_pretrain(body: dict):
    demos = body.get("demos", [])
    epochs = body.get("epochs", 50)
    if not demos:
        return JSONResponse({"error": "No demos specified"}, status_code=400)
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)

    def _run_bc():
        result = _drone_trainer.pretrain_from_demos(demos, epochs=int(epochs))
        logger.info(f"Pretrain result: {result}")

    threading.Thread(target=_run_bc, daemon=True).start()
    return {"status": "started", "demos": demos, "epochs": epochs}


# ── Drone Recordings ─────────────────────────────────────────

@app.get("/api/drone/recordings")
async def drone_recording_list():
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    return _drone_trainer.list_recordings()


@app.post("/api/drone/recordings/start")
async def drone_recording_start(body: dict):
    name = body.get("name", "").strip()
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    result = _drone_trainer.start_recording(name)
    await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))
    return result


@app.post("/api/drone/recordings/stop")
async def drone_recording_stop():
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    result = _drone_trainer.stop_recording()
    await _broadcast_to(_drone_ws, json.dumps({
        "type": "recording_list",
        "recordings": _drone_trainer.list_recordings(),
    }))
    await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))
    return result


@app.post("/api/drone/recordings/delete")
async def drone_recording_delete(body: dict):
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)
    if not _drone_trainer:
        return JSONResponse({"error": "Trainer not initialized"}, status_code=503)
    result = _drone_trainer.delete_recording(name)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    await _broadcast_to(_drone_ws, json.dumps({
        "type": "recording_list",
        "recordings": _drone_trainer.list_recordings(),
    }))
    return result


# ── Drone WebSocket ───────────────────────────────────────────
@app.websocket("/ws/course")
async def ws_course(ws: WebSocket):
    global _drone_replayer
    await ws.accept()
    _drone_ws.add(ws)
    logger.info(f"[drone] WS connected ({len(_drone_ws)} total)")

    if _drone_trainer:
        await ws.send_text(json.dumps(_drone_trainer.get_status()))
        stats = _drone_trainer.get_stats()
        if stats:
            await ws.send_text(json.dumps(stats))
        checkpoints = _drone_trainer.list_checkpoints()
        if checkpoints:
            await ws.send_text(json.dumps({"type": "checkpoint_list", "checkpoints": checkpoints}))
        demos = _drone_trainer.list_demos()
        if demos:
            await ws.send_text(json.dumps({"type": "demo_list", "demos": demos}))
        # Send full course layout so viewer can build obstacles/gates
        await ws.send_text(json.dumps(_drone_trainer.env.get_course_layout()))
        # Send recording list
        recordings = _drone_trainer.list_recordings()
        if recordings:
            await ws.send_text(json.dumps({"type": "recording_list", "recordings": recordings}))
        # If replayer is active, send its status
        if _drone_replayer:
            await ws.send_text(json.dumps(_drone_replayer.get_status()))

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            cmd = msg.get("cmd")
            if cmd == "set_speed" and _drone_trainer:
                _drone_trainer.set_speed(int(msg.get("speed", 1)))
            elif cmd == "set_config" and _drone_trainer:
                _drone_trainer.update_config(msg)
            elif cmd == "set_curriculum" and _drone_trainer:
                _drone_trainer.env.set_curriculum_level(int(msg.get("level", 1)))
                layout = json.dumps(_drone_trainer.env.get_course_layout())
                await _broadcast_to(_drone_ws, layout)

            # ── Manual Flight Commands ────────────────────
            elif cmd == "manual_start" and _drone_trainer:
                _drone_trainer.set_manual_mode(True)
                await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))
            elif cmd == "manual_stop" and _drone_trainer:
                _drone_trainer.set_manual_mode(False)
                await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))
            elif cmd == "manual_input" and _drone_trainer:
                throttle = float(msg.get("throttle", 0))
                pitch = float(msg.get("pitch", 0))
                roll = float(msg.get("roll", 0))
                yaw = float(msg.get("yaw", 0))
                _drone_trainer.set_manual_controls(throttle, pitch, roll, yaw)
            elif cmd == "record_start" and _drone_trainer:
                _drone_trainer.start_demo_recording()
                await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))
            elif cmd == "record_stop" and _drone_trainer:
                _drone_trainer.stop_demo_recording()
                await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))
            elif cmd == "demo_save" and _drone_trainer:
                name = msg.get("name", "").strip()
                if name:
                    result = _drone_trainer.save_demo(name)
                    await ws.send_text(json.dumps({"type": "demo_save_result", **result}))
                    await _broadcast_to(_drone_ws, json.dumps({
                        "type": "demo_list",
                        "demos": _drone_trainer.list_demos(),
                    }))
                    await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))
            elif cmd == "demo_delete" and _drone_trainer:
                name = msg.get("name", "").strip()
                if name:
                    _drone_trainer.delete_demo(name)
                    await _broadcast_to(_drone_ws, json.dumps({
                        "type": "demo_list",
                        "demos": _drone_trainer.list_demos(),
                    }))

            # ── Endless Mode Commands ───────────────────
            elif cmd == "endless_start" and _drone_trainer:
                seed = int(msg.get("seed", 42))
                _drone_trainer.env.set_endless_mode(True, base_seed=seed)
                layout = json.dumps(_drone_trainer.env.get_course_layout())
                await _broadcast_to(_drone_ws, layout)
                await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))
            elif cmd == "endless_stop" and _drone_trainer:
                _drone_trainer.env.set_endless_mode(False)
                layout = json.dumps(_drone_trainer.env.get_course_layout())
                await _broadcast_to(_drone_ws, layout)
                await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))

            # ── User Weapon Commands ────────────────────
            elif cmd == "weapon_mode_on" and _drone_trainer:
                _drone_trainer.env._weapon_mode = True
                _drone_trainer.env._user_shots = 0
                _drone_trainer.env._user_hits = 0
            elif cmd == "weapon_mode_off" and _drone_trainer:
                _drone_trainer.env._weapon_mode = False
            elif cmd == "user_fire" and _drone_trainer:
                if _drone_trainer.env._weapon_mode:
                    tx = float(msg.get("target_x", 0))
                    ty = float(msg.get("target_y", 0))
                    tz = float(msg.get("target_z", 0))
                    _drone_trainer.env.inject_user_projectile(tx, ty, tz)

            # ── Adversary Drone Commands ────────────────
            elif cmd == "adversary_on" and _drone_trainer:
                lethal = bool(msg.get("lethal", False))
                _drone_trainer.env.set_adversary(True, lethal=lethal)
            elif cmd == "adversary_off" and _drone_trainer:
                _drone_trainer.env.set_adversary(False)

            # ── Swarm Visualization ──────────────────
            elif cmd == "swarm_toggle" and _drone_trainer:
                enabled = bool(msg.get("enabled", False))
                _drone_trainer.set_swarm_mode(enabled)

            # ── Replay Commands ───────────────────────
            elif cmd == "replay_load" and _drone_trainer:
                rec_name = msg.get("name", "").strip()
                if rec_name:
                    blob = _drone_trainer.load_recording(rec_name)
                    if blob:
                        # Find metadata from index
                        meta = {}
                        if _redis_client:
                            meta_raw = _redis_client.hget("deeprl:drone:recording:index", rec_name)
                            if meta_raw:
                                meta = json.loads(meta_raw.decode() if isinstance(meta_raw, bytes) else meta_raw)
                        _drone_replayer = TrainingReplayer(blob, meta)
                        layout = _drone_replayer.get_last_layout()
                        if layout:
                            await _broadcast_to(_drone_ws, json.dumps(layout))
                        await _broadcast_to(_drone_ws, json.dumps(_drone_replayer.get_status()))
                    else:
                        await ws.send_text(json.dumps({"type": "error", "message": f"Recording '{rec_name}' not found"}))
            elif cmd == "replay_play":
                if _drone_replayer:
                    speed = float(msg.get("speed", 50))
                    _drone_replayer.play(speed)
                    await _broadcast_to(_drone_ws, json.dumps(_drone_replayer.get_status()))
            elif cmd == "replay_pause":
                if _drone_replayer:
                    _drone_replayer.pause()
                    await _broadcast_to(_drone_ws, json.dumps(_drone_replayer.get_status()))
            elif cmd == "replay_resume":
                if _drone_replayer:
                    _drone_replayer.resume()
                    await _broadcast_to(_drone_ws, json.dumps(_drone_replayer.get_status()))
            elif cmd == "replay_seek":
                if _drone_replayer:
                    fraction = float(msg.get("fraction", 0))
                    _drone_replayer.seek(fraction)
                    await _broadcast_to(_drone_ws, json.dumps(_drone_replayer.get_status()))
            elif cmd == "replay_speed":
                if _drone_replayer:
                    speed = float(msg.get("speed", 50))
                    _drone_replayer.set_speed(speed)
                    await _broadcast_to(_drone_ws, json.dumps(_drone_replayer.get_status()))
            elif cmd == "replay_stop":
                _drone_replayer = None
                if _drone_trainer:
                    await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.env.get_course_layout()))
                    await _broadcast_to(_drone_ws, json.dumps(_drone_trainer.get_status()))
                await _broadcast_to(_drone_ws, json.dumps({"type": "replay_status", "playing": False, "finished": True}))

            elif cmd == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"[drone] WS error: {e}")
    finally:
        _drone_ws.discard(ws)
        logger.info(f"[drone] WS disconnected ({len(_drone_ws)} total)")


# ── Static Files (MUST be last) ────────────────────────────────
# Drone UI served at /drone/
app.mount("/drone", StaticFiles(directory="static-drone", html=True), name="static-drone")
# Soccer UI served at / (catch-all, must be last)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
