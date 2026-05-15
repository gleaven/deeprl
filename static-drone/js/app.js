/**
 * DRONERL — Main controller, WebSocket, UI state.
 */

(function () {
    'use strict';

    // ── State ─────────────────────────────────────────────────
    let ws = null;
    let wsRetryDelay = 1000;
    let currentSpeed = 1;
    let watchMode = false;
    let checkpointList = [];
    let curriculumLevel = 1;

    // Replay state
    let replayActive = false;
    let recordingList = [];

    // Manual flight
    let manualMode = false;
    let recording = false;
    let demoList = [];
    let _manualInterval = null;
    const _keys = {};

    const episodeLog = [];
    const MAX_LOG = 30;
    const MAX_RPM = 12000;
    const COURSE_H = 20.0;

    // ── Keyboard Tracking ────────────────────────────────────
    document.addEventListener('keydown', (e) => {
        if (!manualMode) return;
        // Prevent page scroll when flying
        if (e.code === 'Space' || e.code === 'ArrowUp' || e.code === 'ArrowDown') {
            e.preventDefault();
        }
        _keys[e.code] = true;
    });
    document.addEventListener('keyup', (e) => {
        delete _keys[e.code];
    });

    function _startManualInput() {
        if (_manualInterval) return;
        _manualInterval = setInterval(() => {
            let throttle = 0, pitch = 0, roll = 0, yaw = 0;
            if (_keys['Space'])                               throttle += 1.0;
            if (_keys['ShiftLeft'] || _keys['ShiftRight'])    throttle -= 1.0;
            if (_keys['KeyW'])                                pitch += 1.0;
            if (_keys['KeyS'])                                pitch -= 1.0;
            if (_keys['KeyA'])                                roll += 1.0;
            if (_keys['KeyD'])                                roll -= 1.0;
            if (_keys['KeyQ'])                                yaw -= 1.0;
            if (_keys['KeyE'])                                yaw += 1.0;
            _send({ cmd: 'manual_input', throttle, pitch, roll, yaw });
        }, 50);  // 20 Hz
    }

    function _stopManualInput() {
        if (_manualInterval) {
            clearInterval(_manualInterval);
            _manualInterval = null;
        }
        // Clear pressed keys
        for (const k in _keys) delete _keys[k];
    }

    // ── Init ──────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
            if (window.Arena) {
                window.Arena.init(document.getElementById('arena-container'));
            }
            _initControls();
            _connect();
            _pollGPU();
        }, 100);
    });

    // ── WebSocket ─────────────────────────────────────────────
    function _connect() {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const url = `${proto}://${location.host}/ws/course`;

        ws = new WebSocket(url);

        ws.onopen = () => {
            _setConnStatus('LIVE', true);
            _setStatusDot(true);
            wsRetryDelay = 1000;
        };

        ws.onclose = () => {
            _setConnStatus('RECONNECTING...', false);
            _setStatusDot(false);
            ws = null;
            setTimeout(_connect, wsRetryDelay);
            wsRetryDelay = Math.min(wsRetryDelay * 1.5, 10000);
        };

        ws.onerror = () => {
            _setStatusDot(false);
        };

        ws.onmessage = (evt) => {
            try {
                const msg = JSON.parse(evt.data);
                _handleMessage(msg);
            } catch (e) {
                // ignore parse errors
            }
        };
    }

    function _send(obj) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(obj));
        }
    }

    // ── Message Handling ──────────────────────────────────────
    function _handleMessage(msg) {
        // Drone state includes scenario: 'drone'
        if (msg.scenario === 'drone' || msg.type === 'state') {
            if (msg.drone) { _onDroneState(msg); return; }
        }
        switch (msg.type) {
            case 'course_layout':      _onCourseLayout(msg); break;
            case 'episode_end':        _onEpisodeEnd(msg); break;
            case 'training_stats':     _onTrainingStats(msg); break;
            case 'status':             _onStatus(msg); break;
            case 'checkpoint_list':    _onCheckpointList(msg); break;
            case 'demo_list':          _onDemoList(msg); break;
            case 'pretrain_progress':  _onPretrainProgress(msg); break;
            case 'recording_list':     _onRecordingList(msg); break;
            case 'replay_status':      _onReplayStatus(msg); break;
            case 'pong':               break;
        }
    }

    function _onDroneState(msg) {
        // Store for weapon targeting
        if (msg.drone) window._lastDroneState = msg.drone;

        // Update 3D scene
        if (window.Arena) {
            window.Arena.update(msg);
        }

        // Step counter
        _setText('step-counter', msg.step);
        if (msg.max_steps) _setText('step-max', msg.max_steps);
        _setText('hdr-generation', msg.generation || 0);
        _setText('gen-badge', msg.generation || 0);

        // Gate progress
        const gp = msg.gates_passed || 0;
        const gt = msg.total_gates || 0;
        _setText('gate-count', `${gp}/${gt}`);
        _setText('hdr-gates', `${gp}/${gt}`);
        const bar = document.getElementById('gate-bar');
        if (bar && gt > 0) bar.style.width = (gp / gt * 100) + '%';

        // Curriculum level
        if (msg.curriculum_level) {
            _setText('hdr-level', msg.curriculum_level);
            _updateCurriculumUI(msg.curriculum_level);
            if (msg.level_name) _setText('level-name', msg.level_name);
        }

        // HUD: altitude
        const drone = msg.drone;
        if (drone) {
            const altPct = Math.min(drone.z / COURSE_H * 100, 100);
            const altFill = document.getElementById('alt-fill');
            if (altFill) altFill.style.height = altPct + '%';
            _setText('alt-value', drone.z.toFixed(1) + 'm');

            // HUD: attitude
            const horizon = document.getElementById('att-horizon');
            if (horizon) {
                const pitchDeg = (drone.pitch || 0) * 180 / Math.PI;
                const rollDeg = (drone.roll || 0) * 180 / Math.PI;
                horizon.style.transform = `rotate(${-rollDeg}deg) translateY(${pitchDeg * 0.5}px)`;
            }

            // HUD: motor RPMs
            if (drone.motor_rpms) {
                for (let m = 0; m < 4; m++) {
                    const fill = document.getElementById('motor-' + m);
                    if (fill) fill.style.height = Math.min(drone.motor_rpms[m] / MAX_RPM * 100, 100) + '%';
                }
            }

            // HUD: battery
            if (drone.battery != null) {
                const batFill = document.getElementById('battery-fill');
                if (batFill) {
                    batFill.style.width = (drone.battery * 100) + '%';
                    batFill.className = 'battery-fill' +
                        (drone.battery < 0.1 ? ' critical' : drone.battery < 0.3 ? ' low' : '');
                }
            }
        }

        // EW status indicators
        const ewGps = document.getElementById('ew-gps');
        const ewJam = document.getElementById('ew-jam');
        if (ewGps) ewGps.className = 'status-icon' + (msg.gps_denied ? ' active' : '');
        if (ewJam) ewJam.className = 'status-icon' + (msg.jamming > 0.1 ? ' active' : '');

        // Wind indicator
        const ewWind = document.getElementById('ew-wind');
        if (ewWind && msg.wind) {
            const windMag = Math.sqrt(
                msg.wind.base[0] ** 2 + msg.wind.base[1] ** 2 + msg.wind.base[2] ** 2
            );
            ewWind.className = 'status-icon' + (windMag > 2 ? ' active' : '');
        }

        // Threat warning
        const tw = document.getElementById('threat-warning');
        if (tw) tw.className = 'threat-warning' + ((msg.projectiles && msg.projectiles.length > 0) ? ' active' : '');

        // Endless mode HUD
        if (msg.endless) {
            _setText('endless-scenario', '#' + msg.endless.scenario);
            _setText('endless-difficulty', 'D=' + msg.endless.difficulty.toFixed(2));
            _setText('endless-streak', msg.endless.streak);
            _setText('endless-best', msg.endless.best_streak);
            _showEl('endless-hud', true);
            // Override level display
            _setText('hdr-level', '\u221E');
            _setText('level-name', 'Endless #' + msg.endless.scenario);
        } else {
            _showEl('endless-hud', false);
        }

        // Weapon stats
        if (msg.weapon) {
            _setText('weapon-shots', msg.weapon.shots);
            _setText('weapon-hits', msg.weapon.hits);
            _setText('weapon-accuracy', msg.weapon.accuracy + '%');
            _showEl('weapon-stats', true);
        } else {
            _showEl('weapon-stats', false);
        }

        // Bandit alert
        const banditAlert = document.getElementById('bandit-alert');
        if (banditAlert) {
            if (msg.adversary && msg.adversary.active && drone) {
                const dx = msg.adversary.x - drone.x;
                const dy = msg.adversary.y - drone.y;
                const dz = msg.adversary.z - drone.z;
                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                banditAlert.textContent = 'BANDIT ' + dist.toFixed(0) + 'm';
                banditAlert.classList.add('active');
            } else {
                banditAlert.classList.remove('active');
            }
        }
    }

    function _onCourseLayout(msg) {
        if (window.Arena) {
            window.Arena.buildCourse(msg);
        }
        if (msg.curriculum_level) {
            _updateCurriculumUI(msg.curriculum_level);
            _setText('hdr-level', msg.curriculum_level);
        }
        if (msg.level_name) _setText('level-name', msg.level_name);
    }

    function _onEpisodeEnd(msg) {
        const result = msg.completed ? 'COMPLETE' : msg.collision ? 'CRASH' : 'TIMEOUT';
        const cls = msg.completed ? 'complete' : msg.collision ? 'crash' : 'timeout';
        episodeLog.unshift({
            episode: msg.episode,
            result: result,
            cls: cls,
            gates: msg.gates_passed || 0,
            totalGates: msg.total_gates || 0,
        });
        if (episodeLog.length > MAX_LOG) episodeLog.pop();
        _renderEpisodeLog();
        _setText('m-episodes', msg.episode);
        _setText('episode-counter', msg.episode);
    }

    function _onTrainingStats(msg) {
        _setText('m-ploss', msg.policy_loss != null ? msg.policy_loss.toFixed(4) : '--');
        _setText('m-vloss', msg.value_loss != null ? msg.value_loss.toFixed(4) : '--');
        _setText('m-entropy', msg.entropy != null ? msg.entropy.toFixed(4) : '--');
        _setText('m-steps', _fmtNum(msg.total_steps || 0));
        if (msg.episodes != null) _setText('m-episodes', msg.episodes);

        // Push to charts
        if (typeof Charts !== 'undefined') {
            Charts.pushReward(msg.avg_reward || 0);
            Charts.pushGateRate(msg.avg_gates_passed || 0);
            Charts.pushCompletion(msg.completion_rate != null ? msg.completion_rate : 0);
        }
    }

    function _onStatus(msg) {
        watchMode = msg.watch_mode || false;
        _updateTrainingButtons(msg.training, watchMode);

        // Manual flight state
        const wasManual = manualMode;
        manualMode = msg.manual_mode || false;
        recording = msg.recording || false;
        _updateManualUI();

        // Start/stop keyboard input loop + chase camera
        if (manualMode && !wasManual) {
            _startManualInput();
            if (window.Arena?.setChaseCamera) window.Arena.setChaseCamera(true);
        }
        if (!manualMode && wasManual) {
            _stopManualInput();
            if (window.Arena?.setChaseCamera) window.Arena.setChaseCamera(false);
        }

        if (msg.curriculum_level) {
            curriculumLevel = msg.curriculum_level;
            _updateCurriculumUI(msg.curriculum_level);
            _setText('hdr-level', msg.curriculum_level);
            if (msg.level_name) _setText('level-name', msg.level_name);
        }

        // Session recording state
        const recBtn = document.getElementById('btn-rec-training');
        const recStopBtn = document.getElementById('btn-rec-training-stop');
        if (msg.session_recording) {
            if (recBtn) { recBtn.disabled = true; recBtn.classList.add('rec-training-active'); }
            if (recStopBtn) recStopBtn.disabled = false;
        } else {
            if (recBtn) { recBtn.disabled = false; recBtn.classList.remove('rec-training-active'); }
            if (recStopBtn) recStopBtn.disabled = true;
        }
    }

    function _onCheckpointList(msg) {
        checkpointList = msg.checkpoints || [];
        _renderCheckpointList();
    }

    function _updateTrainingButtons(training, watching) {
        const btnStart = document.getElementById('btn-start');
        const btnStop = document.getElementById('btn-stop');
        if (btnStart) {
            btnStart.disabled = training;
            btnStart.textContent = watching ? 'TRAIN' : 'START';
        }
        if (btnStop) {
            btnStop.disabled = !training && !watching;
            btnStop.textContent = watching ? 'STOP' : 'PAUSE';
        }
    }

    function _onDemoList(msg) {
        demoList = msg.demos || [];
        _renderDemoList();
        // Enable pretrain button if demos exist
        const btn = document.getElementById('btn-pretrain');
        if (btn) btn.disabled = demoList.length === 0;
    }

    function _onPretrainProgress(msg) {
        const bar = document.getElementById('pretrain-bar');
        const info = document.getElementById('pretrain-info');
        const wrap = document.getElementById('pretrain-progress');
        if (!bar || !info || !wrap) return;

        wrap.style.display = 'block';
        const pct = (msg.epoch / msg.total_epochs * 100);
        bar.style.width = pct + '%';
        info.textContent = `${msg.epoch}/${msg.total_epochs} · loss ${msg.loss.toFixed(4)}`;

        if (msg.done) {
            setTimeout(() => { wrap.style.display = 'none'; }, 3000);
        }
    }

    function _onRecordingList(msg) {
        recordingList = msg.recordings || [];
        _renderRecordingList();
    }

    function _onReplayStatus(msg) {
        const bar = document.getElementById('replay-bar');
        if (!bar) return;

        if (!msg.playing && msg.finished && !msg.paused) {
            // Replay ended
            replayActive = false;
            bar.style.display = 'none';
            document.body.classList.remove('replay-active');
            return;
        }

        replayActive = true;
        bar.style.display = 'flex';
        document.body.classList.add('replay-active');

        _setText('replay-name', msg.recording_name || '');
        _setText('replay-gen', 'Gen ' + (msg.current_generation || 0));

        const seek = document.getElementById('replay-seek');
        if (seek && !seek._dragging) {
            seek.value = msg.progress || 0;
        }

        const badge = document.getElementById('replay-badge');
        if (badge) {
            badge.textContent = msg.paused ? '⏸ PAUSED' : '▶ REPLAY';
        }

        const pauseBtn = document.getElementById('replay-pause-btn');
        if (pauseBtn) {
            pauseBtn.textContent = msg.paused ? '▶' : '⏸';
        }
    }

    function _renderRecordingList() {
        const el = document.getElementById('recording-list');
        if (!el) return;

        if (recordingList.length === 0) {
            el.innerHTML = '<div class="rec-empty">No recordings</div>';
            return;
        }

        el.innerHTML = recordingList.map(rec => {
            const gens = rec.generations || 0;
            const dur = rec.duration_seconds || 0;
            const lvl = rec.final_level || '?';
            return `<div class="rec-row" data-name="${rec.name}">` +
                `<div class="rec-info">` +
                `<span class="rec-name">${rec.name}</span>` +
                `<span class="rec-meta">${gens} gens · ${Math.round(dur)}s · Lvl ${lvl}</span>` +
                `</div>` +
                `<div class="rec-actions">` +
                `<button class="rec-play-btn" title="Play">&#9654;</button>` +
                `<button class="rec-del-btn" title="Delete">&times;</button>` +
                `</div></div>`;
        }).join('');

        el.querySelectorAll('.rec-play-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const name = btn.closest('.rec-row').dataset.name;
                const speed = document.getElementById('replay-speed');
                const spd = speed ? parseFloat(speed.value) : 50;
                _send({ cmd: 'replay_load', name });
                setTimeout(() => _send({ cmd: 'replay_play', speed: spd }), 200);
            });
        });

        el.querySelectorAll('.rec-del-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const name = btn.closest('.rec-row').dataset.name;
                if (confirm(`Delete recording "${name}"?`)) {
                    fetch('/api/drone/recordings/delete', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name }),
                    });
                }
            });
        });
    }

    function _updateManualUI() {
        const btnFly = document.getElementById('btn-fly');
        const btnRec = document.getElementById('btn-rec');
        const btnStop = document.getElementById('btn-demo-stop');
        const hints = document.getElementById('key-hints');
        const saveRow = document.getElementById('demo-save-row');
        const status = document.getElementById('demo-status');

        if (btnFly) {
            btnFly.textContent = manualMode ? 'EXIT' : 'FLY';
            btnFly.className = manualMode ? 'btn btn-danger' : 'btn btn-accent';
        }
        if (btnRec) btnRec.disabled = !manualMode || recording;
        if (btnStop) btnStop.disabled = !recording;
        if (hints) hints.style.display = manualMode ? 'block' : 'none';

        // Show save row when we have buffer but not recording
        if (saveRow) saveRow.style.display = (manualMode && !recording) ? 'flex' : 'none';

        if (status) {
            if (recording) {
                status.textContent = 'REC';
                status.className = 'demo-status-tag rec-pulse';
            } else if (manualMode) {
                status.textContent = 'FLYING';
                status.className = 'demo-status-tag flying';
            } else {
                status.textContent = '';
                status.className = 'demo-status-tag';
            }
        }

        // Disable training controls in manual mode
        const btnStart = document.getElementById('btn-start');
        const btnReset = document.getElementById('btn-reset');
        if (btnStart) btnStart.disabled = manualMode || btnStart.disabled;
        if (btnReset) btnReset.disabled = manualMode;
    }

    function _renderDemoList() {
        const el = document.getElementById('demo-list');
        if (!el) return;

        if (demoList.length === 0) {
            el.innerHTML = '<div class="demo-empty">No saved demos</div>';
            return;
        }

        el.innerHTML = demoList.map(demo => {
            return `<div class="demo-row" data-name="${demo.name}">` +
                `<div class="demo-info">` +
                `<span class="demo-name">${demo.name}</span>` +
                `<span class="demo-meta">${demo.steps} steps · ${demo.duration}s · Lvl ${demo.level || '?'}</span>` +
                `</div>` +
                `<div class="demo-actions">` +
                `<button class="ckpt-btn demo-delete" title="Delete">&times;</button>` +
                `</div></div>`;
        }).join('');

        el.querySelectorAll('.demo-delete').forEach(btn => {
            btn.addEventListener('click', () => {
                const name = btn.closest('.demo-row').dataset.name;
                if (confirm(`Delete demo "${name}"?`)) {
                    _send({ cmd: 'demo_delete', name });
                }
            });
        });
    }

    function _updateCurriculumUI(level) {
        curriculumLevel = level;
        document.querySelectorAll('.level-btn').forEach(btn => {
            const btnLevel = parseInt(btn.dataset.level);
            btn.classList.remove('active', 'locked');
            if (btnLevel === level) {
                btn.classList.add('active');
            } else if (btnLevel > level) {
                btn.classList.add('locked');
            }
        });
    }

    // ── Controls ──────────────────────────────────────────────
    function _initControls() {
        _on('btn-start', 'click', () => {
            fetch('/api/drone/training/start', { method: 'POST' });
            watchMode = false;
            _updateTrainingButtons(true, false);
        });

        _on('btn-stop', 'click', () => {
            fetch('/api/drone/training/stop', { method: 'POST' });
            watchMode = false;
            _updateTrainingButtons(false, false);
        });

        _on('btn-reset', 'click', () => {
            if (confirm('Reset training? All learned behavior will be lost.')) {
                fetch('/api/drone/training/reset', { method: 'POST' });
                watchMode = false;
                _updateTrainingButtons(true, false);
            }
        });

        // Curriculum level buttons
        document.querySelectorAll('.level-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const level = parseInt(btn.dataset.level);
                if (btn.classList.contains('locked')) return;
                _send({ cmd: 'set_curriculum', level: level });
            });
        });

        // Speed buttons
        document.querySelectorAll('.speed-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                currentSpeed = parseInt(btn.dataset.speed);
                document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                _send({ cmd: 'set_speed', speed: currentSpeed });
            });
        });

        // Hyperparameter sliders
        _initSlider('sl-lr', 'sl-lr-val', v => {
            const lr = Math.pow(10, parseFloat(v));
            _setText('sl-lr-val', lr.toExponential(0));
            _send({ cmd: 'set_config', learning_rate: lr });
        });

        _initSlider('sl-entropy', 'sl-entropy-val', v => {
            _setText('sl-entropy-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', entropy_coeff: parseFloat(v) });
        });

        // Reward sliders
        _initSlider('sl-waypoint', 'sl-waypoint-val', v => {
            _setText('sl-waypoint-val', parseFloat(v).toFixed(1));
            _send({ cmd: 'set_config', waypoint_reward: parseFloat(v) });
        });

        _initSlider('sl-progress', 'sl-progress-val', v => {
            _setText('sl-progress-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', progress_reward: parseFloat(v) });
        });

        _initSlider('sl-course', 'sl-course-val', v => {
            _setText('sl-course-val', parseInt(v));
            _send({ cmd: 'set_config', course_complete_reward: parseFloat(v) });
        });

        // Penalty sliders
        _initSlider('sl-survival', 'sl-survival-val', v => {
            _setText('sl-survival-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', survival_reward: parseFloat(v) });
        });

        _initSlider('sl-gatealign', 'sl-gatealign-val', v => {
            _setText('sl-gatealign-val', parseFloat(v).toFixed(1));
            _send({ cmd: 'set_config', gate_align_bonus: parseFloat(v) });
        });

        _initSlider('sl-dodge', 'sl-dodge-val', v => {
            _setText('sl-dodge-val', parseFloat(v).toFixed(1));
            _send({ cmd: 'set_config', dodge_bonus: parseFloat(v) });
        });

        _initSlider('sl-altitude', 'sl-altitude-val', v => {
            _setText('sl-altitude-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', altitude_coeff: parseFloat(v) });
        });

        _initSlider('sl-collision', 'sl-collision-val', v => {
            _setText('sl-collision-val', parseFloat(v).toFixed(1));
            _send({ cmd: 'set_config', collision_penalty: -parseFloat(v) });
        });

        _initSlider('sl-crash', 'sl-crash-val', v => {
            _setText('sl-crash-val', parseFloat(v).toFixed(1));
            _send({ cmd: 'set_config', crash_penalty: -parseFloat(v) });
        });

        _initSlider('sl-oob', 'sl-oob-val', v => {
            _setText('sl-oob-val', parseFloat(v).toFixed(1));
            _send({ cmd: 'set_config', oob_penalty: -parseFloat(v) });
        });

        _initSlider('sl-projhit', 'sl-projhit-val', v => {
            _setText('sl-projhit-val', parseFloat(v).toFixed(1));
            _send({ cmd: 'set_config', projectile_hit_penalty: -parseFloat(v) });
        });

        _initSlider('sl-stability', 'sl-stability-val', v => {
            _setText('sl-stability-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', stability_coeff: parseFloat(v) });
        });

        _initSlider('sl-orientation', 'sl-orientation-val', v => {
            _setText('sl-orientation-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', orientation_coeff: parseFloat(v) });
        });

        _initSlider('sl-energy', 'sl-energy-val', v => {
            _setText('sl-energy-val', parseFloat(v).toFixed(4));
            _send({ cmd: 'set_config', energy_coeff: parseFloat(v) });
        });

        _initSlider('sl-smoothness', 'sl-smoothness-val', v => {
            _setText('sl-smoothness-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', smoothness_coeff: parseFloat(v) });
        });

        _initSlider('sl-speed', 'sl-speed-val', v => {
            _setText('sl-speed-val', parseFloat(v).toFixed(4));
            _send({ cmd: 'set_config', speed_coeff: parseFloat(v) });
        });

        // Reset all sliders to defaults
        _on('btn-defaults', 'click', () => {
            const defaults = {
                'sl-lr':          { val: -3.52,  fmt: v => Math.pow(10, v).toExponential(0), cfg: { learning_rate: 3e-4 } },
                'sl-entropy':     { val: 0.005,  fmt: v => v.toFixed(3), cfg: { entropy_coeff: 0.005 } },
                'sl-waypoint':    { val: 5,      fmt: v => v.toFixed(1), cfg: { waypoint_reward: 5 } },
                'sl-progress':    { val: 0.05,   fmt: v => v.toFixed(3), cfg: { progress_reward: 0.05 } },
                'sl-course':      { val: 50,     fmt: v => String(Math.round(v)), cfg: { course_complete_reward: 50 } },
                'sl-survival':    { val: 0.01,   fmt: v => v.toFixed(3), cfg: { survival_reward: 0.01 } },
                'sl-gatealign':   { val: 1,      fmt: v => v.toFixed(1), cfg: { gate_align_bonus: 1 } },
                'sl-dodge':       { val: 0.5,    fmt: v => v.toFixed(1), cfg: { dodge_bonus: 0.5 } },
                'sl-altitude':    { val: 0.005,  fmt: v => v.toFixed(3), cfg: { altitude_coeff: 0.005 } },
                'sl-collision':   { val: 10,     fmt: v => v.toFixed(1), cfg: { collision_penalty: -10 } },
                'sl-crash':       { val: 10,     fmt: v => v.toFixed(1), cfg: { crash_penalty: -10 } },
                'sl-oob':         { val: 10,     fmt: v => v.toFixed(1), cfg: { oob_penalty: -10 } },
                'sl-projhit':     { val: 5,      fmt: v => v.toFixed(1), cfg: { projectile_hit_penalty: -5 } },
                'sl-stability':   { val: 0.01,   fmt: v => v.toFixed(3), cfg: { stability_coeff: 0.01 } },
                'sl-orientation': { val: 0.01,   fmt: v => v.toFixed(3), cfg: { orientation_coeff: 0.01 } },
                'sl-energy':      { val: 0.001,  fmt: v => v.toFixed(4), cfg: { energy_coeff: 0.001 } },
                'sl-smoothness':  { val: 0.005,  fmt: v => v.toFixed(3), cfg: { smoothness_coeff: 0.005 } },
                'sl-speed':       { val: 0.002,  fmt: v => v.toFixed(4), cfg: { speed_coeff: 0.002 } },
            };
            const fullCfg = {};
            for (const [id, d] of Object.entries(defaults)) {
                const slider = document.getElementById(id);
                if (slider) slider.value = d.val;
                _setText(id + '-val', d.fmt(d.val));
                Object.assign(fullCfg, d.cfg);
            }
            _send({ cmd: 'set_config', ...fullCfg });
        });

        // ── Manual Flight / Demo Controls ────────────
        _on('btn-fly', 'click', () => {
            if (manualMode) {
                _send({ cmd: 'manual_stop' });
            } else {
                _send({ cmd: 'manual_start' });
            }
        });

        _on('btn-rec', 'click', () => {
            _send({ cmd: 'record_start' });
        });

        _on('btn-demo-stop', 'click', () => {
            _send({ cmd: 'record_stop' });
        });

        _on('btn-demo-save', 'click', () => {
            const input = document.getElementById('demo-name');
            const name = (input ? input.value.trim() : '');
            if (!name) return;
            _send({ cmd: 'demo_save', name });
            if (input) input.value = '';
        });

        const demoInput = document.getElementById('demo-name');
        if (demoInput) {
            demoInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    const name = demoInput.value.trim();
                    if (name) {
                        _send({ cmd: 'demo_save', name });
                        demoInput.value = '';
                    }
                }
                // Prevent manual flight keys from firing while typing
                e.stopPropagation();
            });
        }

        _on('btn-pretrain', 'click', () => {
            if (demoList.length === 0) return;
            const names = demoList.map(d => d.name);
            fetch('/api/drone/pretrain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ demos: names, epochs: 50 }),
            });
        });

        // Fetch demo list on load
        fetch('/api/drone/demos').then(r => r.json()).then(data => {
            if (Array.isArray(data)) {
                demoList = data;
                _renderDemoList();
                const btn = document.getElementById('btn-pretrain');
                if (btn) btn.disabled = demoList.length === 0;
            }
        }).catch(() => {});

        // Checkpoint controls
        _on('btn-save-ckpt', 'click', _saveCheckpoint);
        const ckptInput = document.getElementById('ckpt-name');
        if (ckptInput) {
            ckptInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') _saveCheckpoint();
            });
        }

        // Fetch checkpoint list on load
        fetch('/api/drone/checkpoints').then(r => r.json()).then(data => {
            if (Array.isArray(data)) {
                checkpointList = data;
                _renderCheckpointList();
            }
        }).catch(() => {});

        // ── Endless Mode Controls ──────────────────
        _on('btn-endless', 'click', () => {
            const btn = document.getElementById('btn-endless');
            if (btn && btn.classList.contains('active')) {
                _send({ cmd: 'endless_stop' });
                btn.classList.remove('active');
                btn.textContent = 'ENDLESS';
            } else {
                const seed = Math.floor(Math.random() * 10000);
                _send({ cmd: 'endless_start', seed });
                if (btn) {
                    btn.classList.add('active');
                    btn.textContent = 'STOP ENDLESS';
                }
            }
        });

        // ── Weapon Controls ────────────────────────
        let weaponArmed = false;
        _on('btn-weapon', 'click', () => {
            weaponArmed = !weaponArmed;
            const btn = document.getElementById('btn-weapon');
            if (weaponArmed) {
                _send({ cmd: 'weapon_mode_on' });
                if (btn) { btn.classList.add('active'); btn.textContent = 'DISARM'; }
                if (window.Arena) window.Arena.setWeaponMode(true);
                document.body.classList.add('weapon-armed');
            } else {
                _send({ cmd: 'weapon_mode_off' });
                if (btn) { btn.classList.remove('active'); btn.textContent = 'ARM'; }
                if (window.Arena) window.Arena.setWeaponMode(false);
                document.body.classList.remove('weapon-armed');
            }
        });

        // Mouse click to fire — uses Raycaster to find aim point
        let _lastFireTime = 0;
        const _raycaster = typeof THREE !== 'undefined' ? new THREE.Raycaster() : null;
        const _mouse = typeof THREE !== 'undefined' ? new THREE.Vector2() : null;

        const canvas = document.getElementById('arena');
        if (canvas) {
            canvas.addEventListener('click', (e) => {
                if (!weaponArmed) return;
                const now = Date.now();
                if (now - _lastFireTime < 500) return;  // 0.5s cooldown
                _lastFireTime = now;

                // Compute normalized device coordinates
                const rect = canvas.getBoundingClientRect();
                const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

                // Use the drone's current position as a target proxy
                // (proper raycasting would need camera access from Arena)
                if (window._lastDroneState) {
                    const d = window._lastDroneState;
                    _send({
                        cmd: 'user_fire',
                        target_x: d.x,
                        target_y: d.y,
                        target_z: d.z,
                    });
                }
            });
        }

        // ── Adversary Controls ─────────────────────
        _on('btn-adversary', 'click', () => {
            const btn = document.getElementById('btn-adversary');
            if (btn && btn.classList.contains('active')) {
                _send({ cmd: 'adversary_off' });
                btn.classList.remove('active');
            } else {
                const lethal = document.getElementById('chk-adversary-lethal');
                _send({ cmd: 'adversary_on', lethal: lethal ? lethal.checked : false });
                if (btn) btn.classList.add('active');
            }
        });

        // ── Swarm Visualization ──────────────────
        _on('btn-swarm', 'click', () => {
            const btn = document.getElementById('btn-swarm');
            const active = btn && btn.classList.contains('active');
            _send({ cmd: 'swarm_toggle', enabled: !active });
            if (window.Arena) window.Arena.setSwarmMode(!active);
            if (btn) {
                btn.classList.toggle('active');
                btn.textContent = active ? 'SWARM' : 'SWARM OFF';
            }
        });

        // ── Training Recording Controls ───────────
        _on('btn-rec-training', 'click', () => {
            const btn = document.getElementById('btn-rec-training');
            const stopBtn = document.getElementById('btn-rec-training-stop');
            fetch('/api/drone/recordings/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: '' }),
            });
            if (btn) btn.disabled = true;
            if (btn) btn.classList.add('rec-training-active');
            if (stopBtn) stopBtn.disabled = false;
        });

        _on('btn-rec-training-stop', 'click', () => {
            const btn = document.getElementById('btn-rec-training');
            const stopBtn = document.getElementById('btn-rec-training-stop');
            fetch('/api/drone/recordings/stop', { method: 'POST' });
            if (btn) { btn.disabled = false; btn.classList.remove('rec-training-active'); }
            if (stopBtn) stopBtn.disabled = true;
        });

        // Fetch recordings on load
        fetch('/api/drone/recordings').then(r => r.json()).then(data => {
            if (Array.isArray(data)) {
                recordingList = data;
                _renderRecordingList();
            }
        }).catch(() => {});

        // ── Replay Controls ──────────────────────────
        _on('replay-pause-btn', 'click', () => {
            const badge = document.getElementById('replay-badge');
            const paused = badge && badge.textContent.includes('PAUSED');
            if (paused) {
                _send({ cmd: 'replay_resume' });
            } else {
                _send({ cmd: 'replay_pause' });
            }
        });

        _on('replay-stop-btn', 'click', () => {
            _send({ cmd: 'replay_stop' });
        });

        const replaySeek = document.getElementById('replay-seek');
        if (replaySeek) {
            replaySeek.addEventListener('mousedown', () => { replaySeek._dragging = true; });
            replaySeek.addEventListener('mouseup', () => {
                replaySeek._dragging = false;
                _send({ cmd: 'replay_seek', fraction: parseFloat(replaySeek.value) });
            });
        }

        const replaySpeed = document.getElementById('replay-speed');
        if (replaySpeed) {
            replaySpeed.addEventListener('change', () => {
                _send({ cmd: 'replay_speed', speed: parseFloat(replaySpeed.value) });
            });
        }

        // Keep-alive ping
        setInterval(() => _send({ cmd: 'ping' }), 30000);
    }

    function _initSlider(sliderId, valId, onChange) {
        const slider = document.getElementById(sliderId);
        if (slider) {
            slider.addEventListener('input', () => onChange(slider.value));
        }
    }

    // ── GPU Polling ───────────────────────────────────────────
    async function _pollGPU() {
        try {
            const r = await fetch('/api/system/stats');
            const d = await r.json();
            const pct = d.gpu_percent || 0;
            const bar = document.getElementById('gpu-bar');
            const val = document.getElementById('gpu-pct');
            if (bar) bar.style.width = pct + '%';
            if (val) val.textContent = Math.round(pct) + '%';
        } catch (_) {}
        setTimeout(_pollGPU, 5000);
    }

    // ── UI Helpers ────────────────────────────────────────────
    function _setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function _showEl(id, visible) {
        const el = document.getElementById(id);
        if (el) el.style.display = visible ? '' : 'none';
    }

    function _on(id, event, handler) {
        const el = document.getElementById(id);
        if (el) el.addEventListener(event, handler);
    }

    function _setStatusDot(connected) {
        const dot = document.getElementById('status-dot');
        if (dot) dot.className = 'status-dot' + (connected ? ' connected' : '');
    }

    function _setConnStatus(text, live) {
        const el = document.getElementById('conn-status');
        if (el) {
            el.textContent = text;
            el.className = 'conn-status' + (live ? ' live' : '');
        }
    }

    function _renderEpisodeLog() {
        const el = document.getElementById('episode-log');
        if (!el) return;
        el.innerHTML = episodeLog.slice(0, 20).map(ep => {
            return `<div class="log-row log-${ep.cls}">` +
                `<span class="log-ep">EP ${ep.episode}</span>` +
                `<span class="log-gates">${ep.gates}/${ep.totalGates}</span>` +
                `<span class="log-result">${ep.result}</span>` +
                `</div>`;
        }).join('');
    }

    function _fmtNum(n) {
        if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
        if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
        return String(n);
    }

    // ── Checkpoint Management ────────────────────────────────
    async function _saveCheckpoint() {
        const input = document.getElementById('ckpt-name');
        const name = (input ? input.value.trim() : '');
        if (!name) return;
        try {
            const r = await fetch('/api/drone/checkpoints/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name }),
            });
            const data = await r.json();
            if (data.error) {
                alert(data.error);
            } else if (input) {
                input.value = '';
            }
        } catch (e) {
            console.error('Save checkpoint failed:', e);
        }
    }

    async function _loadCheckpoint(name, mode) {
        try {
            const r = await fetch('/api/drone/checkpoints/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, mode }),
            });
            const data = await r.json();
            if (data.error) alert(data.error);
        } catch (e) {
            console.error('Load checkpoint failed:', e);
        }
    }

    async function _deleteCheckpoint(name) {
        try {
            const r = await fetch('/api/drone/checkpoints/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name }),
            });
            const data = await r.json();
            if (data.error) alert(data.error);
        } catch (e) {
            console.error('Delete checkpoint failed:', e);
        }
    }

    function _renderCheckpointList() {
        const el = document.getElementById('checkpoint-list');
        if (!el) return;

        if (checkpointList.length === 0) {
            el.innerHTML = '<div class="ckpt-empty">No saved checkpoints</div>';
            return;
        }

        el.innerHTML = checkpointList.map(ckpt => {
            const age = _timeAgo(ckpt.timestamp);
            const activeClass = ckpt.active ? ' ckpt-active' : '';
            return `<div class="ckpt-row${activeClass}" data-name="${ckpt.name}">` +
                `<div class="ckpt-info">` +
                `<span class="ckpt-name">${ckpt.name}</span>` +
                `<span class="ckpt-meta">Gen ${ckpt.generation} · Lvl ${ckpt.curriculum_level || '?'}</span>` +
                `<span class="ckpt-time">${age}</span>` +
                `</div>` +
                `<div class="ckpt-actions">` +
                `<button class="ckpt-btn ckpt-watch" title="Watch">&#9654;</button>` +
                `<button class="ckpt-btn ckpt-train" title="Continue Training">&#9654;&#9654;</button>` +
                `<button class="ckpt-btn ckpt-delete" title="Delete">&times;</button>` +
                `</div></div>`;
        }).join('');

        el.querySelectorAll('.ckpt-watch').forEach(btn => {
            btn.addEventListener('click', () => {
                const name = btn.closest('.ckpt-row').dataset.name;
                _loadCheckpoint(name, 'watch');
            });
        });
        el.querySelectorAll('.ckpt-train').forEach(btn => {
            btn.addEventListener('click', () => {
                const name = btn.closest('.ckpt-row').dataset.name;
                if (confirm(`Continue training from "${name}"?`)) {
                    _loadCheckpoint(name, 'train');
                }
            });
        });
        el.querySelectorAll('.ckpt-delete').forEach(btn => {
            btn.addEventListener('click', () => {
                const name = btn.closest('.ckpt-row').dataset.name;
                if (confirm(`Delete checkpoint "${name}"?`)) {
                    _deleteCheckpoint(name);
                }
            });
        });
    }

    function _timeAgo(timestamp) {
        const seconds = Math.floor(Date.now() / 1000 - timestamp);
        if (seconds < 60) return 'just now';
        if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
        if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ago';
        return Math.floor(seconds / 86400) + 'd ago';
    }

})();
