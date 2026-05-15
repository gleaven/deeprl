/**
 * DEEPRL — Main controller, WebSocket, UI state.
 */

(function () {
    'use strict';

    // ── State ─────────────────────────────────────────────────
    let ws = null;
    let wsRetryDelay = 1000;
    let currentSpeed = 1;
    let watchMode = false;
    let checkpointList = [];

    const episodeLog = [];
    const MAX_LOG = 30;

    // ── Init ──────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        // Arena.init is called from arena.js module — wait a tick
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
        const url = `${proto}://${location.host}/ws/arena`;

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
        switch (msg.type) {
            case 'state':           _onState(msg); break;
            case 'episode_end':     _onEpisodeEnd(msg); break;
            case 'goal_scored':     _onGoalScored(msg); break;
            case 'training_stats':  _onTrainingStats(msg); break;
            case 'status':          _onStatus(msg); break;
            case 'checkpoint_list': _onCheckpointList(msg); break;
            case 'pong':            break;
        }
    }

    function _onState(msg) {
        if (window.Arena) {
            window.Arena.update({
                agents: msg.agents,
                ball: msg.ball,
            });
        }

        _setText('step-counter', msg.step);
        if (msg.max_steps) _setText('step-max', msg.max_steps);
        _setText('episode-counter', msg.episode);
        _setText('hdr-episode', msg.episode);
        _setText('hdr-generation', msg.generation);
        _setText('gen-badge', msg.generation);
        if (msg.total_score) {
            _setText('score-cyan', msg.total_score[0]);
            _setText('score-magenta', msg.total_score[1]);
        }
    }

    function _onEpisodeEnd(msg) {
        episodeLog.unshift({
            episode: msg.episode,
            winner: msg.winner,
            score: msg.final_score,
            steps: msg.duration_steps,
        });
        if (episodeLog.length > MAX_LOG) episodeLog.pop();
        _renderEpisodeLog();
        _setText('m-episodes', msg.episode);
    }

    function _onGoalScored(msg) {
        // Update cumulative score display immediately
        _setText('score-cyan', msg.total_score[0]);
        _setText('score-magenta', msg.total_score[1]);
        // Celebration effect
        _showCelebration(msg.team);
    }

    function _onTrainingStats(msg) {
        _setText('m-ploss', msg.policy_loss != null ? msg.policy_loss.toFixed(4) : '--');
        _setText('m-vloss', msg.value_loss != null ? msg.value_loss.toFixed(4) : '--');
        _setText('m-entropy', msg.entropy != null ? msg.entropy.toFixed(4) : '--');
        _setText('m-steps', _fmtNum(msg.total_steps || 0));
        if (msg.episodes != null) _setText('m-episodes', msg.episodes);
        _setText('hdr-winrate', msg.win_rate_cyan != null ? (msg.win_rate_cyan * 100).toFixed(0) + '%' : '--');

        // Push to charts
        if (typeof Charts !== 'undefined') {
            Charts.pushRewardCyan(msg.avg_reward_cyan || 0);
            Charts.pushRewardMagenta(msg.avg_reward_magenta || 0);
            Charts.pushWinrateCyan(msg.win_rate_cyan != null ? msg.win_rate_cyan : 0.5);
            Charts.pushWinrateMagenta(msg.win_rate_magenta != null ? msg.win_rate_magenta : 0.5);
            Charts.pushGoals(msg.goals_per_episode_avg || 0);
        }
    }

    function _onStatus(msg) {
        watchMode = msg.watch_mode || false;
        _updateTrainingButtons(msg.training, watchMode);

        if (msg.players_per_team) {
            _updateTeamSizeUI(msg.players_per_team);
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

    // ── Controls ──────────────────────────────────────────────
    function _initControls() {
        _on('btn-start', 'click', () => {
            fetch('api/training/start', { method: 'POST' });
            watchMode = false;
            _updateTrainingButtons(true, false);
        });

        _on('btn-stop', 'click', () => {
            fetch('api/training/stop', { method: 'POST' });
            watchMode = false;
            _updateTrainingButtons(false, false);
        });

        _on('btn-reset', 'click', () => {
            if (confirm('Reset training? All learned behavior will be lost.')) {
                fetch('api/training/reset', { method: 'POST' });
                watchMode = false;
                _updateTrainingButtons(true, false);
            }
        });

        // Team size buttons
        document.querySelectorAll('.team-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const ppt = parseInt(btn.dataset.ppt);
                if (confirm(`Switch to ${ppt}v${ppt}? This resets training.`)) {
                    document.querySelectorAll('.team-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    _send({ cmd: 'set_team_size', players_per_team: ppt });
                }
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

        _initSlider('sl-goal', 'sl-goal-val', v => {
            _setText('sl-goal-val', parseFloat(v).toFixed(1));
            _send({ cmd: 'set_config', goal_reward: parseFloat(v) });
        });

        _initSlider('sl-approach', 'sl-approach-val', v => {
            _setText('sl-approach-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', approach_reward: parseFloat(v) });
        });

        _initSlider('sl-ballgoal', 'sl-ballgoal-val', v => {
            _setText('sl-ballgoal-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', ball_goal_reward: parseFloat(v) });
        });

        _initSlider('sl-kick', 'sl-kick-val', v => {
            _setText('sl-kick-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', kick_reward: parseFloat(v) });
        });

        _initSlider('sl-dribble', 'sl-dribble-val', v => {
            _setText('sl-dribble-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', dribble_reward: parseFloat(v) });
        });

        _initSlider('sl-possession', 'sl-possession-val', v => {
            _setText('sl-possession-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', possession_reward: parseFloat(v) });
        });

        _initSlider('sl-juke', 'sl-juke-val', v => {
            _setText('sl-juke-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', juke_reward: parseFloat(v) });
        });

        _initSlider('sl-draw', 'sl-draw-val', v => {
            _setText('sl-draw-val', parseFloat(v).toFixed(1));
            _send({ cmd: 'set_config', draw_penalty: parseFloat(v) });
        });

        _initSlider('sl-energy', 'sl-energy-val', v => {
            _setText('sl-energy-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', energy_penalty: parseFloat(v) });
        });

        _initSlider('sl-wall', 'sl-wall-val', v => {
            _setText('sl-wall-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', wall_penalty: parseFloat(v) });
        });

        _initSlider('sl-corner', 'sl-corner-val', v => {
            _setText('sl-corner-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', corner_ball_penalty: parseFloat(v) });
        });

        _initSlider('sl-ballwall', 'sl-ballwall-val', v => {
            _setText('sl-ballwall-val', parseFloat(v).toFixed(3));
            _send({ cmd: 'set_config', ball_wall_penalty: parseFloat(v) });
        });

        _initSlider('sl-maxsteps', 'sl-maxsteps-val', v => {
            const steps = parseInt(v);
            _setText('sl-maxsteps-val', steps);
            _setText('step-max', steps);
            _send({ cmd: 'set_config', max_steps: steps });
        });

        _initSlider('sl-maxgoals', 'sl-maxgoals-val', v => {
            _setText('sl-maxgoals-val', parseInt(v));
            _send({ cmd: 'set_config', max_goals: parseInt(v) });
        });

        // Reset all sliders to defaults
        _on('btn-defaults', 'click', () => {
            const defaults = {
                'sl-lr':       { val: -3.52, fmt: v => Math.pow(10, v).toExponential(0), cfg: { learning_rate: 3e-4 } },
                'sl-entropy':  { val: 0.01,  fmt: v => v.toFixed(3), cfg: { entropy_coeff: 0.01 } },
                'sl-goal':     { val: 10,    fmt: v => v.toFixed(1), cfg: { goal_reward: 10 } },
                'sl-approach': { val: 0.02,  fmt: v => v.toFixed(3), cfg: { approach_reward: 0.02 } },
                'sl-ballgoal': { val: 0.05,  fmt: v => v.toFixed(3), cfg: { ball_goal_reward: 0.05 } },
                'sl-kick':     { val: 0.1,   fmt: v => v.toFixed(3), cfg: { kick_reward: 0.1 } },
                'sl-dribble':  { val: 0.08,  fmt: v => v.toFixed(3), cfg: { dribble_reward: 0.08 } },
                'sl-possession': { val: 0.01, fmt: v => v.toFixed(3), cfg: { possession_reward: 0.01 } },
                'sl-juke':     { val: 0.12,  fmt: v => v.toFixed(3), cfg: { juke_reward: 0.12 } },
                'sl-draw':     { val: 3.0,   fmt: v => v.toFixed(1), cfg: { draw_penalty: 3.0 } },
                'sl-energy':   { val: 0.003, fmt: v => v.toFixed(3), cfg: { energy_penalty: 0.003 } },
                'sl-wall':     { val: 0.02,  fmt: v => v.toFixed(3), cfg: { wall_penalty: 0.02 } },
                'sl-corner':   { val: 0.03,  fmt: v => v.toFixed(3), cfg: { corner_ball_penalty: 0.03 } },
                'sl-ballwall': { val: 0.05,  fmt: v => v.toFixed(3), cfg: { ball_wall_penalty: 0.05 } },
                'sl-maxsteps': { val: 500,   fmt: v => String(Math.round(v)), cfg: { max_steps: 500 } },
                'sl-maxgoals': { val: 1,     fmt: v => String(Math.round(v)), cfg: { max_goals: 1 } },
            };
            const fullCfg = {};
            for (const [id, d] of Object.entries(defaults)) {
                const slider = document.getElementById(id);
                if (slider) slider.value = d.val;
                _setText(id + '-val', d.fmt(d.val));
                Object.assign(fullCfg, d.cfg);
            }
            _send({ cmd: 'set_config', ...fullCfg });
            _setText('step-max', 500);
        });

        // Checkpoint controls
        _on('btn-save-ckpt', 'click', _saveCheckpoint);
        const ckptInput = document.getElementById('ckpt-name');
        if (ckptInput) {
            ckptInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') _saveCheckpoint();
            });
        }

        // Fetch checkpoint list on load
        fetch('api/checkpoints').then(r => r.json()).then(data => {
            if (Array.isArray(data)) {
                checkpointList = data;
                _renderCheckpointList();
            }
        }).catch(() => {});

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
            const r = await fetch('api/system/stats');
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

    function _on(id, event, handler) {
        const el = document.getElementById(id);
        if (el) el.addEventListener(event, handler);
    }

    function _setStatusDot(connected) {
        const dot = document.getElementById('status-dot');
        if (dot) {
            dot.className = 'status-dot' + (connected ? ' connected' : '');
        }
    }

    function _setConnStatus(text, live) {
        const el = document.getElementById('conn-status');
        if (el) {
            el.textContent = text;
            el.className = 'conn-status' + (live ? ' live' : '');
        }
    }

    function _showCelebration(team) {
        // CSS overlay
        const el = document.getElementById('goal-celebration');
        if (el) {
            el.removeAttribute('hidden');
            el.className = `goal-celebration team-${team}`;
            setTimeout(() => el.setAttribute('hidden', ''), 2000);
        }
        // 3D particles
        if (window.Arena) {
            window.Arena.showCelebration(team);
        }
    }

    function _renderEpisodeLog() {
        const el = document.getElementById('episode-log');
        if (!el) return;
        el.innerHTML = episodeLog.slice(0, 20).map(ep => {
            const cls = ep.winner === 'cyan' ? 'cyan' : ep.winner === 'magenta' ? 'magenta' : 'draw';
            return `<div class="log-row log-${cls}">` +
                `<span class="log-ep">EP ${ep.episode}</span>` +
                `<span class="log-score">${ep.score[0]}-${ep.score[1]}</span>` +
                `<span class="log-winner">${ep.winner.toUpperCase()}</span>` +
                `</div>`;
        }).join('');
    }

    function _fmtNum(n) {
        if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
        if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
        return String(n);
    }

    function _updateTeamSizeUI(ppt) {
        document.querySelectorAll('.team-btn').forEach(btn => {
            if (parseInt(btn.dataset.ppt) === ppt) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    // ── Checkpoint Management ────────────────────────────────
    async function _saveCheckpoint() {
        const input = document.getElementById('ckpt-name');
        const name = (input ? input.value.trim() : '');
        if (!name) return;
        try {
            const r = await fetch('api/checkpoints/save', {
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
            const r = await fetch('api/checkpoints/load', {
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
            const r = await fetch('api/checkpoints/delete', {
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
            const wr = ckpt.win_rate != null ? (ckpt.win_rate * 100).toFixed(0) + '%' : '--';
            const activeClass = ckpt.active ? ' ckpt-active' : '';
            return `<div class="ckpt-row${activeClass}" data-name="${ckpt.name}">` +
                `<div class="ckpt-info">` +
                `<span class="ckpt-name">${ckpt.name}</span>` +
                `<span class="ckpt-meta">Gen ${ckpt.generation} · ${ckpt.players_per_team}v${ckpt.players_per_team} · WR ${wr}</span>` +
                `<span class="ckpt-time">${age}</span>` +
                `</div>` +
                `<div class="ckpt-actions">` +
                `<button class="ckpt-btn ckpt-watch" title="Watch">&#9654;</button>` +
                `<button class="ckpt-btn ckpt-train" title="Continue Training">&#9654;&#9654;</button>` +
                `<button class="ckpt-btn ckpt-delete" title="Delete">&times;</button>` +
                `</div></div>`;
        }).join('');

        // Bind action buttons
        el.querySelectorAll('.ckpt-watch').forEach(btn => {
            btn.addEventListener('click', () => {
                const name = btn.closest('.ckpt-row').dataset.name;
                _loadCheckpoint(name, 'watch');
            });
        });
        el.querySelectorAll('.ckpt-train').forEach(btn => {
            btn.addEventListener('click', () => {
                const name = btn.closest('.ckpt-row').dataset.name;
                if (confirm(`Continue training from "${name}"? Current model will be replaced.`)) {
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
