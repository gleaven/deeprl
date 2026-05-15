/**
 * DRONERL — Pure Canvas 2D metric charts (no Chart.js dependency).
 */

const Charts = (() => {
    'use strict';

    class RingBuffer {
        constructor(maxLen) {
            this.data = [];
            this.max = maxLen;
        }
        push(v) {
            this.data.push(v);
            if (this.data.length > this.max) this.data.shift();
        }
        toArray() { return this.data; }
        get length() { return this.data.length; }
    }

    const rewardBuf      = new RingBuffer(200);
    const gateRateBuf    = new RingBuffer(200);
    const completionBuf  = new RingBuffer(200);

    function _drawLine(ctx, data, color, w, h, pad, yMin, yMax) {
        const toX = (i) => (i / (data.length - 1)) * w;
        const toY = (v) => pad + (1 - (v - yMin) / (yMax - yMin)) * (h - 2 * pad);

        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 3;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        data.forEach((v, i) => {
            const x = toX(i), y = toY(v);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Fill below
        ctx.lineTo(toX(data.length - 1), h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.globalAlpha = 0.06;
        ctx.fillStyle = color;
        ctx.fill();
        ctx.restore();
    }

    function _drawChart(canvas, buffer, color, yMinHint, yMaxHint, fmtVal) {
        if (!fmtVal) fmtVal = v => v.toFixed(1);
        const data = buffer.toArray();
        const ctx = canvas.getContext('2d');

        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(8, 8, 16, 0.8)';
        ctx.fillRect(0, 0, w, h);

        if (data.length < 2) {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
            ctx.font = "10px 'Share Tech Mono', monospace";
            ctx.textAlign = 'center';
            ctx.fillText('AWAITING DATA...', w / 2, h / 2);
            return;
        }

        let yMin = Math.min(yMinHint, ...data);
        let yMax = Math.max(yMaxHint, ...data);
        if (Math.abs(yMax - yMin) < 0.001) { yMin -= 0.5; yMax += 0.5; }
        const pad = 4;

        // Grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            const gy = pad + (h - 2 * pad) * i / 4;
            ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
        }

        // Zero line if range spans zero
        if (yMin < 0 && yMax > 0) {
            const zeroY = pad + (1 - (0 - yMin) / (yMax - yMin)) * (h - 2 * pad);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY); ctx.stroke();
            ctx.setLineDash([]);
        }

        _drawLine(ctx, data, color, w, h, pad, yMin, yMax);

        // Latest value
        const last = data[data.length - 1];
        ctx.fillStyle = color;
        ctx.font = "10px 'Share Tech Mono', monospace";
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText(fmtVal(last), w - 4, 3);

        // Y-axis labels
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.font = "8px 'Share Tech Mono', monospace";
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(yMax.toFixed(1), 2, 2);
        ctx.textBaseline = 'bottom';
        ctx.fillText(yMin.toFixed(1), 2, h - 2);
    }

    function pushReward(v)     { rewardBuf.push(v); }
    function pushGateRate(v)   { gateRateBuf.push(v); }
    function pushCompletion(v) { completionBuf.push(v); }

    function render() {
        const r = document.getElementById('chart-reward');
        const g = document.getElementById('chart-gates');
        const c = document.getElementById('chart-completion');
        if (r) _drawChart(r, rewardBuf, '#ff9900', -10, 10, v => v.toFixed(1));
        if (g) _drawChart(g, gateRateBuf, '#00ff88', 0, 10, v => v.toFixed(1));
        if (c) _drawChart(c, completionBuf, '#ffd700', 0, 1, v => (v * 100).toFixed(0) + '%');
    }

    // Render at 4fps
    setInterval(render, 250);

    return { pushReward, pushGateRate, pushCompletion };
})();
