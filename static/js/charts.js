/**
 * DEEPRL — Pure Canvas 2D metric charts (no Chart.js dependency).
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

    const rewardCyanBuf      = new RingBuffer(200);
    const rewardMagentaBuf   = new RingBuffer(200);
    const winrateCyanBuf     = new RingBuffer(200);
    const winrateMagentaBuf  = new RingBuffer(200);
    const goalsBuf           = new RingBuffer(200);

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

    function _drawDualChart(canvas, buf1, buf2, color1, color2, yMinHint, yMaxHint, fmtVal) {
        if (!fmtVal) fmtVal = v => v.toFixed(1);
        const data1 = buf1.toArray();
        const data2 = buf2.toArray();
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

        const maxLen = Math.max(data1.length, data2.length);
        if (maxLen < 2) {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
            ctx.font = "10px 'Share Tech Mono', monospace";
            ctx.textAlign = 'center';
            ctx.fillText('AWAITING DATA...', w / 2, h / 2);
            return;
        }

        // Compute Y range across both datasets
        const allData = [...data1, ...data2];
        let yMin = Math.min(yMinHint, ...allData);
        let yMax = Math.max(yMaxHint, ...allData);
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

        // Draw both lines
        if (data1.length >= 2) _drawLine(ctx, data1, color1, w, h, pad, yMin, yMax);
        if (data2.length >= 2) _drawLine(ctx, data2, color2, w, h, pad, yMin, yMax);

        // Latest values — cyan top-right, magenta below
        ctx.font = "10px 'Share Tech Mono', monospace";
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        if (data1.length > 0) {
            ctx.fillStyle = color1;
            ctx.fillText(fmtVal(data1[data1.length - 1]), w - 4, 3);
        }
        if (data2.length > 0) {
            ctx.fillStyle = color2;
            ctx.fillText(fmtVal(data2[data2.length - 1]), w - 4, 15);
        }

        // Y-axis labels
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.font = "8px 'Share Tech Mono', monospace";
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(yMax.toFixed(1), 2, 2);
        ctx.textBaseline = 'bottom';
        ctx.fillText(yMin.toFixed(1), 2, h - 2);
    }

    function _drawChart(canvas, buffer, color, yMinHint, yMaxHint) {
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

        _drawLine(ctx, data, color, w, h, pad, yMin, yMax);

        // Latest value
        const last = data[data.length - 1];
        ctx.fillStyle = color;
        ctx.font = "10px 'Share Tech Mono', monospace";
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText(last.toFixed(3), w - 4, 3);

        // Y-axis labels
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.font = "8px 'Share Tech Mono', monospace";
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(yMax.toFixed(1), 2, 2);
        ctx.textBaseline = 'bottom';
        ctx.fillText(yMin.toFixed(1), 2, h - 2);
    }

    function pushRewardCyan(v)      { rewardCyanBuf.push(v); }
    function pushRewardMagenta(v)   { rewardMagentaBuf.push(v); }
    function pushWinrateCyan(v)     { winrateCyanBuf.push(v); }
    function pushWinrateMagenta(v)  { winrateMagentaBuf.push(v); }
    function pushGoals(v)           { goalsBuf.push(v); }

    function render() {
        const r = document.getElementById('chart-reward');
        const w = document.getElementById('chart-winrate');
        const g = document.getElementById('chart-goals');
        if (r) _drawDualChart(r, rewardCyanBuf, rewardMagentaBuf, '#00e5ff', '#ff00aa', -10, 10);
        if (w) _drawDualChart(w, winrateCyanBuf, winrateMagentaBuf, '#00e5ff', '#ff00aa', 0, 1, v => (v * 100).toFixed(0) + '%');
        if (g) _drawChart(g, goalsBuf,   '#ffaa00', 0, 4);
    }

    // Render at 4fps
    setInterval(render, 250);

    return { pushRewardCyan, pushRewardMagenta, pushWinrateCyan, pushWinrateMagenta, pushGoals };
})();
