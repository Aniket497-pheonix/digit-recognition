/* ══════════════════════════════════════════════════════════════════════════
   CNN BACKGROUND VISUALIZATION — SIMPLIFIED & VIVID
   ══════════════════════════════════════════════════════════════════════════ */

const CNN = (() => {
    const cv = document.getElementById('cnnBg');
    const ctx = cv.getContext('2d');
    let W, H, layers, animFrame, isActive = false, activeTimer = null;

    // Simpler architecture: 5 layers, fewer nodes each — cleaner visual
    const LAYER_COUNTS = [6, 10, 14, 10, 6];
    const LAYER_LABELS = ['Input', 'Conv', 'Pool', 'Dense', 'Out'];

    function resize() {
        W = cv.width = window.innerWidth;
        H = cv.height = window.innerHeight;
        buildLayers();
    }

    function buildLayers() {
        const n = LAYER_COUNTS.length;
        layers = LAYER_COUNTS.map((count, i) => {
            const xFrac = 0.1 + (i / (n - 1)) * 0.8; // spread 10%–90% of width
            const nodes = [];
            for (let j = 0; j < count; j++) {
                const yFrac = 0.15 + (j / (count - 1 || 1)) * 0.7;
                nodes.push({
                    x: W * xFrac,
                    y: H * yFrac,
                    active: false,
                    activation: 0,
                    phase: Math.random() * Math.PI * 2,
                    label: j === 0 ? LAYER_LABELS[i] : null
                });
            }
            return nodes;
        });
    }

    function getAccent() {
        // Read the CSS variable for node color
        const theme = document.documentElement.getAttribute('data-theme');
        return theme === 'light'
            ? { r: 234, g: 88,  b: 12  }
            : { r: 251, g: 146, b: 60  };
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        const c = getAccent();

        // ── Draw connections ──────────────────────────────────────────────
        for (let i = 0; i < layers.length - 1; i++) {
            const src = layers[i];
            const dst = layers[i + 1];

            src.forEach(a => {
                dst.forEach(b => {
                    const firing = isActive && (a.active || b.active);
                    const alpha = firing
                        ? 0.12 + Math.max(a.activation, b.activation) * 0.55
                        : 0.07;
                    const width = firing ? 1.2 : 0.6;

                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${alpha})`;
                    ctx.lineWidth = width;
                    ctx.stroke();
                });
            });
        }

        // ── Draw nodes ────────────────────────────────────────────────────
        const t = performance.now() / 1000;

        layers.forEach(layer => {
            layer.forEach(n => {
                // Decay activation
                if (n.active) {
                    n.activation -= 0.018;
                    if (n.activation <= 0) {
                        n.active = false;
                        n.activation = 0;
                    }
                }

                // Idle gentle pulse
                const idlePulse = Math.sin(t * 0.8 + n.phase) * 0.5 + 0.5; // 0..1

                const radius = n.active
                    ? 5 + n.activation * 4
                    : 3 + idlePulse * 1.2;

                const alpha = n.active
                    ? 0.4 + n.activation * 0.6
                    : 0.18 + idlePulse * 0.18;

                // Outer glow when active
                if (n.active && n.activation > 0.3) {
                    const glowR = radius * 3;
                    const grd = ctx.createRadialGradient(n.x, n.y, radius * 0.5, n.x, n.y, glowR);
                    grd.addColorStop(0, `rgba(${c.r},${c.g},${c.b},${n.activation * 0.3})`);
                    grd.addColorStop(1, `rgba(${c.r},${c.g},${c.b},0)`);
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, glowR, 0, Math.PI * 2);
                    ctx.fillStyle = grd;
                    ctx.fill();
                }

                // Node circle
                ctx.beginPath();
                ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},${alpha})`;
                ctx.fill();

                // Crisp ring
                ctx.beginPath();
                ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
                ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${alpha + 0.25})`;
                ctx.lineWidth = 1;
                ctx.stroke();
            });
        });

        animFrame = requestAnimationFrame(draw);
    }

    // Simulate a forward pass: fire neurons layer by layer
    function activate() {
        isActive = true;
        clearTimeout(activeTimer);

        // Reset
        layers.forEach(layer => layer.forEach(n => {
            n.active = false;
            n.activation = 0;
            n.phase = Math.random() * Math.PI * 2;
        }));

        // Stagger layer activations
        layers.forEach((layer, i) => {
            setTimeout(() => {
                const ratio = 0.45 + Math.random() * 0.4;
                const count = Math.ceil(layer.length * ratio);
                [...layer].sort(() => Math.random() - 0.5).slice(0, count).forEach(n => {
                    n.active = true;
                    n.activation = 0.65 + Math.random() * 0.35;
                });
            }, i * 180);
        });

        activeTimer = setTimeout(() => { isActive = false; }, 3500);

        // Activate the pulse dot in UI
        const dot = document.getElementById('pulseDot');
        if (dot) {
            dot.classList.add('active');
            setTimeout(() => dot.classList.remove('active'), 3500);
        }
    }

    window.addEventListener('resize', resize);
    resize();
    draw();

    return { activate };
})();


/* ══════════════════════════════════════════════════════════════════════════
   MAIN APP
   ══════════════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

    // ── Elements ────────────────────────────────────────────────────────────

    const canvas        = document.getElementById('drawingCanvas');
    const ctx           = canvas.getContext('2d');
    const predictBtn    = document.getElementById('predictBtn');
    const clearBtn      = document.getElementById('clearBtn');
    const brushSlider   = document.getElementById('brushSlider');
    const brushValue    = document.getElementById('brushValue');
    const brushMinus    = document.getElementById('brushMinus');
    const brushPlus     = document.getElementById('brushPlus');
    const realtimeToggle= document.getElementById('realtimeToggle');
    const predictedDigit= document.getElementById('predictedDigit');
    const confidenceValue= document.getElementById('confidenceValue');
    const probabilities = document.getElementById('probabilities');
    const warningMessage= document.getElementById('warningMessage');
    const themeToggle   = document.getElementById('themeToggle');
    const themeIcon     = themeToggle.querySelector('.theme-icon');

    // ── State ───────────────────────────────────────────────────────────────

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let debounceTimer = null;

    // ── Canvas Setup ────────────────────────────────────────────────────────

    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 12;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // ── Theme Toggle ────────────────────────────────────────────────────────

    themeToggle.addEventListener('click', () => {
        const html = document.documentElement;
        const isDark = html.getAttribute('data-theme') === 'dark';
        html.setAttribute('data-theme', isDark ? 'light' : 'dark');
        themeIcon.textContent = isDark ? '🌙' : '☀️';
    });

    // ── Brush Size ──────────────────────────────────────────────────────────

    function setBrush(val) {
        val = Math.min(30, Math.max(5, val));
        brushSlider.value = val;
        brushValue.textContent = val + 'px';
        ctx.lineWidth = val;
    }

    brushSlider.addEventListener('input', () => setBrush(parseInt(brushSlider.value)));
    brushMinus.addEventListener('click',  () => setBrush(parseInt(brushSlider.value) - 2));
    brushPlus.addEventListener('click',   () => setBrush(parseInt(brushSlider.value) + 2));

    // ── Drawing ─────────────────────────────────────────────────────────────

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        return [e.clientX - rect.left, e.clientY - rect.top];
    }

    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getPos(e);
    }

    function doDraw(e) {
        if (!isDrawing) return;
        const [x, y] = getPos(e);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        [lastX, lastY] = [x, y];
        if (realtimeToggle.checked) triggerDebounce();
    }

    function stopDrawing() { isDrawing = false; }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', doDraw);
    canvas.addEventListener('mouseup',   stopDrawing);
    canvas.addEventListener('mouseout',  stopDrawing);

    // ── Touch Support ───────────────────────────────────────────────────────

    function toMouse(type, touch) {
        canvas.dispatchEvent(new MouseEvent(type, {
            clientX: touch.clientX,
            clientY: touch.clientY
        }));
    }

    canvas.addEventListener('touchstart', e => { e.preventDefault(); toMouse('mousedown', e.touches[0]); });
    canvas.addEventListener('touchmove',  e => { e.preventDefault(); toMouse('mousemove', e.touches[0]); });
    canvas.addEventListener('touchend',   e => { e.preventDefault(); toMouse('mouseup', e.changedTouches[0]); });

    // ── Debounce ────────────────────────────────────────────────────────────

    function triggerDebounce() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => predict(), 400);
    }

    // ── Predict ─────────────────────────────────────────────────────────────

    predictBtn.addEventListener('click', () => predict());

    async function predict() {
        CNN.activate();

        const imageData = canvas.toDataURL('image/png');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();

            if (data.error) {
                warningMessage.textContent = data.error;
                return;
            }

            updateUI(data);

        } catch (err) {
            console.error('Prediction error:', err);
            warningMessage.textContent = 'Connection error';
        }
    }

    // ── Update UI ───────────────────────────────────────────────────────────

    function updateUI(data) {
        predictedDigit.classList.remove('animate');
        void predictedDigit.offsetWidth;
        predictedDigit.textContent = data.prediction;
        predictedDigit.classList.add('animate');

        confidenceValue.textContent = data.confidence.toFixed(1) + '%';
        warningMessage.textContent = data.low_confidence ? (data.message || 'Low confidence') : '';

        probabilities.innerHTML = '';
        data.probabilities.forEach((prob, digit) => {
            const row = document.createElement('div');
            row.className = 'prob-row' + (digit === data.prediction ? ' active' : '');
            row.innerHTML = `
                <span class="prob-digit">${digit}</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: 0%"></div>
                </div>
                <span class="prob-value">${prob.toFixed(1)}%</span>
            `;
            probabilities.appendChild(row);

            setTimeout(() => {
                row.querySelector('.prob-fill').style.width = Math.min(prob, 100) + '%';
            }, 50);
        });
    }

    // ── Clear ───────────────────────────────────────────────────────────────

    clearBtn.addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        predictedDigit.textContent = '?';
        confidenceValue.textContent = '--';
        warningMessage.textContent = '';
        probabilities.innerHTML = '<div class="empty-state">Draw a digit to see predictions</div>';
    });

});