// RealtimeVAD demo page.
// Audio: 16 kHz AudioContext -> worklet -> int16 chunks over /ws.
// Display: everything is placed by stream time (seconds of audio sent), the same clock the
// server stamps on every event, so RealtimeVAD and Silero-alone decisions line up exactly.

const RATE = 16000, CHUNK = 320, BLOCK = 160, WINDOW = 8;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = { cyan: css("--cyan"), green: css("--green"), violet: css("--violet"), amber: css("--amber"),
            red: css("--red"), muted: css("--muted") };

const state = {
  ctx: null, ws: null, source: null, mode: null,
  sent: 0, captured: 0, pending: new Float32Array(0), blockAcc: [], lastPerf: 0,
  blocks: [],          // {t, rms} per 10 ms
  frames: [],          // {t, p, quiet} per 32 ms
  segments: new Map(), // n -> {start, pauses, end, sileroOff, sileroEnd, ...}
  current: null,       // segment number while in speech
  inferMs: [],         // recent model times per frame
  onsets: [],          // times of "maybe speech" frames
};

// ---------------------------------------------------------------- audio in
async function start(mode) {
  stop();
  const ctx = new AudioContext({ sampleRate: RATE });
  await ctx.audioWorklet.addModule("/static/capture-worklet.js");
  const capture = new AudioWorkletNode(ctx, "capture");
  const mute = ctx.createGain(); mute.gain.value = 0;
  capture.connect(mute).connect(ctx.destination);   // keeps the worklet clocked
  capture.port.onmessage = (e) => onAudio(e.data);

  let source;
  if (mode === "mic") {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: {
      // noise suppression makes the gaps after words actually quiet, which the 10 ms clock needs
      channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false } });
    source = ctx.createMediaStreamSource(stream);
    source.stop = () => stream.getTracks().forEach((t) => t.stop());
  } else {
    const response = await fetch("/sample.wav");
    if (!response.ok) throw new Error("No sample: start the server with --sample file.wav");
    source = ctx.createBufferSource();
    source.buffer = await ctx.decodeAudioData(await response.arrayBuffer());
    source.connect(ctx.destination);                  // let the viewer hear it
    source.onended = () => setTimeout(() => state.mode === "sample" && stop(), 2500);
    source.start(ctx.currentTime + 0.7);
  }
  source.connect(capture);

  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onopen = () => { setStatus("streaming", "ok"); sendPauseMs(); };
  ws.onclose = () => state.ws === ws && setStatus("disconnected", "bad");

  reset();
  Object.assign(state, { ctx, ws, source, mode, lastPerf: performance.now() });
  setButtons();
}

function stop() {
  const { ctx, ws, source } = state;
  state.ws = null; state.mode = null;
  try { source && source.stop(); } catch {}
  ws && ws.close();
  ctx && ctx.close();
  state.ctx = state.source = null;
  setStatus("idle");
  setButtons();
}

function onAudio(samples) {
  if (!state.ws) return;
  for (const s of samples) {                          // 10 ms RMS blocks for the waveform
    state.blockAcc.push(s);
    state.captured++;
    if (state.blockAcc.length === BLOCK) {
      let sum = 0; for (const v of state.blockAcc) sum += v * v;
      state.blocks.push({ t: state.captured / RATE, rms: Math.sqrt(sum / BLOCK) });
      state.blockAcc = [];
    }
  }
  const joined = new Float32Array(state.pending.length + samples.length);
  joined.set(state.pending); joined.set(samples, state.pending.length);
  let offset = 0;
  for (; offset + CHUNK <= joined.length; offset += CHUNK) {
    const pcm = new Int16Array(CHUNK);
    for (let i = 0; i < CHUNK; i++) pcm[i] = Math.max(-1, Math.min(1, joined[offset + i])) * 32767;
    if (state.ws.readyState === 1) state.ws.send(pcm.buffer);
    state.sent += CHUNK;
  }
  state.pending = joined.slice(offset);
  state.lastPerf = performance.now();
}

// ---------------------------------------------------------------- events in
function segment(n) {
  if (!state.segments.has(n)) state.segments.set(n, { n, pauses: [] });
  return state.segments.get(n);
}

// RealtimeVAD's speech-off for a turn: its last pause at or before Silero's first silent
// frame (the dot right before Silero drops); without one, its first pause after it.
function rvadOff(s) {
  const upto = s.end == null ? Infinity : s.end + 1e-6;
  if (s.sileroOff == null) return s.pauses.filter((t) => t <= upto).pop();
  const before = s.pauses.filter((t) => t <= s.sileroOff + 1e-6 && t >= s.sileroOff - 0.6).pop();
  return before ?? s.pauses.find((t) => t > s.sileroOff && t <= upto);
}

function onMessage(m) {
  if (m.type === "frame") {
    state.frames.push({ t: m.time, p: m.p, quiet: m.quiet });
    state.inferMs.push(m.inferMs); if (state.inferMs.length > 120) state.inferMs.shift();
    return;
  }
  if (m.type === "onset") { state.onsets.push(m.time); setLive("onset"); return; }
  const s = segment(m.segment);
  if (m.type === "start") {
    s.start = m.time; s.confirmed = m.confirmed; state.current = m.segment; setLive("speaking");
  } else if (m.type === "pause") {
    s.pauses.push(m.time); setLive("pause");
  } else if (m.type === "end") {
    Object.assign(s, { end: m.time, speechEnd: m.speechEnd, voiced: m.voicedS });
    state.current = null; setLive("ended");
  } else if (m.type === "sileroEnd") {
    Object.assign(s, { sileroEnd: m.time, sileroOff: m.silentSince });
  }
  if (s.end != null && s.sileroEnd != null && !s.reported) report(s);
}

// Both decisions are in: update the tiles and add the turn card.
function report(s) {
  s.reported = true;
  const off = rvadOff(s);
  s.offLead = off != null ? (s.sileroOff - off) * 1000 : null;
  s.endLead = (s.sileroEnd - s.end) * 1000;
  s.wait = off != null ? (s.end - off) * 1000 : null;
  if (s.offLead != null) tile("t-off", signed(s.offLead));
  tile("t-end", signed(s.endLead));
  if (s.wait != null) tile("t-wait", `${s.wait.toFixed(0)} ms`);
  const card = document.createElement("article");
  card.className = "turn";
  const chip = (cls, label, value) => `<span class="chip ${cls}">${label} <b>${value}</b></span>`;
  card.innerHTML = `<p>Turn ${s.n} · ${s.voiced.toFixed(1)} s of speech · ${s.pauses.length} pauses</p><div class="chips">
    ${chip("cyan", "speech-off before Silero", s.offLead != null ? signed(s.offLead) : "–")}
    ${chip("green", "turn end before Silero", signed(s.endLead))}
    ${chip("violet", "speech-off → turn end", s.wait != null ? `${s.wait.toFixed(0)} ms` : "–")}</div>`;
  $("turns").prepend(card);
}
const signed = (ms) => (Math.abs(ms) < 1 ? "same time" : ms > 0 ? `${ms.toFixed(0)} ms` : `−${(-ms).toFixed(0)} ms`);

// ---------------------------------------------------------------- page text
function setLive(kind) {
  const el = $("live-state");
  el.textContent = { onset: "onset · speech?", speaking: "speaking", pause: "pause · speech-off?", ended: "turn ended",
                     listening: "listening" }[kind];
  el.className = kind;
}
function tile(id, text) { const el = $(id); el.textContent = text; el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }
function toast(text) { const t = $("toast"); t.textContent = text; t.classList.add("show"); clearTimeout(toast.timer); toast.timer = setTimeout(() => t.classList.remove("show"), 5000); }
function setStatus(text, kind = "") { const el = $("status"); el.textContent = text; el.className = `status ${kind}`; }
setInterval(() => {
  if (!state.inferMs.length) return;
  const avg = state.inferMs.reduce((a, b) => a + b, 0) / state.inferMs.length;
  $("t-cpu").textContent = `${avg.toFixed(2)} ms`;
}, 500);

// ---------------------------------------------------------------- drawing
const canvas = $("scope"), g = canvas.getContext("2d");
let W = 0, H = 0;
function resize() {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  W = r.width; H = r.height; canvas.width = W * dpr; canvas.height = H * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0);
}
new ResizeObserver(resize).observe(canvas);

function now() {
  const ahead = state.ws ? Math.min(0.05, (performance.now() - state.lastPerf) / 1000) : 0;
  return state.sent / RATE + ahead;
}

function draw() {
  requestAnimationFrame(draw);
  if (!W) return;
  // A name column on the left; time runs right to "now".
  const gutter = 150, nowX = W - 18, pps = (nowX - gutter) / WINDOW, t1 = now(), t0 = t1 - WINDOW;
  const x = (t) => nowX - (t1 - t) * pps;
  const A = { top: 14, h: 120, name: "Audio", sub: "dashed: quiet line", color: C.muted };
  const P = { top: 150, h: 50, name: "Silero", sub: "speech probability", color: C.violet };
  const R1 = { top: 232, h: 46, name: "RealtimeVAD", sub: "speech-off ● wait ▮ end", color: C.cyan };
  const R2 = { top: 306, h: 46, name: "Silero alone", sub: "speech-off ● wait ▮ end", color: C.red };
  A.mid = A.top + A.h / 2;
  g.clearRect(0, 0, W, H);
  prune(t0 - 2);

  // lane backgrounds, names, seconds grid
  for (const l of [A, P, R1, R2]) {
    g.fillStyle = l === R1 || l === R2 ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.015)";
    g.fillRect(gutter, l.top, nowX - gutter, l.h);
    g.fillStyle = l.color; g.font = `700 ${l === R1 || l === R2 ? 15 : 13}px Inter, sans-serif`;
    g.fillText(l.name, 14, l.top + l.h / 2 - 1);
    g.fillStyle = C.muted; g.font = "11px Inter, sans-serif"; g.fillText(l.sub, 14, l.top + l.h / 2 + 14);
  }
  g.strokeStyle = "rgba(255,255,255,0.05)"; g.lineWidth = 1; g.fillStyle = C.muted; g.font = "11px Inter, sans-serif";
  for (let s = Math.ceil(t0); s <= t1; s++) {
    const gx = x(s); if (gx < gutter) continue;
    g.beginPath(); g.moveTo(gx, A.top); g.lineTo(gx, R2.top + R2.h); g.stroke();
    g.fillText(`${s}s`, gx + 3, H - 3);
  }
  g.save(); g.beginPath(); g.rect(gutter, 0, nowX - gutter + 1, H); g.clip();   // nothing draws into the names

  const segs = [...state.segments.values()].filter((s) => s.start != null && (s.sileroEnd ?? s.end ?? t1) >= t0);

  // audio: waveform bright while RealtimeVAD is in speech, quiet threshold dashed
  const barW = Math.max(1, 0.01 * pps - 0.6);
  const inSeg = (t) => segs.some((s) => t >= s.start && t <= (s.end ?? t1));
  const quietPath = new Path2D(), voicePath = new Path2D();
  for (const b of state.blocks) {
    if (b.t < t0) continue;
    const h = Math.max(1, level(b.rms) * (A.h / 2 - 3));
    (inSeg(b.t) ? voicePath : quietPath).rect(x(b.t), A.mid - h, barW, 2 * h);
  }
  g.fillStyle = "#2d3860"; g.fill(quietPath);
  g.fillStyle = C.cyan; g.fill(voicePath);
  g.save(); g.strokeStyle = C.amber; g.globalAlpha = 0.6; g.setLineDash([4, 4]); g.lineWidth = 1;
  for (const sign of [-1, 1]) {
    g.beginPath(); let first = true;
    for (const f of state.frames) {
      if (f.t < t0 || f.quiet <= 0) continue;
      const y = A.mid + sign * level(f.quiet) * (A.h / 2 - 3);
      first ? g.moveTo(x(f.t), y) : g.lineTo(x(f.t), y); first = false;
    }
    g.stroke();
  }
  g.restore();

  // Silero probability, with the 0.45 line below which a frame counts as silent
  const py = (p) => P.top + P.h - p * P.h;
  g.save(); g.setLineDash([3, 4]); g.strokeStyle = "rgba(255,107,122,0.5)";
  g.beginPath(); g.moveTo(gutter, py(0.45)); g.lineTo(nowX, py(0.45)); g.stroke(); g.restore();
  const vis = state.frames.filter((f) => f.t >= t0);
  if (vis.length > 1) {
    const path = new Path2D();
    vis.forEach((f, i) => (i ? path.lineTo(x(f.t), py(f.p)) : path.moveTo(x(f.t), py(f.p))));
    g.strokeStyle = C.violet; g.lineWidth = 1.8; g.stroke(path);
  }

  // helpers
  const label = (text, px, y, color, size = 12) => {
    g.font = `700 ${size}px 'JetBrains Mono', monospace`; const w = g.measureText(text).width;
    g.fillStyle = "rgba(5,7,13,0.92)"; g.fillRect(px + 4, y - size, w + 10, size + 7);
    g.fillStyle = color; g.fillText(text, px + 9, y + 1);
  };
  const bar = (row, a, b, color) => {
    const xa = x(Math.max(a, t0 - 1)), xb = x(b);
    g.save(); g.globalAlpha = 0.28; g.fillStyle = color; g.fillRect(xa, row.top + 10, xb - xa, row.h - 20); g.restore();
  };
  const wait = (row, a, b, color) => {                 // striped: silent, but waiting to be sure
    const xa = x(a), xb = x(b), y = row.top + 10, h = row.h - 20;
    g.save(); g.beginPath(); g.rect(xa, y, xb - xa, h); g.clip();
    g.strokeStyle = color; g.globalAlpha = 0.55; g.lineWidth = 1.5;
    for (let k = xa - h; k < xb; k += 6) { g.beginPath(); g.moveTo(k, y + h); g.lineTo(k + h, y); g.stroke(); }
    g.restore();
    if (xb - xa > 92) { g.fillStyle = "#fff"; g.font = "600 10px Inter, sans-serif"; g.fillText(`waits ${((b - a) * 1000).toFixed(0)} ms`, xa + 6, row.top + row.h / 2 + 4); }
  };
  const off = (row, t, color) => {
    g.save(); g.fillStyle = color; g.shadowColor = color; g.shadowBlur = 12;
    g.beginPath(); g.arc(x(t), row.top + row.h / 2, 6, 0, 7); g.fill(); g.restore();
  };
  const end = (row, t, color) => {
    g.save(); g.fillStyle = color; g.shadowColor = color; g.shadowBlur = 14;
    g.fillRect(x(t) - 2, row.top + 2, 4, row.h - 4); g.restore();
  };
  const bracket = (a, b, y, text, color) => {
    const xa = x(a), xb = x(b);
    g.save(); g.strokeStyle = color; g.lineWidth = 2;
    g.beginPath(); g.moveTo(xa, y - 5); g.lineTo(xa, y); g.lineTo(xb, y); g.lineTo(xb, y + 5); g.stroke(); g.restore();
    label(text, xb, y + 5, color, 11);
  };

  // onsets: a ring at each "maybe speech" frame. Confirmed ones link to their start
  // confirmation; the rest (grey) never became speech, e.g. a click or a knock.
  for (const t of state.onsets) {
    if (t < t0) continue;
    const s = segs.find((q) => q.confirmed != null && t >= q.start - 0.001 && t <= q.confirmed + 0.001);
    const y = R1.top + R1.h / 2, color = s ? C.green : C.muted;
    g.save(); g.strokeStyle = color; g.lineWidth = 2; g.beginPath(); g.arc(x(t), y, 7, 0, 7); g.stroke(); g.restore();
    if (s && t === state.onsets.find((o) => o >= s.start - 0.001)) {
      g.save(); g.strokeStyle = C.green; g.lineWidth = 2;
      g.beginPath(); g.moveTo(x(t) + 7, y); g.lineTo(x(s.confirmed), y); g.stroke();
      g.fillRect(x(s.confirmed) - 1.5, R1.top + 6, 3, R1.h - 12); g.restore();
      label(`start confirmed +${((s.confirmed - t) * 1000).toFixed(0)} ms`, x(s.confirmed), R1.top - 2, C.green, 11);
    } else if (!s) label("onset only", x(t), y + 5, C.muted, 10);
  }

  for (const s of segs) {
    const rOff = s.end != null ? rvadOff(s) : null, sOff = s.sileroOff;
    // RealtimeVAD: solid while in speech, amber dots at its pauses
    bar(R1, s.start, rOff ?? s.end ?? t1, C.cyan);
    g.fillStyle = C.amber;
    for (const t of s.pauses) if (t >= t0 && t !== rOff) { g.beginPath(); g.arc(x(t), R1.top + R1.h / 2, 3, 0, 7); g.fill(); }
    // Silero alone: same start, solid until its first silent frame
    bar(R2, s.start, sOff ?? s.sileroEnd ?? t1, C.red);
    if (s.end != null && rOff != null) {
      wait(R1, rOff, s.end, C.cyan); off(R1, rOff, C.amber); end(R1, s.end, C.cyan);
      label("turn end", x(s.end), R1.top + R1.h / 2 + 5, C.cyan, 11);
    } else if (s.end != null) end(R1, s.end, C.cyan);
    if (s.sileroEnd != null && sOff != null) {
      wait(R2, sOff, s.sileroEnd, C.red); off(R2, sOff, C.red); end(R2, s.sileroEnd, C.red);
      label("turn end", x(s.sileroEnd), R2.top + R2.h / 2 + 5, C.red, 11);
    }
    // the two comparisons, between the rows
    const mid = R1.top + R1.h + 12;
    if (rOff != null && sOff != null) {
      const d = (sOff - rOff) * 1000;
      bracket(rOff, sOff, mid - 2, Math.abs(d) < 1 ? "speech-off: same time"
        : d > 0 ? `speech-off ${d.toFixed(0)} ms sooner` : `speech-off: Silero ${(-d).toFixed(0)} ms sooner`, C.amber);
    }
    if (s.end != null && s.sileroEnd != null) {
      const d = (s.sileroEnd - s.end) * 1000;
      if (Math.abs(d) >= 1) bracket(s.end, s.sileroEnd, mid + 2, `turn end ${Math.abs(d).toFixed(0)} ms ${d > 0 ? "sooner" : "later"}`, C.green);
    }
  }
  g.restore();

  // now
  g.fillStyle = C.cyan; g.fillRect(nowX, A.top, 2, R2.top + R2.h - A.top);
}

function level(rms) { return Math.max(0, Math.min(1, (20 * Math.log10(Math.max(rms, 1e-6)) + 60) / 60)); }
function prune(before) {
  while (state.blocks.length && state.blocks[0].t < before) state.blocks.shift();
  while (state.frames.length && state.frames[0].t < before) state.frames.shift();
  while (state.onsets.length && state.onsets[0] < before) state.onsets.shift();
}
function reset() {
  Object.assign(state, { sent: 0, captured: 0, pending: new Float32Array(0), blockAcc: [], blocks: [], frames: [], current: null,
                         inferMs: [], onsets: [] });
  state.segments = new Map();
  $("turns").innerHTML = ""; setLive("listening");
  ["t-off", "t-end", "t-wait", "t-cpu"].forEach((id) => ($(id).textContent = "–"));
}

// ---------------------------------------------------------------- controls
function setButtons() {
  const mic = $("mic"), sample = $("sample");
  mic.classList.toggle("on", state.mode === "mic");
  mic.lastChild.textContent = state.mode === "mic" ? "Stop" : "Start microphone";
  sample.textContent = state.mode === "sample" ? "Stop sample" : "Play sample";
}
async function toggle(mode) {
  if (state.mode === mode) return stop();
  try { await start(mode); } catch (err) { stop(); toast(err.message || String(err)); }
}
function sendPauseMs() {                              // both sliders
  const pauseMs = Number($("pause-ms").value), startMs = Number($("start-ms").value);
  $("pause-ms-value").textContent = `${pauseMs} ms`;
  $("start-ms-value").textContent = `${startMs} ms`;
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify({ pauseMs, startMs }));
}
$("pause-ms").oninput = sendPauseMs;
$("start-ms").oninput = sendPauseMs;
$("mic").onclick = () => toggle("mic");
$("sample").onclick = () => toggle("sample");
requestAnimationFrame(draw);
