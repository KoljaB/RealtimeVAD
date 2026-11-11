// RealtimeVAD × RealtimeSTT demo page.
// Audio: 16 kHz AudioContext -> worklet -> int16 chunks over /ws.
// Display: everything is placed by stream time (seconds of audio sent), the same clock
// the server stamps on every event, so VAD and ASR decisions line up to the sample.

const RATE = 16000, CHUNK = 320, BLOCK = 160, WINDOW = 8;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = { cyan: css("--cyan"), green: css("--green"), violet: css("--violet"), amber: css("--amber"),
            red: css("--red"), muted: css("--muted"), line: css("--line") };

const state = {
  ctx: null, ws: null, source: null, mode: null,
  sent: 0, pending: new Float32Array(0), blockAcc: [], lastPerf: 0,
  blocks: [],          // {t, rms} per 10 ms
  frames: [],          // {t, p, quiet, inSpeech} per 32 ms
  segments: new Map(), // n -> segment record
  current: null,       // segment number in speech
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
      channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: false } });
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
  ws.onopen = () => setStatus("streaming", "ok");
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
  if (!state.segments.has(n)) state.segments.set(n, { n, pauses: [], previews: [], text: "", partial: "" });
  return state.segments.get(n);
}

function onMessage(m) {
  if (m.type === "frame") {
    state.frames.push({ t: m.time, p: m.p, quiet: m.quiet, inSpeech: m.inSpeech, voiced: m.voiced });
    if (m.voiced && state.current && liveState === "pause") setLive("speaking");
    return;
  }
  const s = m.segment != null ? segment(m.segment) : null;
  if (m.type === "start") {
    for (const old of state.segments.values()) if (old.end && !old.sileroEnd && !old.sileroOpen) markSileroOpen(old);
    s.start = m.time; state.current = m.segment;
    $("live-text").innerHTML = ""; setLive("speaking"); badge("");
  } else if (m.type === "pause") {
    s.pauses.push({ t: m.time, source: m.source });
    setLive("pause");
  } else if (m.type === "end") {
    Object.assign(s, { end: m.time, speechEnd: m.speechEnd, voiced: m.voicedS });
    state.current = null; setLive("ended");
    const endMs = (s.end - s.speechEnd) * 1000;
    tile("t-end", `${endMs.toFixed(0)} ms`);
    const before = s.previews.filter((p) => p.arrive <= s.end);
    s.lead = before.length ? (s.end - before[before.length - 1].arrive) * 1000 : null;
    tile("t-lead", s.lead != null ? `${s.lead.toFixed(0)} ms` : "after end");
    renderTurn(s);
  } else if (m.type === "sileroEnd") {
    s.sileroEnd = m.time;
    const later = s.end != null ? (s.sileroEnd - s.end) * 1000 : 0;
    tile("t-silero", later < 1 ? "same time" : `+${later.toFixed(0)} ms`);
    renderTurn(s);
  } else if (m.type === "asr") {
    onAsr(s, m);
  }
}

function onAsr(s, m) {
  if (m.event === "error") return toast(m.text);
  if (m.event === "partial") { s.partial = m.text; if (!s.text) showLive(s); return; }
  if (m.event === "completion") { s.done = true; setLive("done"); renderTurn(s); return; }
  if (m.source === "live_flush") {                    // preview (flush) or previewRefined
    s.previews.push({ kind: m.event, pause: m.pauseTime, arrive: m.now, ms: m.latencyMs, text: m.text });
    s.text = m.text;
    if (m.event === "preview") { tile("t-preview", `${m.latencyMs.toFixed(0)} ms`); badge(`pause → text ${m.latencyMs.toFixed(0)} ms`); }
    showLive(s, true);
  } else if (m.event === "previewRecord") {
    s.record = m.text;                                // full-model text for the same pause
  } else if (m.event === "preview") {                 // full-model answer at finalize
    s.final = m.text; showLive(s, true); renderTurn(s);
  }
}

// ---------------------------------------------------------------- page text
let liveState = "listening";
function setLive(kind) {
  liveState = kind;
  const label = { speaking: "speaking", pause: "pause · transcribing", ended: "turn ended", done: "turn complete" }[kind] || kind;
  const el = $("live-state"); el.textContent = label;
  el.className = { speaking: "speaking", pause: "pause", ended: "done", done: "done" }[kind] || "";
}
function showLive(s, fresh) {
  if (state.current !== s.n && !s.end) return;
  const text = s.final || s.text || s.partial;
  $("live-text").innerHTML = `<span class="${fresh ? "fresh" : ""}">${escape(text)}</span>`;
}
function badge(text) { const b = $("live-badge"); b.textContent = text; b.classList.toggle("show", !!text); }
function tile(id, text) { const el = $(id); el.textContent = text; el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }
function toast(text) { const t = $("toast"); t.textContent = text; t.classList.add("show"); clearTimeout(toast.timer); toast.timer = setTimeout(() => t.classList.remove("show"), 5000); }
function setStatus(text, kind = "") { const el = $("status"); el.textContent = text; el.className = `status ${kind}`; }
function escape(text) { return String(text || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]); }
function markSileroOpen(s) { s.sileroOpen = true; renderTurn(s); }

function renderTurn(s) {
  if (s.end == null) return;
  if (!s.el) { s.el = document.createElement("article"); s.el.className = "turn"; $("turns").prepend(s.el); }
  const flushes = s.previews.filter((p) => p.kind === "preview").map((p) => p.ms).sort((a, b) => a - b);
  const median = flushes.length ? flushes[Math.floor(flushes.length / 2)] : null;
  const silero = s.sileroEnd != null
    ? ((s.sileroEnd - s.end) * 1000 < 1 ? "same time" : `+${((s.sileroEnd - s.end) * 1000).toFixed(0)} ms later`)
    : s.sileroOpen ? "still listening" : "…";
  const chip = (cls, label, value) => `<span class="chip ${cls}">${label} <b>${value}</b></span>`;
  s.el.innerHTML = `<p>${escape(s.final || s.text || s.partial || "…")}</p><div class="chips">
    ${chip("", "voiced", `${s.voiced.toFixed(1)} s`)}
    ${chip("violet", "speech stopped → end", `${((s.end - s.speechEnd) * 1000).toFixed(0)} ms`)}
    ${chip("green", "transcript ready before end", s.lead != null ? `${s.lead.toFixed(0)} ms` : "–")}
    ${chip("cyan", `${flushes.length} pauses · pause → text`, median != null ? `${median.toFixed(0)} ms median` : "–")}
    ${chip("red", "Silero alone", silero)}</div>`;
}

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
  const t1 = now(), nowX = W - 70, pps = (nowX - 12) / WINDOW, t0 = t1 - WINDOW;
  const x = (t) => nowX - (t1 - t) * pps;
  const P = { top: 22, h: 78 };                       // probability lane
  const A = { top: 118, h: 200 }; A.mid = A.top + A.h / 2;
  const E = { top: A.top + A.h + 16, h: H - (A.top + A.h + 16) - 6 };
  g.clearRect(0, 0, W, H);
  prune(t0 - 2);

  // lanes and grid
  g.fillStyle = "rgba(255,255,255,0.015)";
  [P, A, E].forEach((l) => g.fillRect(12, l.top, nowX - 12, l.h));
  g.font = "11px Inter, sans-serif"; g.fillStyle = C.muted;
  g.fillText("SPEECH PROBABILITY", 16, P.top - 7); g.fillText("AUDIO", 16, A.top - 7); g.fillText("DECISIONS", 16, E.top - 5);
  g.strokeStyle = "rgba(255,255,255,0.05)"; g.lineWidth = 1;
  for (let s = Math.ceil(t0); s <= t1; s++) {
    const gx = x(s); g.beginPath(); g.moveTo(gx, P.top); g.lineTo(gx, E.top + E.h); g.stroke();
    g.fillText(`${s}s`, gx + 3, H - 4);
  }

  const segs = [...state.segments.values()].filter((s) => s.start != null && (s.end ?? t1) >= t0);

  // segment bands, and the stretch Silero alone would have kept waiting
  for (const s of segs) {
    const a = x(Math.max(s.start, t0)), b = x(s.end ?? t1);
    const grad = g.createLinearGradient(0, A.top, 0, A.top + A.h);
    grad.addColorStop(0, "rgba(56,217,245,0.10)"); grad.addColorStop(1, "rgba(56,217,245,0.02)");
    g.fillStyle = grad; g.fillRect(a, A.top, b - a, A.h);
    g.fillStyle = C.cyan; g.fillRect(a, A.top, b - a, 2);
    if (s.end != null && s.sileroEnd != null && s.sileroEnd - s.end > 0.001) {
      const c = x(s.sileroEnd);
      g.save(); g.beginPath(); g.rect(b, A.top, c - b, A.h); g.clip();
      g.strokeStyle = "rgba(255,107,122,0.35)";
      for (let k = b - A.h; k < c; k += 7) { g.beginPath(); g.moveTo(k, A.top + A.h); g.lineTo(k + A.h, A.top); g.stroke(); }
      g.restore();
    }
  }

  // waveform: one bar per 10 ms block, dB scaled, cyan inside speech
  const barW = Math.max(1, 0.01 * pps - 0.6);
  const inSeg = (t) => segs.some((s) => t >= s.start && t <= (s.end ?? t1));
  const quietPath = new Path2D(), voicePath = new Path2D();
  for (const b of state.blocks) {
    if (b.t < t0) continue;
    const h = Math.max(1, level(b.rms) * (A.h / 2 - 6));
    (inSeg(b.t) ? voicePath : quietPath).rect(x(b.t), A.mid - h, barW, 2 * h);
  }
  g.fillStyle = "#33406a"; g.fill(quietPath);
  g.save(); g.shadowColor = C.cyan; g.shadowBlur = 12;
  const vg = g.createLinearGradient(0, A.top, 0, A.top + A.h);
  vg.addColorStop(0, C.violet); vg.addColorStop(0.5, C.cyan); vg.addColorStop(1, C.violet);
  g.fillStyle = vg; g.fill(voicePath); g.restore();

  // the quiet threshold the 10 ms clock compares against (follows room noise)
  g.save(); g.strokeStyle = C.amber; g.globalAlpha = 0.8; g.setLineDash([5, 4]); g.lineWidth = 1.3;
  for (const sign of [-1, 1]) {
    g.beginPath(); let first = true;
    for (const f of state.frames) {
      if (f.t < t0 || f.quiet <= 0) continue;
      const y = A.mid + sign * level(f.quiet) * (A.h / 2 - 6);
      first ? g.moveTo(x(f.t), y) : g.lineTo(x(f.t), y); first = false;
    }
    g.stroke();
  }
  g.restore();

  // speech probability with the start / continue thresholds
  const py = (p) => P.top + P.h - p * P.h;
  g.save(); g.setLineDash([3, 5]); g.strokeStyle = "rgba(255,255,255,0.18)";
  for (const [p, label] of [[0.9, "start 0.90"], [0.45, "continue 0.45"]]) {
    g.beginPath(); g.moveTo(12, py(p)); g.lineTo(nowX, py(p)); g.stroke();
    g.fillStyle = C.muted; g.fillText(label, nowX + 6, py(p) + 4);
  }
  g.restore();
  const vis = state.frames.filter((f) => f.t >= t0);
  if (vis.length > 1) {
    const path = new Path2D();
    vis.forEach((f, i) => (i ? path.lineTo(x(f.t), py(f.p)) : path.moveTo(x(f.t), py(f.p))));
    const fill = new Path2D(path); fill.lineTo(x(vis[vis.length - 1].t), P.top + P.h); fill.lineTo(x(vis[0].t), P.top + P.h);
    const pg = g.createLinearGradient(0, P.top, 0, P.top + P.h);
    pg.addColorStop(0, "rgba(168,139,255,0.45)"); pg.addColorStop(1, "rgba(168,139,255,0)");
    g.fillStyle = pg; g.fill(fill);
    g.save(); g.strokeStyle = C.violet; g.lineWidth = 2; g.shadowColor = C.violet; g.shadowBlur = 10; g.stroke(path); g.restore();
  }

  // decisions: start, pauses -> Preview arrivals, speech stop, turn end, Silero alone
  const vline = (t, color, top, bottom, width = 1.5, dash = []) => {
    g.save(); g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash);
    g.beginPath(); g.moveTo(x(t), top); g.lineTo(x(t), bottom); g.stroke(); g.restore();
  };
  const tag = (t, text, color, y, align = "left") => {
    g.font = "600 11px Inter, sans-serif"; const w = g.measureText(text).width + 12;
    const bx = align === "left" ? x(t) + 4 : x(t) - w - 4;
    g.fillStyle = "rgba(5,7,13,0.85)"; g.fillRect(bx, y - 12, w, 17);
    g.fillStyle = color; g.fillText(text, bx + 6, y);
  };
  const laneY = E.top + E.h / 2;
  for (const s of segs) {
    if (s.start >= t0) { vline(s.start, C.cyan, P.top, E.top + E.h, 1); tag(s.start, "start", C.cyan, A.top + 16); }
    let lastLabel = -1e9;
    for (const p of s.pauses) {
      if (p.t < t0) continue;
      g.fillStyle = C.amber; g.beginPath(); const px = x(p.t);
      g.moveTo(px, laneY - 6); g.lineTo(px + 5, laneY); g.lineTo(px, laneY + 6); g.lineTo(px - 5, laneY); g.fill();
    }
    for (const pv of s.previews) {
      if (pv.kind !== "preview" || pv.pause == null || pv.arrive < t0) continue;
      const a = x(pv.pause), b = x(pv.arrive);
      g.save(); g.strokeStyle = "rgba(255,255,255,0.85)"; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(a, laneY); g.quadraticCurveTo((a + b) / 2, laneY - 18, b, laneY); g.stroke();
      g.fillStyle = "#fff"; g.shadowColor = "#fff"; g.shadowBlur = 10; g.beginPath(); g.arc(b, laneY, 3.5, 0, 7); g.fill(); g.restore();
      if (b - lastLabel > 46) { g.font = "600 10px 'JetBrains Mono', monospace"; g.fillStyle = "#fff"; g.fillText(`${pv.ms.toFixed(0)}ms`, a - 4, laneY + 17); lastLabel = b; }
    }
    if (s.end != null && s.end >= t0) {
      vline(s.speechEnd, "rgba(255,255,255,0.7)", A.top, E.top + E.h, 1, [2, 3]);
      tag(s.speechEnd, "speech stopped", "#fff", A.top + A.h - 6, "right");
      g.save(); g.shadowColor = C.green; g.shadowBlur = 14; vline(s.end, C.green, P.top, E.top + E.h, 2.5); g.restore();
      tag(s.end, `turn end +${((s.end - s.speechEnd) * 1000).toFixed(0)} ms`, C.green, A.top + 34);
      if (s.sileroEnd != null) {
        const later = (s.sileroEnd - s.end) * 1000;
        vline(s.sileroEnd, C.red, P.top, E.top + E.h, 1.5, [6, 4]);
        tag(s.sileroEnd, later < 1 ? "Silero alone: same frame" : `Silero alone +${later.toFixed(0)} ms`, C.red, A.top + 52);
      }
    }
  }

  // now
  const ng = g.createLinearGradient(0, P.top, 0, E.top + E.h);
  ng.addColorStop(0, "rgba(56,217,245,0)"); ng.addColorStop(0.5, C.cyan); ng.addColorStop(1, "rgba(56,217,245,0)");
  g.fillStyle = ng; g.fillRect(nowX, P.top, 2, E.top + E.h - P.top);
  g.fillStyle = C.cyan; g.font = "600 11px Inter, sans-serif"; g.fillText("now", nowX + 6, A.mid + 4);
}

function level(rms) { return Math.max(0, Math.min(1, (20 * Math.log10(Math.max(rms, 1e-6)) + 60) / 60)); }
function prune(before) {
  while (state.blocks.length && state.blocks[0].t < before) state.blocks.shift();
  while (state.frames.length && state.frames[0].t < before) state.frames.shift();
}
function reset() {
  Object.assign(state, { sent: 0, captured: 0, pending: new Float32Array(0), blockAcc: [], blocks: [], frames: [], current: null });
  state.segments = new Map();
  $("turns").innerHTML = ""; badge(""); setLive("listening");
  ["t-preview", "t-lead", "t-end", "t-silero"].forEach((id) => ($(id).textContent = "–"));
}

// ---------------------------------------------------------------- buttons
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
$("mic").onclick = () => toggle("mic");
$("sample").onclick = () => toggle("sample");
requestAnimationFrame(draw);
