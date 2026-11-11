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
      // as in Miep: noise suppression makes word gaps actually quiet, which the 10 ms clock needs
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
  if (!state.segments.has(n)) state.segments.set(n, { n, pauses: [], previews: [], sileroPauses: [], text: "", partial: "" });
  return state.segments.get(n);
}

// RealtimeVAD's speech-off for a turn: its last pause at or before Silero's first silent
// frame (the dot right before Silero drops); without one, its first pause after it.
function rvadOff(s) {
  const upto = s.end == null ? Infinity : s.end + 1e-6;
  if (s.sileroOff == null) return (s.pauses.filter((p) => p.t <= upto).pop() || {}).t;
  const before = s.pauses.filter((p) => p.t <= s.sileroOff + 1e-6 && p.t >= s.sileroOff - 0.6).pop();
  return before ? before.t : (s.pauses.find((p) => p.t > s.sileroOff && p.t <= upto) || {}).t;
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
    const endMs = (s.end - (rvadOff(s) ?? s.speechEnd)) * 1000;
    tile("t-end", `${endMs.toFixed(0)} ms`);
    const before = s.previews.filter((p) => p.arrive <= s.end);
    s.lead = before.length ? (s.end - before[before.length - 1].arrive) * 1000 : null;
    if (s.lead != null) tile("t-lead", `${s.lead.toFixed(0)} ms`);
    renderTurn(s);
  } else if (m.type === "sileroPause") {
    s.sileroPauses.push({ pause: m.pauseTime, t: m.time });
  } else if (m.type === "sileroEnd") {
    s.sileroEnd = m.time; s.sileroOff = m.silentSince;
    const rOff = rvadOff(s);
    if (s.end != null && rOff != null) tile("t-end", `${((s.end - rOff) * 1000).toFixed(0)} ms`);
    if (rOff != null && s.sileroOff != null) {
      const d = (s.sileroOff - rOff) * 1000;
      tile("t-silero", d >= 0 ? `${d.toFixed(0)} ms` : `-${(-d).toFixed(0)} ms`);
    }
    renderTurn(s);
  } else if (m.type === "asr") {
    onAsr(s, m);
  }
}

function onAsr(s, m) {
  if (m.event === "error") return toast(m.text);
  if (m.event === "partial") { s.partial = m.text; if (!s.text) showLive(s); return; }
  if (m.event === "completion") {
    s.done = true;
    // A voice assistant ignores a segment without words (a tap, a cough): so does the demo.
    s.rejected = !(s.final || s.text || s.partial || "").trim();
    setLive(s.rejected ? "noise" : "done");
    if (s.rejected && state.current == null) $("live-text").innerHTML = `<span class="hint">Noise without words, ignored.</span>`;
    renderTurn(s); return;
  }
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
  const label = { speaking: "speaking", pause: "pause · transcribing", ended: "turn ended", done: "turn complete",
                  noise: "noise · ignored" }[kind] || kind;
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
  if (s.rejected) {
    s.el.className = "turn noise";
    s.el.innerHTML = `<p>Noise without words (${s.voiced.toFixed(2)} s voiced by Silero) · ignored, no turn</p>`;
    return;
  }
  const flushes = s.previews.filter((p) => p.kind === "preview").map((p) => p.ms).sort((a, b) => a - b);
  const median = flushes.length ? flushes[Math.floor(flushes.length / 2)] : null;
  const silero = s.sileroEnd != null
    ? ((s.sileroEnd - s.end) * 1000 < 1 ? "same time" : `+${((s.sileroEnd - s.end) * 1000).toFixed(0)} ms later`)
    : s.sileroOpen ? "still listening" : "…";
  const quiet = s.pauses.filter((p) => p.source === "quiet");
  const missed = quiet.filter((p) => !s.sileroPauses.some((q) => Math.abs(q.pause - p.t) < 1e-6)).length;
  const leads = s.sileroPauses.map((q) => (q.t - q.pause) * 1000);
  const chip = (cls, label, value) => `<span class="chip ${cls}">${label} <b>${value}</b></span>`;
  s.el.innerHTML = `<p>${escape(s.final || s.text || s.partial || "…")}</p><div class="chips">
    ${chip("", "voiced", `${s.voiced.toFixed(1)} s`)}
    ${chip("violet", "speech stopped → end", `${((s.end - s.speechEnd) * 1000).toFixed(0)} ms`)}
    ${chip("green", "transcript ready before end", s.lead != null ? `${s.lead.toFixed(0)} ms` : "–")}
    ${chip("cyan", `${flushes.length} pauses · pause → text`, median != null ? `${median.toFixed(0)} ms median` : "–")}
    ${chip("red", "pauses Silero never saw", `${missed} of ${quiet.length}`)}
    ${chip("red", "pause before Silero", leads.length ? leads.map((v) => `${v.toFixed(0)} ms`).join(", ") : "–")}
    ${chip("red", "turn end, Silero alone", silero)}</div>`;
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
  // A name column on the left; time runs right to "now".
  const gutter = 150, nowX = W - 18, pps = (nowX - gutter) / WINDOW, t1 = now(), t0 = t1 - WINDOW;
  const x = (t) => nowX - (t1 - t) * pps;
  const A = { top: 14, h: 120, name: "Audio", sub: "dashed: quiet line", color: C.muted };
  const P = { top: 148, h: 46, name: "Silero", sub: "speech probability", color: C.violet };
  const R1 = { top: 222, h: 44, name: "RealtimeVAD", sub: "speech-off ● wait ▮ end", color: C.cyan };
  const R2 = { top: 292, h: 44, name: "Silero alone", sub: "speech-off ● wait ▮ end", color: C.red };
  const T = { top: 362, h: 34, name: "Transcript", sub: "pause → text", color: "#fff" };
  A.mid = A.top + A.h / 2;
  g.clearRect(0, 0, W, H);
  prune(t0 - 2);

  // lane backgrounds, names, seconds grid
  for (const l of [A, P, R1, R2, T]) {
    g.fillStyle = l === R1 || l === R2 ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.015)";
    g.fillRect(gutter, l.top, nowX - gutter, l.h);
    g.fillStyle = l.color; g.font = `700 ${l === R1 || l === R2 ? 15 : 13}px Inter, sans-serif`;
    g.fillText(l.name, 14, l.top + l.h / 2 - 1);
    g.fillStyle = C.muted; g.font = "11px Inter, sans-serif"; g.fillText(l.sub, 14, l.top + l.h / 2 + 14);
  }
  g.strokeStyle = "rgba(255,255,255,0.05)"; g.lineWidth = 1; g.fillStyle = C.muted; g.font = "11px Inter, sans-serif";
  for (let s = Math.ceil(t0); s <= t1; s++) {
    const gx = x(s); if (gx < gutter) continue;
    g.beginPath(); g.moveTo(gx, A.top); g.lineTo(gx, T.top + T.h); g.stroke();
    g.fillText(`${s}s`, gx + 3, H - 3);
  }
  g.save(); g.beginPath(); g.rect(gutter, 0, nowX - gutter + 1, H); g.clip();   // nothing draws into the names

  const segs = [...state.segments.values()].filter((s) => s.start != null && (s.sileroEnd ?? s.end ?? t1) >= t0);
  const words = (s) => !s.rejected;

  // audio: waveform bright while RealtimeVAD is in speech, quiet threshold dashed
  const barW = Math.max(1, 0.01 * pps - 0.6);
  const inSeg = (t) => segs.some((s) => words(s) && t >= s.start && t <= (s.end ?? t1));
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
    const xa = x(Math.max(a, t0 - 1)), xb = x(b), y = row.top + 10, h = row.h - 20;
    g.save(); g.globalAlpha = 0.28; g.fillStyle = color; g.fillRect(xa, y, xb - xa, h); g.restore();
  };
  const end = (row, t, color) => {
    g.save(); g.fillStyle = color; g.shadowColor = color; g.shadowBlur = 14;
    g.fillRect(x(t) - 2, row.top + 2, 4, row.h - 4); g.restore();
  };

  // Each row: in speech ─ speech-off ● ─ striped wait to be sure ─ turn end ▮
  const wait = (row, a, b, color) => {
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
  const bracket = (a, b, y, text, color) => {
    const xa = x(a), xb = x(b);
    g.save(); g.strokeStyle = color; g.lineWidth = 2;
    g.beginPath(); g.moveTo(xa, y - 5); g.lineTo(xa, y); g.lineTo(xb, y); g.lineTo(xb, y + 5); g.stroke(); g.restore();
    label(text, xb, y + 5, color, 11);
  };

  for (const s of segs) {
    const live = words(s);
    const rOff = s.end != null ? rvadOff(s) : null;
    const sOff = s.sileroOff;
    // RealtimeVAD: solid while in speech, amber dots at its pauses (each asks for the transcript)
    bar(R1, s.start, rOff ?? s.end ?? t1, live ? C.cyan : C.muted);
    g.fillStyle = C.amber;
    for (const p of s.pauses) if (p.t >= t0 && p.t !== rOff) { g.beginPath(); g.arc(x(p.t), R1.top + R1.h / 2, 3, 0, 7); g.fill(); }
    // Silero alone: same start, solid until its first silent frame
    bar(R2, s.start, sOff ?? s.sileroEnd ?? t1, live ? C.red : C.muted);
    if (!live) { label("noise, no words: ignored", x(Math.max(s.start, t0)), R1.top + R1.h / 2 + 5, C.muted, 11); continue; }

    if (s.end != null && rOff != null) {
      wait(R1, rOff, s.end, C.cyan); off(R1, rOff, C.amber); end(R1, s.end, C.cyan);
      label("turn end", x(s.end), R1.top + R1.h / 2 + 5, C.cyan, 11);
    }
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

  // transcript: a dot when each pause's text arrived, with the round trip
  let lastLabel = -1e9;
  for (const s of segs) for (const pv of s.previews) {
    if (pv.kind !== "preview" || pv.pause == null || pv.arrive < t0) continue;
    const a = x(pv.pause), b = x(pv.arrive), y = T.top + T.h / 2;
    g.strokeStyle = "rgba(255,255,255,0.5)"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(a, y); g.lineTo(b, y); g.stroke();
    g.save(); g.fillStyle = "#fff"; g.shadowColor = "#fff"; g.shadowBlur = 8; g.beginPath(); g.arc(b, y, 4, 0, 7); g.fill(); g.restore();
    if (b - lastLabel > 56) { g.font = "600 11px 'JetBrains Mono', monospace"; g.fillStyle = "#fff"; g.fillText(`${pv.ms.toFixed(0)}ms`, b + 7, y + 4); lastLabel = b; }
  }
  g.restore();

  // now
  g.fillStyle = C.cyan; g.fillRect(nowX, A.top, 2, T.top + T.h - A.top);
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
function sendPauseMs() {
  const ms = Number($("pause-ms").value);
  $("pause-ms-value").textContent = `${ms} ms`;
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify({ pauseMs: ms }));
}
$("pause-ms").oninput = sendPauseMs;
$("mic").onclick = () => toggle("mic");
$("sample").onclick = () => toggle("sample");
requestAnimationFrame(draw);
