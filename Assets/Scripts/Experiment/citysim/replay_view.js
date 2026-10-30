/* citysim/replay_view.js -- plays back the flights citysim/playback.py records, over the cities they flew in.
   A study page's report builder inlines this file (split_report.py does); every study page has a Cities tab built
   on it (docs/studies/lessons.md 30). The page supplies its colours as CSS custom properties (TOKENS) and its data
   through RV.load(path): window.RV_INLINE[path] when the page carries the data itself (a page opened as a local
   file, where fetch is refused), otherwise fetch(path) beside the page.

   City: blocks and buildings (polygons), block flags (1 interior, 2 empty), the force law's avoidance cylinders,
   the street centreline samples a flight's first-seen times index. Flight: frames every dt of every drone's position,
   heading, shown-feed mask, the pilot's heading and waypoint; losses, contacts, decisions (a block's arrival, commit
   and candidate streets with the time each was previewed), the drones apart from the formation's main group. */
"use strict";
const RV = (() => {
  // ------------------------------------------------------------------ data
  const cache = new Map();
  function load(path) {
    if (cache.has(path)) return cache.get(path);
    const inl = window.RV_INLINE;
    const p = inl && inl[path] ? Promise.resolve(inl[path])
      : fetch(path).then(r => { if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`); return r.json(); });
    cache.set(path, p);
    p.catch(() => cache.delete(path));
    return p;
  }
  function bytes(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; }
  const i16 = b => new Int16Array(bytes(b).buffer), u16 = b => new Uint16Array(bytes(b).buffer);
  function polys(P) {
    const n = u16(P.n), xz = i16(P.xz), out = [];
    let k = 0;
    for (let i = 0; i < n.length; i++) {
      const a = new Float32Array(n[i] * 2);
      for (let j = 0; j < a.length; j++) a[j] = xz[k + j] / 10;
      k += a.length; out.push(a);
    }
    return out;
  }
  function bbox(a) { let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity; for (let j = 0; j < a.length; j += 2) { x0 = Math.min(x0, a[j]); x1 = Math.max(x1, a[j]); z0 = Math.min(z0, a[j + 1]); z1 = Math.max(z1, a[j + 1]); } return [x0, z0, x1, z1]; }
  function centroid(a) { let x = 0, z = 0; const n = a.length / 2; for (let j = 0; j < a.length; j += 2) { x += a[j]; z += a[j + 1]; } return [x / n, z / n]; }
  function area(a) { let s = 0; const n = a.length / 2; for (let i = 0; i < n; i++) { const j = (i + 1) % n; s += a[2 * i] * a[2 * j + 1] - a[2 * j] * a[2 * i + 1]; } return Math.abs(s) / 2; }
  function inside(a, x, z) { let c = false; const n = a.length / 2; for (let i = 0, j = n - 1; i < n; j = i++) { const xi = a[2 * i], zi = a[2 * i + 1], xj = a[2 * j], zj = a[2 * j + 1]; if ((zi > z) !== (zj > z) && x < (xj - xi) * (z - zi) / (zj - zi) + xi) c = !c; } return c; }
  function city(c) {
    if (c._d) return c._d;
    const d = { id: c.id, arena: c.arena, v: c.variant, p: c.p, s: c.s, seed: c.seed, stats: c.stats || {}, anatomy: c.anatomy || null };
    d.blocks = polys(c.blocks); d.bld = polys(c.buildings);
    d.flags = c.block_flags ? bytes(c.block_flags) : new Uint8Array(d.blocks.length);
    d.blockBld = c.block_building ? new Int16Array(bytes(c.block_building).buffer) : null;
    d.cyl = c.cylinders ? Float32Array.from(i16(c.cylinders), v => v / 10) : null;
    d.samples = c.samples ? Float32Array.from(i16(c.samples), v => v / 10) : null;
    d.blockBox = d.blocks.map(a => a.length ? bbox(a) : null);
    d.bldBox = d.bld.map(bbox);
    d.blockC = d.blocks.map(a => a.length ? centroid(a) : null);
    c._d = d; return d;
  }
  function flight(f) {
    if (f._d) return f._d;
    const d = Object.assign({}, f);
    d.xz = i16(f.xz); d.yaw = bytes(f.yaw); d.shown = u16(f.shown); d.body = bytes(f.body); d.wpi = u16(f.wp);
    d.seen = f.seen ? u16(f.seen) : null;
    d.apartMask = f.apart ? u16(f.apart.mask) : null;
    d.tEnd = f.t0 + (f.frames - 1) * f.dt;
    d.decisions = f.decisions || []; d.losses = f.losses || []; d.contacts = f.contacts || [];
    // the formation's centre per frame (live drones), for its path
    const n = f.n, C = new Float32Array(f.frames * 2);
    for (let k = 0; k < f.frames; k++) {
      let x = 0, z = 0, m = 0;
      for (let i = 0; i < n; i++) if (k < f.alive_to[i]) { x += d.xz[(k * n + i) * 2]; z += d.xz[(k * n + i) * 2 + 1]; m++; }
      C[2 * k] = m ? x / m / 10 : NaN; C[2 * k + 1] = m ? z / m / 10 : NaN;
    }
    d.centre = C;
    // a flight may carry its own seen points (its pilot's map, e.g. the footpath round buildings); otherwise its first
    // sights index the city's street samples
    d.seenXZ = f.seen_xz ? Float32Array.from(i16(f.seen_xz), v => v / 10) : null;
    if (d.seen) {                                   // the samples in order of first sight, for drawing a prefix
      const idx = []; for (let i = 0; i < d.seen.length; i++) if (d.seen[i] !== 65535) idx.push(i);
      idx.sort((a, b) => d.seen[a] - d.seen[b]); d.seenOrder = Uint32Array.from(idx);
    }
    f._d = d; return d;
  }
  const lerp = (a, b, t) => a + (b - a) * t;
  function frameAt(F, t) { const x = Math.max(0, Math.min(F.frames - 1, (t - F.t0) / F.dt)), i0 = Math.floor(x); return { i0, i1: Math.min(i0 + 1, F.frames - 1), a: x - i0 }; }
  function droneAt(F, fr, i) {
    if (fr.i0 >= F.alive_to[i]) return null;
    const a = fr.i1 < F.alive_to[i] ? fr.a : 0, n = F.n, k0 = (fr.i0 * n + i) * 2, k1 = (fr.i1 * n + i) * 2;
    const y0 = F.yaw[fr.i0 * n + i] * 360 / 256, y1 = F.yaw[fr.i1 * n + i] * 360 / 256;
    let dy = ((y1 - y0 + 540) % 360) - 180;
    return { i, x: lerp(F.xz[k0], F.xz[k1], a) / 10, z: lerp(F.xz[k0 + 1], F.xz[k1 + 1], a) / 10, yaw: y0 + dy * a,
             shown: (F.shown[fr.i0] >> i) & 1 };
  }
  function drones(F, t) { const fr = frameAt(F, t), out = []; for (let i = 0; i < F.n; i++) { const d = droneAt(F, fr, i); if (d) out.push(d); } return out; }
  function apartAt(F, t) { if (!F.apartMask) return 0; const A = F.apart, k = Math.max(0, Math.min(F.apartMask.length - 1, Math.floor((t - A.t0) / A.dt + 1e-6))); return F.apartMask[k]; }
  function aliveCount(F, t) { const fr = frameAt(F, t); let m = 0; for (let i = 0; i < F.n; i++) if (fr.i0 < F.alive_to[i]) m++; return m; }
  const popcount = n => { let c = 0; while (n) { n &= n - 1; c++; } return c; };
  // a flight's find range (a run's sight setting; flights recorded before it existed are 80 m) and its find time
  const findM = F => F.find_m || 80;
  const foundAt = F => { const v = F.found && F.found[`r${findM(F)}`]; return v === null || v === undefined ? null : v; };
  const fmtT = s => { s = Math.max(0, s); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };

  // ------------------------------------------------------------------ colours (CSS custom properties)
  const TOKENS = ["bg", "panel", "ink", "ink-2", "muted", "line", "grid", "chip", "map-street", "map-block", "map-edge",
    "map-plaza", "map-bldg", "map-cyl", "good", "warn", "bad"];
  let T = {};
  const themeFns = [];
  function readTokens() { const cs = getComputedStyle(document.documentElement); T = {}; for (const k of TOKENS) T[k] = cs.getPropertyValue("--" + k).trim(); }
  function colour(v) {          // a CSS colour or var(--x) to its value
    const m = /^var\((--[\w-]+)\)$/.exec(String(v).trim());
    return m ? getComputedStyle(document.documentElement).getPropertyValue(m[1]).trim() : v;
  }
  function rgba(c, a) {
    c = colour(c);
    let m = /^#([0-9a-f]{6})$/i.exec(c);
    if (m) { const v = parseInt(m[1], 16); return `rgba(${v >> 16},${(v >> 8) & 255},${v & 255},${a})`; }
    m = /^#([0-9a-f]{3})$/i.exec(c);
    if (m) { const h = m[1]; return `rgba(${parseInt(h[0] + h[0], 16)},${parseInt(h[1] + h[1], 16)},${parseInt(h[2] + h[2], 16)},${a})`; }
    m = /^rgba?\(([^)]+)\)$/.exec(c);
    if (m) { const p = m[1].split(",").map(x => x.trim()); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; }
    return c;
  }
  readTokens();
  let themeTimer = 0;
  const themeChanged = () => { clearTimeout(themeTimer); themeTimer = setTimeout(() => { readTokens(); themeFns.forEach(f => f()); }, 30); };
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", themeChanged); } catch (e) { /* old browser */ }
  new MutationObserver(themeChanged).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] });
  const onTheme = f => themeFns.push(f);

  // ------------------------------------------------------------------ a canvas map with pan and zoom
  class MapView {
    constructor(canvas, o = {}) {
      this.canvas = canvas; this.ctx = canvas.getContext("2d");
      this.aspect = o.aspect || 0.7; this.minH = o.minH || 240; this.maxH = o.maxH || 720; this.minW = o.minW || 200;
      this.onHover = o.onHover || null; this.onMove = o.onMove || null;
      this.k = 1; this.cx = 0; this.cz = 0; this.W = 0; this.H = 0; this.dpr = 1;
      this.bounds = null; this.pad = 12; this.userMoved = false; this.draw = null; this.raf = 0;
      new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
      if (o.interactive !== false) this.bind();
      this.resize();
    }
    resize() {
      const w = Math.max(this.minW, Math.floor(this.canvas.parentElement.clientWidth));
      const h = Math.round(Math.min(this.maxH, Math.max(this.minH, w * this.aspect)));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (w === this.W && h === this.H && dpr === this.dpr) return;
      this.W = w; this.H = h; this.dpr = dpr;
      this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
      this.canvas.style.height = h + "px";
      if (this.bounds && !this.userMoved) this.fit(this.bounds, this.pad);
      this.request();
    }
    fit(b, pad = 12) {
      this.bounds = b; this.pad = pad; this.userMoved = false;
      this.k = Math.min((this.W - 2 * pad) / Math.max(1, b[1] - b[0]), (this.H - 2 * pad) / Math.max(1, b[3] - b[2]));
      this.cx = (b[0] + b[1]) / 2; this.cz = (b[2] + b[3]) / 2; this.request();
    }
    look(x, z, k) { this.cx = x; this.cz = z; if (k) this.k = k; this.request(); }
    sx(x) { return this.W / 2 + (x - this.cx) * this.k; }
    sy(z) { return this.H / 2 - (z - this.cz) * this.k; }
    wx(px) { return this.cx + (px - this.W / 2) / this.k; }
    wz(py) { return this.cz - (py - this.H / 2) / this.k; }
    visible(b, m = 0) { return b && !(this.sx(b[2] + m) < -2 || this.sx(b[0] - m) > this.W + 2 || this.sy(b[1] - m) < -2 || this.sy(b[3] + m) > this.H + 2); }
    request() { if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.render(); }); }
    render() { const c = this.ctx; c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); c.clearRect(0, 0, this.W, this.H); if (this.draw) this.draw(c, this); }
    zoomBy(f, px = this.W / 2, py = this.H / 2) {
      const x = this.wx(px), z = this.wz(py);
      this.k = Math.max(0.05, Math.min(60, this.k * f));
      this.cx = x - (px - this.W / 2) / this.k; this.cz = z + (py - this.H / 2) / this.k;
      this.userMoved = true; this.request(); if (this.onMove) this.onMove();
    }
    bind() {
      const c = this.canvas, pts = new Map();
      let drag = null, pinch = null;
      c.addEventListener("pointerdown", e => {
        pts.set(e.pointerId, [e.clientX, e.clientY]); c.setPointerCapture(e.pointerId);
        if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), k: this.k }; drag = null; }
        else drag = { x: e.clientX, y: e.clientY, cx: this.cx, cz: this.cz, moved: false };
      });
      c.addEventListener("pointermove", e => {
        const r = c.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
        if (pts.has(e.pointerId)) pts.set(e.pointerId, [e.clientX, e.clientY]);
        if (pinch && pts.size === 2) {
          const [a, b] = [...pts.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
          const mx = (a[0] + b[0]) / 2 - r.left, my = (a[1] + b[1]) / 2 - r.top;
          this.zoomBy(pinch.k * d / Math.max(1, pinch.d) / this.k, mx, my); return;
        }
        if (drag) {
          const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
          if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
          if (drag.moved) { this.cx = drag.cx - dx / this.k; this.cz = drag.cz + dy / this.k; this.userMoved = true; this.request(); tipHide(); if (this.onMove) this.onMove(); }
          return;
        }
        if (this.onHover) this.onHover(this.wx(px), this.wz(py), e);
      });
      const end = e => { pts.delete(e.pointerId); if (pts.size < 2) pinch = null; drag = null; };
      c.addEventListener("pointerup", end); c.addEventListener("pointercancel", end);
      c.addEventListener("pointerleave", tipHide);
      c.addEventListener("wheel", e => { if (!(e.ctrlKey || e.metaKey)) return; e.preventDefault(); const r = c.getBoundingClientRect(); this.zoomBy(Math.exp(-e.deltaY * 0.0016), e.clientX - r.left, e.clientY - r.top); }, { passive: false });
      c.addEventListener("dblclick", e => { const r = c.getBoundingClientRect(); this.zoomBy(2, e.clientX - r.left, e.clientY - r.top); });
    }
  }

  // ------------------------------------------------------------------ the tooltip (built with textContent)
  let TIP = null;
  function tipShow(x, y, title, lines) {
    if (!TIP) { TIP = document.createElement("div"); TIP.className = "rv-tip"; TIP.hidden = true; TIP.setAttribute("role", "status"); document.body.appendChild(TIP); }
    TIP.replaceChildren();
    const t = document.createElement("div"); t.className = "t"; t.textContent = title; TIP.appendChild(t);
    for (const l of lines) { const d = document.createElement("div"); d.textContent = l; TIP.appendChild(d); }
    TIP.hidden = false;
    const pad = 14, w = TIP.offsetWidth, h = TIP.offsetHeight;
    let px = x + pad, py = y + pad;
    if (px + w > innerWidth - 8) px = x - w - pad;
    if (py + h > innerHeight - 8) py = y - h - pad;
    TIP.style.left = Math.max(8, px) + "px"; TIP.style.top = Math.max(8, py) + "px";
  }
  function tipHide() { if (TIP) TIP.hidden = true; }

  // ------------------------------------------------------------------ drawing
  function poly(c, v, a) { c.moveTo(v.sx(a[0]), v.sy(a[1])); for (let j = 2; j < a.length; j += 2) c.lineTo(v.sx(a[j]), v.sy(a[j + 1])); c.closePath(); }
  function drawCity(c, v, C, o = {}) {
    const a = C.arena;
    c.fillStyle = T.panel; c.fillRect(0, 0, v.W, v.H);
    c.fillStyle = T["map-street"]; c.fillRect(v.sx(a[0]), v.sy(a[3]), (a[1] - a[0]) * v.k, (a[3] - a[2]) * v.k);
    for (const pass of ["map-edge", "map-block", "map-plaza"]) {
      c.beginPath();
      for (let b = 0; b < C.blocks.length; b++) {
        const A = C.blocks[b]; if (!A.length || !v.visible(C.blockBox[b])) continue;
        const f = C.flags[b], kind = (f & 2) ? "map-plaza" : (f & 1) ? "map-block" : "map-edge";
        if (kind === pass) poly(c, v, A);
      }
      c.fillStyle = T[pass]; c.fill();
    }
    if (o.highlight !== undefined && o.highlight !== null && C.blocks[o.highlight] && C.blocks[o.highlight].length) {
      c.beginPath(); poly(c, v, C.blocks[o.highlight]); c.fillStyle = rgba(T.ink, 0.07); c.fill();
    }
    c.beginPath();
    for (let i = 0; i < C.bld.length; i++) if (v.visible(C.bldBox[i])) poly(c, v, C.bld[i]);
    c.fillStyle = T["map-bldg"]; c.fill();
    if (o.cylinders && C.cyl) {
      c.setLineDash([4, 3]); c.lineWidth = 1; c.strokeStyle = T["map-cyl"]; c.beginPath();
      for (let i = 0; i < C.cyl.length; i += 3) {
        const x = C.cyl[i], z = C.cyl[i + 1], r = C.cyl[i + 2];
        if (!v.visible([x - r, z - r, x + r, z + r])) continue;
        c.moveTo(v.sx(x) + r * v.k, v.sy(z)); c.arc(v.sx(x), v.sy(z), r * v.k, 0, 2 * Math.PI);
      }
      c.stroke(); c.setLineDash([]);
    }
    c.strokeStyle = T.line; c.lineWidth = 1; c.setLineDash([4, 4]);
    c.strokeRect(v.sx(a[0]), v.sy(a[3]), (a[1] - a[0]) * v.k, (a[3] - a[2]) * v.k); c.setLineDash([]);
  }
  function label(c, x, y, text, o = {}) {
    c.font = o.font || "600 11px Barlow, system-ui, sans-serif"; c.textAlign = o.align || "left"; c.textBaseline = "middle";
    if (o.halo !== false) { c.lineWidth = 3; c.strokeStyle = rgba(T.panel, 0.9); c.strokeText(text, x, y); }
    c.fillStyle = o.color || T["ink-2"]; c.fillText(text, x, y);
  }
  function drawScale(c, v) {
    const raw = Math.min(140, v.W * 0.22) / v.k, steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500];
    let m = steps[0]; for (const s of steps) if (s <= raw) m = s;
    const L = m * v.k, x = 14, y = v.H - 16;
    c.strokeStyle = T.ink; c.lineWidth = 1.5; c.beginPath(); c.moveTo(x, y - 4); c.lineTo(x, y); c.lineTo(x + L, y); c.lineTo(x + L, y - 4); c.stroke();
    label(c, x + L + 6, y - 2, `${m} m`, { color: T.ink, font: "500 11px 'IBM Plex Mono', monospace" });
    label(c, v.W - 12, v.H - 16, "N ↑", { color: T["ink-2"], align: "right" });
  }
  function arrowHead(c, x0, y0, x1, y1, s) {
    const a = Math.atan2(y1 - y0, x1 - x0);
    c.beginPath(); c.moveTo(x1, y1); c.lineTo(x1 - s * Math.cos(a - 0.45), y1 - s * Math.sin(a - 0.45)); c.lineTo(x1 - s * Math.cos(a + 0.45), y1 - s * Math.sin(a + 0.45)); c.closePath(); c.fill();
  }
  function star(c, x, y, r) {
    c.beginPath();
    for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r; c.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); }
    c.closePath();
  }
  const HFOV = 74.4, WEDGE_M = 40;

  // ------------------------------------------------------------------ the player
  /* el: an empty element; library: index.json's groups; o: { labels: {cond: text}, colors: {cond: css}, path: {
     city(id), flight(id) }, notes(flight) -> extra lines for the live panel } */
  function Player(el, library, o) {
    const label_ = c => (o.labels && o.labels[c]) || c, col = c => (o.colors && o.colors[c]) || "var(--ink)";
    el.classList.add("rv");
    el.innerHTML = `<div class="rv-groups seg" role="group" aria-label="Flight sets"></div>
      <div class="rv-head"><div class="rv-picks"></div><div class="rv-about small"></div></div>
      <div class="rv-work"><div class="rv-fig"><div class="rv-frame"><canvas aria-label="The flight from above"></canvas>
        <div class="rv-mapbar"><button type="button" data-z="in" aria-label="Zoom in">+</button><button type="button" data-z="out" aria-label="Zoom out">−</button><button type="button" data-z="fit" aria-label="Show the whole city">⤢</button></div>
        <div class="rv-loading" hidden>Loading…</div></div>
        <div class="rv-ctl"><button type="button" class="rv-play">Play</button>
          <input type="range" class="rv-t" min="0" max="100" step="0.1" value="0" aria-label="Time">
          <span class="rv-clock mono">0:00</span>
          <label>Speed <select class="rv-speed"><option value="1">1×</option><option value="2">2×</option><option value="5">5×</option><option value="10" selected>10×</option><option value="25">25×</option><option value="50">50×</option></select></label></div>
        <div class="rv-tl"></div>
        <div class="rv-layers small"></div>
        <div class="legend rv-legend"></div></div>
        <div class="rv-side"><div class="box rv-now"></div><div class="box rv-sum"></div></div></div>`;
    const $ = s => el.querySelector(s);
    const canvas = $("canvas"), slider = $(".rv-t"), clock = $(".rv-clock"), btn = $(".rv-play"), spd = $(".rv-speed");
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const layers = { seen: true, route: true, trails: true, wedges: true, cylinders: false, other: true, follow: true };
    const LAYER_TEXT = [["follow", "follow the formation"], ["seen", "streets seen so far"], ["route", "the pilot's route ahead"],
      ["trails", "trails"], ["wedges", "feeds' view (zoomed in)"], ["cylinders", "avoidance cylinders"], ["other", "the other source's drones"]];
    let G = null, entry = null, F = null, C = null, OTHER = null, t = 0, playing = false, last = 0, raf = 0, cam = null, visible = true;
    let shownGroup = null;
    const view = new MapView(canvas, { aspect: 0.66, minH: 300, maxH: 760, onHover: hover, onMove: () => { if (layers.follow) { layers.follow = false; drawLayers(); } } });
    el.querySelectorAll(".rv-mapbar button").forEach(b => b.addEventListener("click", () => {
      if (b.dataset.z === "in") view.zoomBy(1.5); else if (b.dataset.z === "out") view.zoomBy(1 / 1.5);
      else { layers.follow = false; drawLayers(); if (C) view.fit(C.arena, 10); }
    }));

    // ---- the library
    const groupsEl = $(".rv-groups");
    groupsEl.innerHTML = library.map((g, i) => `<button type="button" aria-pressed="false" data-g="${i}">${esc(g.short || g.title)}</button>`).join("");
    groupsEl.querySelectorAll("button").forEach(b => b.addEventListener("click", () => selectGroup(+b.dataset.g)));
    function esc(s) { return String(s).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch])); }
    function selectGroup(i, keep) {
      G = library[i];
      groupsEl.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.g === i)));
      const listed = G.kind === "manoeuvres" || G.kind === "block";   // flights picked by title, not by condition
      spd.value = listed ? "2" : "10";       // a manoeuvre lasts under a minute, a walk ten
      const picks = $(".rv-picks");
      if (listed) {
        picks.innerHTML = `<div class="seg">${G.flights.map((f, k) => `<button type="button" aria-pressed="false" data-f="${k}"><span class="ln" style="border-color:${col(f.cond)}"></span>${esc(f.title)}</button>`).join("")}</div>`;
        picks.querySelectorAll("button").forEach(b => b.addEventListener("click", () => selectFlight(G.flights[+b.dataset.f])));
        selectFlight(keep || G.flights[0]);
      } else {
        const conds = [...new Set(G.flights.map(f => f.cond))];
        const hasUnity = G.flights.some(f => f.source === "unity");
        picks.innerHTML = `<div class="seg rv-conds">${conds.map(c => `<button type="button" aria-pressed="false" data-c="${c}"><span class="ln" style="border-color:${col(c)}"></span>${esc(label_(c))}</button>`).join("")}</div>
          ${hasUnity ? `<div class="seg rv-src" style="margin-top:6px"><button type="button" aria-pressed="true" data-s="replica">The replica</button><button type="button" aria-pressed="false" data-s="unity">Unity (headless bench)</button></div>` : ""}`;
        picks.querySelectorAll(".rv-conds button").forEach(b => b.addEventListener("click", () => pickCond(b.dataset.c, entry ? entry.source : "replica")));
        picks.querySelectorAll(".rv-src button").forEach(b => b.addEventListener("click", () => pickCond(entry ? entry.cond : conds[0], b.dataset.s)));
        pickCond(keep ? keep.cond : (G.first && conds.includes(G.first) ? G.first : conds.includes("swarm_hop") ? "swarm_hop" : conds[0]), keep ? keep.source : "replica");
      }
    }
    function pickCond(cond, source) {
      let f = G.flights.find(x => x.cond === cond && x.source === source) || G.flights.find(x => x.cond === cond);
      selectFlight(f);
    }
    async function selectFlight(e) {
      entry = e;
      const picks = $(".rv-picks");
      if (G.kind === "manoeuvres" || G.kind === "block") picks.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(G.flights[+b.dataset.f] === e)));
      else {
        picks.querySelectorAll(".rv-conds button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.c === e.cond)));
        const both = G.flights.filter(x => x.cond === e.cond).map(x => x.source);
        picks.querySelectorAll(".rv-src button").forEach(b => { b.setAttribute("aria-pressed", String(b.dataset.s === e.source)); b.disabled = !both.includes(b.dataset.s); b.title = b.disabled ? "Not flown in Unity (the bench's lone drone cannot look round at zero stick)" : ""; });
      }
      $(".rv-about").innerHTML = G.kind === "block" ? `<p>${esc(G.about)}</p><p>${esc(e.about || "")}</p>` : `<p>${esc(G.kind === "manoeuvres" ? e.about : G.about)}</p>`;
      $(".rv-loading").hidden = false;
      try {
        const other = G.flights.find(x => x.cond === e.cond && x.source !== e.source);
        const [fj, cj, oj] = await Promise.all([load(o.path.flight(e.id)), load(o.path.city(e.city)), other ? load(o.path.flight(other.id)) : null]);
        if (entry !== e) return;                          // a later click won
        const newCity = !C || C.id !== cj.id;
        // another condition of the same walk or search keeps the moment; anything else starts from the beginning
        const same = G.kind !== "manoeuvres" && shownGroup === G.id;      // a "block" group is one block: keep the moment
        F = flight(fj); C = city(cj); OTHER = oj ? flight(oj) : null;
        shownGroup = G.id;
        t = same ? Math.min(t, F.tEnd) : 0;
        slider.max = String(F.tEnd);
        if (!same) cam = null;
        if (newCity && !layers.follow) view.fit(C.arena, 10);
        $(".rv-loading").hidden = true;
        drawLayers(); legend(); summary(); timeline(); sync();
      } catch (err) {
        $(".rv-loading").textContent = `Could not load the flight: ${err.message}`;
      }
    }

    // ---- layers and legend
    function drawLayers() {
      const show = LAYER_TEXT.filter(([k]) => k !== "cylinders" || (C && C.cyl)).filter(([k]) => k !== "other" || OTHER);
      $(".rv-layers").innerHTML = show.map(([k, txt]) => `<label><input type="checkbox" data-l="${k}" ${layers[k] ? "checked" : ""}> ${txt}</label>`).join("");
      $(".rv-layers").querySelectorAll("input").forEach(i => i.addEventListener("change", () => {
        layers[i.dataset.l] = i.checked;
        if (i.dataset.l === "follow") { cam = null; view.userMoved = false; if (!i.checked && C) view.fit(C.arena, 10); }
        sync();
      }));
    }
    function legend() {
      const cc = colour(col(F.cond));
      const items = [
        `<span><span class="sw" style="background:${cc};border-radius:50%;width:10px"></span>drone with its feed shown</span>`,
        F.n > 1 ? `<span><span class="sw" style="background:var(--panel);border:1.6px solid ${cc};border-radius:50%;width:10px"></span>inside the formation, not shown</span>` : "",
        F.apartMask ? `<span><span class="sw" style="background:transparent;border:2px solid var(--warn);border-radius:50%;width:12px;height:12px"></span>apart from the main group</span>` : "",
        `<span><span class="sw" style="background:${cc};width:6px;height:6px;border-radius:1px"></span>street seen</span>`,
        F.decisions.length ? `<span><span class="ln" style="border-color:var(--ink-2);border-top-style:dashed"></span>block's street, not yet entered</span><span><span class="ln" style="border-color:${cc}"></span>entered (the old preview)</span>` : "",
        F.decisions.some(d => d.opts.some(op => op.end)) ? `<span><span class="sw" style="background:var(--panel);border:1.8px solid var(--ink-2);width:9px;height:9px"></span>building at a street's end, filled once looked down</span>` : "",
        `<span><b style="color:var(--bad);margin-right:5px">×</b>drone lost</span>`,
        F.contacts.length ? `<span><span class="sw" style="background:var(--warn);width:7px;height:7px;transform:rotate(45deg)"></span>building contact</span>` : "",
        F.target ? `<span>★ the target (dashed: ${findM(F)} m)</span>` : "",
        OTHER ? `<span><span class="sw" style="background:transparent;border:1.5px solid var(--ink-2);width:9px;height:9px"></span>${OTHER.source === "unity" ? "Unity's" : "the replica's"} flight of the same ${F.unit || (F.kind === "search" ? "route" : "walk")}</span>` : ""];
      $(".rv-legend").innerHTML = items.filter(Boolean).join("");
    }

    // ---- the decisions
    function decisionAt(tt) {
      const D = F.decisions; let k = -1;
      for (let i = 0; i < D.length; i++) { const s = D[i].t0 !== null ? D[i].t0 : D[i].t; if (s !== null && s <= tt + 1e-6) k = i; }
      return k;
    }

    // ---- drawing
    view.draw = (c, v) => {
      if (!C) return;
      const k = decisionAt(t), Dk = k >= 0 ? F.decisions[k] : null;
      drawCity(c, v, C, { cylinders: layers.cylinders, highlight: Dk ? Dk.block : null });
      const cc = colour(col(F.cond));
      // streets (or the pilot's own map) seen so far
      const SXZ = F.seenXZ || C.samples;
      if (layers.seen && F.seenOrder && SXZ) {
        const lim = t / 0.5, s = Math.max(1.6, Math.min(3.2, 1.2 * v.k)); c.fillStyle = rgba(cc, 0.72);
        for (let j = 0; j < F.seenOrder.length; j++) {
          const i = F.seenOrder[j]; if (F.seen[i] > lim) break;
          const x = v.sx(SXZ[2 * i]), y = v.sy(SXZ[2 * i + 1]);
          if (x < -4 || y < -4 || x > v.W + 4 || y > v.H + 4) continue;
          c.fillRect(x - s / 2, y - s / 2, s, s);
        }
      }
      // visited blocks: their order along the flight
      F.decisions.forEach((d, i) => {
        if ((d.t0 !== null ? d.t0 : d.t) > t || !C.blockC[d.block] || i === k) return;
        const [x, z] = C.blockC[d.block];
        label(c, v.sx(x), v.sy(z), String(i + 1), { color: T["ink-2"], align: "center", font: "600 10px Barlow, system-ui, sans-serif" });
      });
      // the block being taken: its streets, previewed or not
      if (Dk) {
        c.lineCap = "round";
        for (const op of Dk.opts) {
          const x0 = v.sx(op.xz[0]), y0 = v.sy(op.xz[1]), x1 = v.sx(op.xz[2]), y1 = v.sy(op.xz[3]);
          const seen = op.t !== null && op.t <= t + 1e-6, chosen = op.chosen && Dk.t !== null && t >= Dk.t;
          c.lineWidth = Math.max(3, Math.min(7, 0.45 * v.k * 6)); c.setLineDash(seen ? [] : [5, 5]);
          c.strokeStyle = seen ? rgba(cc, 0.8) : rgba(T["ink-2"], 0.55);
          c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke(); c.setLineDash([]);
          if (chosen) { c.fillStyle = T.ink; c.strokeStyle = T.ink; c.lineWidth = 2; c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke(); arrowHead(c, x0, y0, x1, y1, 9); }
          if (op.end) {                                   // what stands at the street's far end: looked down yet?
            const ex = v.sx(op.end[0]), ey = v.sy(op.end[1]), down = op.te !== null && op.te !== undefined && op.te <= t + 1e-6;
            c.lineWidth = 1.2; c.strokeStyle = T.ink; c.setLineDash([2, 3]);
            c.beginPath(); c.moveTo(x1, y1); c.lineTo(ex, ey); c.stroke(); c.setLineDash([]);
            c.beginPath(); c.rect(ex - 5, ey - 5, 10, 10); c.fillStyle = down ? cc : T.panel; c.fill(); c.lineWidth = 1.8; c.strokeStyle = down ? T.ink : T["ink-2"]; c.stroke();
          }
        }
        c.lineCap = "butt";
        if (C.blocks[Dk.block] && C.blocks[Dk.block].length) { c.beginPath(); poly(c, v, C.blocks[Dk.block]); c.strokeStyle = T.ink; c.lineWidth = 1.6; c.stroke(); }
      }
      // the block it is heading for (on the way to the first, or once this one is decided)
      const nx = k < 0 ? F.decisions[0] : (Dk && Dk.t !== null && t >= Dk.t ? F.decisions[k + 1] : null);
      if (nx && C.blocks[nx.block] && C.blocks[nx.block].length) {
        c.beginPath(); poly(c, v, C.blocks[nx.block]); c.setLineDash([4, 3]); c.strokeStyle = rgba(T.ink, 0.7); c.lineWidth = 1.4; c.stroke(); c.setLineDash([]);
      }
      // the route ahead
      if (layers.route && F.wps && F.wps.length) {
        const fr = frameAt(F, t), w0 = Math.min(F.wpi[fr.i0], F.wps.length - 1), ctr = [F.centre[2 * fr.i0], F.centre[2 * fr.i0 + 1]];
        c.strokeStyle = rgba(T["ink-2"], 0.55); c.lineWidth = 1.3; c.setLineDash([3, 4]); c.beginPath();
        if (Number.isFinite(ctr[0])) c.moveTo(v.sx(ctr[0]), v.sy(ctr[1]));
        for (let j = w0; j < Math.min(F.wps.length, w0 + 8); j++) c.lineTo(v.sx(F.wps[j][0]), v.sy(F.wps[j][1]));
        c.stroke(); c.setLineDash([]);
      }
      // the target
      if (F.target) {
        const x = v.sx(F.target[0]), y = v.sy(F.target[1]), tf = foundAt(F), found = tf !== null && t >= tf;
        c.setLineDash([4, 4]); c.strokeStyle = rgba(T.ink, 0.45); c.lineWidth = 1; c.beginPath(); c.arc(x, y, findM(F) * v.k, 0, 2 * Math.PI); c.stroke(); c.setLineDash([]);
        star(c, x, y, 9); c.fillStyle = found ? T.good : T.panel; c.fill(); c.strokeStyle = found ? T.good : T.ink; c.lineWidth = 1.6; c.stroke();
        label(c, x + 12, y - 10, found ? `found at ${fmtT(tf)}` : "target", { color: found ? T.good : T.ink });
      }
      // the other source's flight of the same walk / route
      if (layers.other && OTHER) {
        const ods = drones(OTHER, t);
        c.strokeStyle = rgba(T["ink-2"], 0.9); c.lineWidth = 1.4;
        for (const d of ods) { const x = v.sx(d.x), y = v.sy(d.z); c.strokeRect(x - 4, y - 4, 8, 8); }
      }
      // the formation's path and the drones' trails
      const fr = frameAt(F, t);
      c.strokeStyle = rgba(cc, 0.42); c.lineWidth = 1.6; c.beginPath();
      let pen = false;
      for (let j = 0; j <= fr.i0; j++) { const x = F.centre[2 * j], z = F.centre[2 * j + 1]; if (!Number.isFinite(x)) { pen = false; continue; } if (!pen) { c.moveTo(v.sx(x), v.sy(z)); pen = true; } else c.lineTo(v.sx(x), v.sy(z)); }
      c.stroke();
      if (layers.trails) {
        const back = Math.round(8 / F.dt); c.strokeStyle = rgba(cc, 0.35); c.lineWidth = 1;
        for (let i = 0; i < F.n; i++) {
          c.beginPath(); let on = false;
          for (let j = Math.max(0, fr.i0 - back); j <= fr.i0; j++) { if (j >= F.alive_to[i]) break; const x = v.sx(F.xz[(j * F.n + i) * 2] / 10), y = v.sy(F.xz[(j * F.n + i) * 2 + 1] / 10); if (!on) { c.moveTo(x, y); on = true; } else c.lineTo(x, y); }
          c.stroke();
        }
      }
      const ds = drones(F, t), apart = apartAt(F, t);
      if (layers.wedges && v.k >= 1.1) {
        for (const d of ds) if (d.shown || F.n === 1) {
          const sx = v.sx(d.x), sy = v.sy(d.z), r = WEDGE_M * v.k, h = HFOV / 2 * Math.PI / 180, th = (d.yaw - 90) * Math.PI / 180;
          c.beginPath(); c.moveTo(sx, sy); c.arc(sx, sy, r, th - h, th + h); c.closePath();
          c.fillStyle = rgba(cc, 0.045); c.fill(); c.strokeStyle = rgba(cc, 0.2); c.lineWidth = 1; c.stroke();
        }
      }
      // contacts and losses so far
      for (const e of F.contacts) if (e.t <= t) {
        const x = v.sx(e.x), y = v.sy(e.z), age = t - e.t;
        c.fillStyle = T.warn; c.beginPath(); c.moveTo(x, y - 4); c.lineTo(x + 4, y); c.lineTo(x, y + 4); c.lineTo(x - 4, y); c.closePath(); c.fill();
        if (age < 2) { c.strokeStyle = rgba(T.warn, 1 - age / 2); c.lineWidth = 2; c.beginPath(); c.arc(x, y, 6 + 10 * age, 0, 2 * Math.PI); c.stroke(); }
      }
      for (const e of F.losses) if (e.t <= t) {
        const x = v.sx(e.x), y = v.sy(e.z), s = 5;
        c.strokeStyle = T.bad; c.lineWidth = 2.4; c.beginPath(); c.moveTo(x - s, y - s); c.lineTo(x + s, y + s); c.moveTo(x - s, y + s); c.lineTo(x + s, y - s); c.stroke();
      }
      for (const d of ds) {
        const x = v.sx(d.x), y = v.sy(d.z), on = d.shown || F.n === 1;
        if ((apart >> d.i) & 1) { c.strokeStyle = T.warn; c.lineWidth = 2; c.beginPath(); c.arc(x, y, 8, 0, 2 * Math.PI); c.stroke(); }
        if (on) {                                         // the feed's heading
          const th = (d.yaw - 90) * Math.PI / 180;
          c.strokeStyle = cc; c.lineWidth = 2; c.beginPath(); c.moveTo(x, y); c.lineTo(x + 9 * Math.cos(th), y + 9 * Math.sin(th)); c.stroke();
        }
        c.beginPath(); c.arc(x, y, on ? 4.4 : 3.8, 0, 2 * Math.PI); c.fillStyle = on ? cc : T.panel; c.fill(); c.lineWidth = 1.6; c.strokeStyle = cc; c.stroke();
      }
      drawScale(c, v);
    };
    function placeCamera() {
      if (!layers.follow || !F) return;
      const fr = frameAt(F, t), x = F.centre[2 * fr.i0], z = F.centre[2 * fr.i0 + 1];
      if (!Number.isFinite(x)) return;
      const span = F.span || (F.kind === "M4" || F.kind === "M3" || F.kind === "M3b" ? 170 : 240), kWant = Math.min(view.W, view.H) / span;
      if (!cam) cam = { x, z, k: kWant };
      cam.x = lerp(cam.x, x, 0.15); cam.z = lerp(cam.z, z, 0.15); cam.k = lerp(cam.k, kWant, 0.08);
      view.cx = cam.x; view.cz = cam.z; view.k = cam.k;
    }

    // ---- the readouts
    function phaseText() {
      const k = decisionAt(t), D = F.decisions;
      if (t >= F.tEnd - 1e-6) return "the flight has ended";
      if (k < 0) return D.length ? "on the way to the first block" : "";
      const d = D[k], n = k + 1, what = d.plaza ? "a plaza, crossed by the streets" : TAKE[F.cond] || "taking the block";
      if (d.t !== null && t < d.t) return `block ${n}: ${what}`;
      if (d.plaza) return `block ${n}, a plaza crossed by the streets: on to the next`;
      if (d.look !== null && d.look !== undefined)
        return `block ${n} done: ${Math.round(100 * d.look)}% of its exits looked down at the decision${d.prev !== null && d.prev !== undefined ? ` (${Math.round(100 * d.prev)}% entered)` : ""}; on to the next`;
      return `block ${n} done: ${d.prev !== null && d.prev !== undefined ? `${Math.round(100 * d.prev)}% of its streets previewed at the decision; ` : ""}on to the next`;
    }
    const TAKE = { single: "decides on arrival at the corner", single_look: "holds at the corner and looks down each street",
      swarm_street: "decides on arrival at the corner", swarm_hold: "holds at the corner", swarm_pass: "passes round the block",
      swarm_hop: "surrounds the block", swarm_hop_p2: "surrounds the block (Phase 2's setting)" };
    function live() {
      const n = aliveCount(F, t), fr = frameAt(F, t), sh = popcount(F.shown[fr.i0]);
      const lost = F.losses.filter(e => e.t <= t).length, cont = F.contacts.filter(e => e.t <= t).length;
      let seenKm = null;
      if (F.seen) { let m = 0; const lim = t / 0.5; for (let i = 0; i < F.seen.length; i++) if (F.seen[i] <= lim) m++; seenKm = m * 2 / 1000; }
      const k = decisionAt(t), d = k >= 0 ? F.decisions[k] : null;
      const nDown = d ? d.opts.filter(op => op.end && op.te !== null && op.te !== undefined && op.te <= t + 1e-6).length : 0;
      const nEnds = d ? d.opts.filter(op => op.end).length : 0;
      const nPrev = d ? d.opts.filter(op => op.t !== null && op.t <= t + 1e-6).length : 0;
      let near = null;
      if (F.target) { for (const dd of drones(F, t)) { const r = Math.hypot(dd.x - F.target[0], dd.z - F.target[1]); near = near === null ? r : Math.min(near, r); } }
      const rows = [["drones flying", `${n} / ${F.n}`], F.n > 1 ? ["feeds shown", String(sh)] : null,
        ["drones lost so far", String(lost)], F.source === "replica" ? ["building contacts so far", String(cont)] : null,
        seenKm !== null ? [F.seenXZ ? "footpath seen so far" : "street seen so far", `${seenKm.toFixed(2)} km`] : null,
        d && !d.plaza ? ["this block's streets entered", `${nPrev} of ${d.opts.length}`] : null,
        d && !d.plaza && nEnds ? ["... looked down to the building at the end", `${nDown} of ${nEnds}`] : null,
        near !== null ? ["nearest drone to the target", `${Math.round(near)} m`] : null].filter(Boolean);
      $(".rv-now").innerHTML = `<p class="eyebrow">Now, ${fmtT(t)}</p><p class="rv-phase">${esc(phaseText())}</p>
        <dl class="rv-kv">${rows.map(([a, b]) => `<dt>${a}</dt><dd class="mono">${esc(b)}</dd>`).join("")}</dl>`;
    }
    function summary() {
      const s = F.summary || {}, rows = (o.summaryRows ? o.summaryRows(F, s) : []).filter(Boolean);
      $(".rv-sum").innerHTML = `<p class="eyebrow">The whole flight · ${F.source === "unity" ? "Unity" : "the replica"}</p>
        <dl class="rv-kv">${rows.map(([a, b]) => `<dt>${a}</dt><dd class="mono">${esc(b)}</dd>`).join("")}</dl>
        ${o.summaryNote ? `<p class="small muted">${o.summaryNote(F)}</p>` : ""}`;
    }

    // ---- the timeline
    let tl = null;
    function timeline() {
      const host = $(".rv-tl"); host.innerHTML = "";
      const W = Math.max(300, host.clientWidth || 600), H = 84, m = { l: 18, r: 18, t: 6, b: 18 };
      const x = s => m.l + (W - m.l - m.r) * s / Math.max(1e-6, F.tEnd), cc = colour(col(F.cond));
      let g = "";
      const rowY = [m.t, m.t + 18, m.t + 34];
      for (const d of F.decisions) {
        const a = d.t0 !== null ? d.t0 : d.t, b = d.t;
        if (a === null || b === null) continue;
        g += `<rect x="${x(a).toFixed(1)}" y="${rowY[0]}" width="${Math.max(1.5, x(b) - x(a)).toFixed(1)}" height="14" rx="2" fill="${d.plaza ? rgba(T.muted, 0.35) : rgba(cc, 0.45)}"/>`;
        g += `<line x1="${x(b).toFixed(1)}" x2="${x(b).toFixed(1)}" y1="${rowY[0] - 2}" y2="${rowY[0] + 16}" stroke="${T.ink}" stroke-width="1.2"/>`;
      }
      if (F.apart) {
        for (const [a, b] of F.apart.apart || []) g += `<rect x="${x(a).toFixed(1)}" y="${rowY[1]}" width="${Math.max(1.5, x(b) - x(a)).toFixed(1)}" height="10" fill="${rgba(T.warn, 0.4)}"/>`;
        for (const [a, b] of F.apart.split || []) g += `<rect x="${x(a).toFixed(1)}" y="${rowY[1]}" width="${Math.max(1.5, x(b) - x(a)).toFixed(1)}" height="10" fill="${T.warn}"/>`;
      }
      for (const e of F.contacts) g += `<rect x="${(x(e.t) - 2.5).toFixed(1)}" y="${rowY[2]}" width="5" height="5" transform="rotate(45 ${x(e.t).toFixed(1)} ${rowY[2] + 2.5})" fill="${T.warn}"/>`;
      for (const e of F.losses) g += `<text x="${x(e.t).toFixed(1)}" y="${rowY[2] + 16}" text-anchor="middle" fill="${T.bad}" style="font:700 12px var(--font-body, sans-serif)">×</text>`;
      if (foundAt(F) !== null) g += `<text x="${x(foundAt(F)).toFixed(1)}" y="${rowY[2] + 16}" text-anchor="middle" fill="${T.good}" style="font:700 13px var(--font-body, sans-serif)">★</text>`;
      const step = F.tEnd > 900 ? 120 : F.tEnd > 300 ? 60 : F.tEnd > 90 ? 15 : 5;
      for (let s = 0; s <= F.tEnd + 1e-6; s += step) g += `<line x1="${x(s)}" x2="${x(s)}" y1="${H - m.b}" y2="${H - m.b + 3}" stroke="${T.line}"/><text x="${x(s)}" y="${H - 4}" text-anchor="middle" fill="${T["ink-2"]}" style="font:500 10px var(--font-mono, monospace)">${fmtT(s)}</text>`;
      g += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="${T.line}"/>`;
      g += `<line class="rv-ph" x1="0" x2="0" y1="${m.t - 4}" y2="${H - m.b}" stroke="${T.ink}" stroke-width="1.5"/>`;
      g += `<rect class="rv-hit" x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent" tabindex="0" aria-label="Timeline: arrow keys move through the flight"/>`;
      host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Timeline of the flight">${g}</svg>`;
      const svg = host.querySelector("svg"), ph = host.querySelector(".rv-ph"), hit = host.querySelector(".rv-hit");
      const seek = ev => { const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width; setT((px - m.l) / (W - m.l - m.r) * F.tEnd); };
      let down = false;
      hit.addEventListener("pointerdown", ev => { down = true; hit.setPointerCapture(ev.pointerId); seek(ev); });
      hit.addEventListener("pointermove", ev => {
        if (down) { seek(ev); return; }
        const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width, s = (px - m.l) / (W - m.l - m.r) * F.tEnd;
        const lines = [];
        const d = F.decisions.findIndex(dd => dd.t !== null && Math.abs(x(dd.t) - px) < 4);
        if (d >= 0) lines.push(F.decisions[d].plaza ? `block ${d + 1}: a plaza, crossed by the streets` : `block ${d + 1}: decided${F.decisions[d].prev !== null && F.decisions[d].prev !== undefined ? `, ${Math.round(100 * F.decisions[d].prev)}% of its streets previewed` : ""}`);
        for (const e of F.losses) if (Math.abs(x(e.t) - px) < 5) lines.push(`drone ${e.i + 1} lost${e.why && e.why !== "lost" ? ` (${WHY[e.why] || e.why})` : ""}`);
        for (const e of F.contacts) if (Math.abs(x(e.t) - px) < 4) lines.push(`drone ${e.i + 1} touches a building`);
        if (F.apart) for (const [a, b] of F.apart.split || []) if (s >= a && s <= b) { lines.push("the formation is split in two groups"); break; }
        tipShow(ev.clientX, ev.clientY, fmtT(s), lines.length ? lines : ["click to go here"]);
      });
      hit.addEventListener("pointerup", () => { down = false; });
      hit.addEventListener("pointerleave", () => { down = false; tipHide(); });
      hit.addEventListener("keydown", keys);
      tl = { ph, x };
    }
    const WHY = { drone: "collided with another drone", isolated: "too far from the others", building: "inside a building" };

    // ---- the clock
    function setT(s) { t = Math.max(0, Math.min(F.tEnd, s)); cam = null; sync(); }
    function sync() {
      if (!F) return;
      slider.value = String(t); clock.textContent = `${fmtT(t)} / ${fmtT(F.tEnd)}`;
      if (tl) { tl.ph.setAttribute("x1", tl.x(t)); tl.ph.setAttribute("x2", tl.x(t)); }
      placeCamera(); live(); view.request();
    }
    function tick(ts) {
      raf = 0; if (!playing || !visible) return;
      const dt = last ? Math.min(0.1, (ts - last) / 1000) : 0; last = ts;
      t = Math.min(F.tEnd, t + dt * +spd.value); if (t >= F.tEnd) play(false);
      sync(); if (playing) raf = requestAnimationFrame(tick);
    }
    function play(on) {
      if (!F) return;
      if (on && t >= F.tEnd - 1e-6) t = 0;
      playing = on; btn.textContent = on ? "Pause" : "Play"; last = 0;
      if (on && !raf && visible) raf = requestAnimationFrame(tick);
    }
    function keys(ev) {
      if (!F) return;
      if (ev.key === " ") { ev.preventDefault(); play(!playing); }
      else if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") { ev.preventDefault(); setT(t + (ev.key === "ArrowRight" ? 1 : -1) * (ev.shiftKey ? 30 : 5)); }
    }
    btn.addEventListener("click", () => play(!playing));
    slider.addEventListener("input", () => setT(+slider.value));
    canvas.tabIndex = 0; canvas.addEventListener("keydown", keys);
    function hover(xw, zw, ev) {
      if (!F) return;
      for (const d of drones(F, t)) if (Math.hypot(d.x - xw, d.z - zw) * view.k < 8) {
        const ap = (apartAt(F, t) >> d.i) & 1;
        return tipShow(ev.clientX, ev.clientY, `Drone ${d.i + 1}`, [F.n > 1 ? (d.shown ? "on the formation's edge: its feed is on the pilot's ring" : "inside the formation: its feed is not shown") : "the single drone",
          ap ? "apart from the formation's main group (more than 30 m of free path)" : "", `heading ${Math.round(((d.yaw % 360) + 360) % 360)}°`].filter(Boolean));
      }
      for (const e of F.losses) if (e.t <= t && Math.hypot(e.x - xw, e.z - zw) * view.k < 8) return tipShow(ev.clientX, ev.clientY, `Drone ${e.i + 1} lost at ${fmtT(e.t)}`, [WHY[e.why] || ""].filter(Boolean));
      const k = decisionAt(t);
      if (k >= 0) for (const op of F.decisions[k].opts) if (op.end && Math.hypot(op.end[0] - xw, op.end[1] - zw) * view.k < 8) {
        return tipShow(ev.clientX, ev.clientY, "What stands at this street's far end", [op.te === null || op.te === undefined ? "never looked down in this flight"
          : `looked down at ${fmtT(op.te)}${F.decisions[k].t !== null && op.te > F.decisions[k].t ? ", after the decision" : ""}`,
          // a flight that does not score the far end (a search's) carries no tf at all
          op.tf === undefined ? "" : op.tf === null ? "the street's far end never seen" : `the street's far end seen at ${fmtT(op.tf)}`].filter(Boolean));
      }
      if (k >= 0) for (const op of F.decisions[k].opts) {
        const ax = op.xz[0], az = op.xz[1], bx = op.xz[2], bz = op.xz[3], L2 = (bx - ax) ** 2 + (bz - az) ** 2;
        const u = Math.max(0, Math.min(1, ((xw - ax) * (bx - ax) + (zw - az) * (bz - az)) / Math.max(L2, 1e-9)));
        if (Math.hypot(ax + u * (bx - ax) - xw, az + u * (bz - az) - zw) * view.k < 6) {
          return tipShow(ev.clientX, ev.clientY, `One of block ${k + 1}'s streets`, [op.t === null ? "never previewed in this flight" : `previewed at ${fmtT(op.t)}${F.decisions[k].t !== null && op.t > F.decisions[k].t ? ", after the decision" : ""}`,
            op.chosen ? "the exit taken" : "", op.score !== undefined && op.score !== null ? `unseen street nearby: ${op.score.toFixed(2)} m per m of travel` : ""].filter(Boolean));
        }
      }
      tipHide();
    }
    new ResizeObserver(() => { if (F) { timeline(); sync(); } }).observe($(".rv-tl"));
    onTheme(() => { if (F) { legend(); timeline(); sync(); } });
    return {
      selectGroup, show() { visible = true; if (F) { timeline(); sync(); } if (playing && !raf) { last = 0; raf = requestAnimationFrame(tick); } },
      hide() { visible = false; if (raf) cancelAnimationFrame(raf); raf = 0; },
      start(i = 0) {
        selectGroup(i);
        if (reduced) return;
        // play the first time the player comes into view, not while it is still below the fold
        if (!("IntersectionObserver" in window)) { setTimeout(() => play(true), 400); return; }
        const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); if (!playing) play(true); } }, { threshold: 0.3 });
        io.observe(canvas);
      },
      play, get flight() { return F; }
    };
  }

  return { load, city, flight, drones, frameAt, MapView, drawCity, drawScale, label, rgba, colour, onTheme, tokens: () => T,
           Player, tipShow, tipHide, fmtT, area, inside, centroid, poly };
})();
