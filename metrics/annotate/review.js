// Review page: a metrics run (per_pair.csv + profiles.npz, served as JSON by server.py --review) over the image.
// Separate from annotation mode; read-only.
"use strict";
const L = window.AnnLib;
const $ = id => document.getElementById(id);
const R = {runs: [], run: null, meta: null, method: null, images: [], idx: 0, stem: null, img: null, base: null, rows: [],
           sel: null, zoom: 1, panX: 0, panY: 0, ev: 0, f: {type: "", status: "", bin: "", flag: ""}};
const FLAGS = ["low_snr", "invalid", "low_contrast", "input_excursion", "geom_cross", "textured", "length_out_of_range"];

async function get(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`);
  return (r.headers.get("Content-Type") || "").includes("json") ? r.json() : r.arrayBuffer(); }
const num = v => (v === "" || v == null || v === "None") ? NaN : +v;
const bool = v => v === "True";
const vec = s => (s || "").split(" ").map(Number);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

function status(r) { return r.kind === "b" ? "b" : (r.exclusions ? "excluded" : (bool(r.pass_wide) ? "pass" : "fail")); }
function color(r) { return {pass: "#6fd08c", fail: "#ff5a4f", excluded: "#8a8f98", b: "#d36bff"}[status(r)]; }
function visible() {
  const f = R.f;
  return R.rows.filter(r => r.method === R.method && (!f.type || r.type === f.type) && (!f.status || status(r) === f.status) &&
    (!f.bin || r.bin_wide === f.bin) && (!f.flag || bool(r[f.flag])));
}

async function init() {
  R.runs = (await get("/api/review/runs")).runs;
  $("runsel").innerHTML = R.runs.map(n => `<option>${esc(n)}</option>`).join("");
  const want = new URLSearchParams(location.search).get("run");
  if (want && R.runs.includes(want)) $("runsel").value = want;
  $("runsel").onchange = () => openRun($("runsel").value);
  $("methsel").onchange = () => { R.method = $("methsel").value; renderAll(); };
  $("imgsel").onchange = () => openImage(R.images.indexOf($("imgsel").value));
  $("prev").onclick = () => openImage(R.idx - 1); $("next").onclick = () => openImage(R.idx + 1);
  for (const k of ["type", "status", "bin", "flag"]) $(`f_${k}`).onchange = e => { R.f[k] = e.target.value; renderAll(); };
  $("ev").oninput = e => { R.ev = +e.target.value; renderBase(); draw(); };
  bindCanvas();
  window.addEventListener("resize", draw);
  if (R.runs.length) await openRun($("runsel").value);
  else $("nums").textContent = "no runs in metrics/runs (run metrics/run.sh run_metrics.py first)";
}

async function openRun(name) {
  R.run = name;
  R.meta = await get(`/api/review/${encodeURIComponent(name)}/meta`);
  $("methsel").innerHTML = R.meta.methods.map(m => `<option>${esc(m)}</option>`).join("");
  R.method = R.meta.methods[0];
  R.images = R.meta.images;
  $("imgsel").innerHTML = R.images.map(s => `<option>${esc(s)}</option>`).join("");
  const want = new URLSearchParams(location.search).get("image");
  await openImage(Math.max(0, R.images.indexOf(want)));
}

async function openImage(i) {
  R.idx = (i + R.images.length) % R.images.length;
  R.stem = R.images[R.idx];
  $("imgsel").value = R.stem;
  const [meta, raw, lI, valid, rows] = await Promise.all([get(`/api/img/${R.stem}/meta.json`), get(`/api/img/${R.stem}/raw.u16.gz`),
    get(`/api/img/${R.stem}/lI.f32.gz`), get(`/api/img/${R.stem}/valid.u8.gz`),
    get(`/api/review/${encodeURIComponent(R.run)}/pairs?image=${encodeURIComponent(R.stem)}`)]);
  R.img = {W: meta.W, H: meta.H, raw: new Uint16Array(raw), lI: new Float32Array(lI), valid: new Uint8Array(valid), noise: meta.noise,
           tone_ref: meta.tone_ref, lI_range: meta.lI_range};
  R.rows = rows; R.sel = null;
  renderBase(); fit(); renderAll();
}

function renderBase() {
  const c = R.base || document.createElement("canvas");
  c.width = R.img.W; c.height = R.img.H;
  const ctx = c.getContext("2d"), id = ctx.createImageData(R.img.W, R.img.H);
  L.render(R.img, {ev: R.ev, gamma: 2.2, log: false}, id.data);
  ctx.putImageData(id, 0, 0);
  R.base = c;
}
function cv() { return $("cv"); }
function fit() { const el = cv(); R.zoom = Math.min(el.clientWidth / R.img.W, el.clientHeight / R.img.H);
  R.panX = -(el.clientWidth / R.zoom - R.img.W) / 2; R.panY = -(el.clientHeight / R.zoom - R.img.H) / 2; draw(); }
const toImg = (sx, sy) => ({x: sx / R.zoom + R.panX, y: sy / R.zoom + R.panY});
const toScr = (x, y) => ({x: (x - R.panX) * R.zoom, y: (y - R.panY) * R.zoom});

function draw() {
  if (!R.img) return;
  const el = cv(), dpr = window.devicePixelRatio || 1;
  if (el.width !== el.clientWidth * dpr) { el.width = el.clientWidth * dpr; el.height = el.clientHeight * dpr; }
  const ctx = el.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, el.width, el.height);
  ctx.setTransform(R.zoom * dpr, 0, 0, R.zoom * dpr, -R.panX * R.zoom * dpr, -R.panY * R.zoom * dpr);
  ctx.imageSmoothingEnabled = R.zoom < 1;
  ctx.drawImage(R.base, 0, 0);
  const lw = 1.5 / R.zoom;
  for (const r of visible()) {
    const sel = R.sel && r.id === R.sel.id, col = sel ? "#fff" : color(r), ax = +r.ax, ay = +r.ay, bx = +r.bx, by = +r.by;
    ctx.strokeStyle = col; ctx.lineWidth = sel ? 2 * lw : lw;
    if (status(r) === "excluded") ctx.setLineDash([4 / R.zoom, 3 / R.zoom]);
    ctx.beginPath(); ctx.moveTo(ax + .5, ay + .5); ctx.lineTo(bx + .5, by + .5); ctx.stroke();
    ctx.strokeRect(ax - +r.ar, ay - +r.ar, 2 * +r.ar + 1, 2 * +r.ar + 1); ctx.strokeRect(bx - +r.br, by - +r.br, 2 * +r.br + 1, 2 * +r.br + 1);
    ctx.setLineDash([]);
  }
  $("hud").textContent = `${R.run} · ${R.method} · ${R.stem} · zoom ${R.zoom.toFixed(2)}×`;
}

function bindCanvas() {
  const el = cv();
  el.addEventListener("contextmenu", e => e.preventDefault());
  el.addEventListener("mousedown", e => {
    const b = el.getBoundingClientRect(), sx = e.clientX - b.left, sy = e.clientY - b.top;
    if (e.button !== 0) { R.pan = {sx, sy, px: R.panX, py: R.panY}; return; }
    let best = null;
    for (const r of visible()) for (const [x, y] of [[+r.ax, +r.ay], [+r.bx, +r.by]]) {
      const q = toScr(x + .5, y + .5), d = Math.hypot(q.x - sx, q.y - sy);
      if (d < 10 && (!best || d < best.d)) best = {r, d};
    }
    if (best) select(best.r);
  });
  window.addEventListener("mousemove", e => { if (!R.pan) return; const b = el.getBoundingClientRect();
    R.panX = R.pan.px - (e.clientX - b.left - R.pan.sx) / R.zoom; R.panY = R.pan.py - (e.clientY - b.top - R.pan.sy) / R.zoom; draw(); });
  window.addEventListener("mouseup", () => { R.pan = null; });
  el.addEventListener("wheel", e => { e.preventDefault(); const b = el.getBoundingClientRect(), sx = e.clientX - b.left, sy = e.clientY - b.top, q = toImg(sx, sy);
    R.zoom = Math.min(Math.max(R.zoom * (e.deltaY < 0 ? 1.25 : 0.8), 0.1), 40); R.panX = q.x - sx / R.zoom; R.panY = q.y - sy / R.zoom; draw(); }, {passive: false});
}

async function select(r) {
  R.sel = r;
  const p = await get(`/api/review/${encodeURIComponent(R.run)}/profile?method=${encodeURIComponent(R.method)}&image=${encodeURIComponent(R.stem)}&id=${encodeURIComponent(r.id)}`);
  R.prof = p;
  renderAll();
}

function crop(cid, x, y, rr) {
  const c = $(cid), ctx = c.getContext("2d"), half = Math.max(rr + 6, 8), s = c.width / (2 * half + 1);
  ctx.imageSmoothingEnabled = false; ctx.fillStyle = "#000"; ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(R.base, x - half, y - half, 2 * half + 1, 2 * half + 1, 0, 0, c.width, c.height);
  ctx.strokeStyle = "#ffb020"; ctx.lineWidth = 2; ctx.strokeRect((half - rr) * s, (half - rr) * s, (2 * rr + 1) * s, (2 * rr + 1) * s);
}

// excursion along the profile: channel-norm distance of p(t) to the segment [a, b] (as core/profiles.excursion)
function excursion(P, a, b) {
  const d = [0, 1, 2].map(c => b[c] - a[c]), dd = d.reduce((s, v) => s + v * v, 0);
  let best = {h: -1, k: 0};
  P.forEach((p, k) => {
    if (!p.every(Number.isFinite)) return;
    let s = dd > 0 ? [0, 1, 2].reduce((t, c) => t + (p[c] - a[c]) * d[c], 0) / dd : 0;
    s = Math.min(Math.max(s, 0), 1);
    const h = Math.sqrt([0, 1, 2].reduce((t, c) => t + (p[c] - a[c] - s * d[c]) ** 2, 0) / 3);
    if (h > best.h) best = {h, k};
  });
  return best;
}

function plot(id, title, t, P, a, b, H) {
  const W = 420, Hh = 150, pad = 28, all = P.flat().concat(a, b).filter(Number.isFinite);
  const lo = Math.min(...all), hi = Math.max(...all), span = Math.max(hi - lo, 1e-3), T = t[t.length - 1] || 1;
  const X = v => pad + (W - pad - 8) * v / T, Y = v => Hh - 18 - (Hh - 30) * (v - lo) / span;
  const cols = ["#ff6b6b", "#6fd08c", "#5aa9ff"];
  let s = `<svg viewBox="0 0 ${W} ${Hh}" xmlns="http://www.w3.org/2000/svg"><text x="6" y="12" fill="#9aa1aa" font-size="11">${esc(title)}</text>`;
  for (let c = 0; c < 3; c++) {
    const pts = P.map((p, k) => Number.isFinite(p[c]) ? `${X(t[k]).toFixed(1)},${Y(p[c]).toFixed(1)}` : null).filter(Boolean).join(" ");
    s += `<polyline fill="none" stroke="${cols[c]}" stroke-width="1.5" points="${pts}"/>`;
    s += `<line x1="${X(0)}" y1="${Y(a[c])}" x2="${X(T)}" y2="${Y(b[c])}" stroke="${cols[c]}" stroke-dasharray="4 3" opacity=".6"/>`;
    s += `<circle cx="${X(0)}" cy="${Y(a[c])}" r="3" fill="${cols[c]}"/><circle cx="${X(T)}" cy="${Y(b[c])}" r="3" fill="${cols[c]}"/>`;
  }
  const ex = excursion(P, a, b);
  if (ex.h >= 0) s += `<line x1="${X(t[ex.k])}" y1="16" x2="${X(t[ex.k])}" y2="${Hh - 16}" stroke="#fff" opacity=".7"/>` +
    `<text x="${Math.min(X(t[ex.k]) + 4, W - 90)}" y="26" fill="#fff" font-size="11">H = ${Number.isFinite(H) ? H.toFixed(3) : ex.h.toFixed(3)}</text>`;
  s += `<text x="${pad}" y="${Hh - 3}" fill="#9aa1aa" font-size="10">t = 0 (A)</text><text x="${W - 70}" y="${Hh - 3}" fill="#9aa1aa" font-size="10">t = ${T.toFixed(0)} px (B)</text>`;
  $(id).innerHTML = s + "</svg>";
}

function renderSel() {
  const r = R.sel;
  if (!r) { $("nums").innerHTML = `<span class="dim">click a pair</span>`; ["plotI", "plotR", "plotL"].forEach(i => $(i).innerHTML = ""); return; }
  const st = status(r), f = FLAGS.filter(k => bool(r[k]));
  const rows = [["pair", `${r.id} · ${r.type} (${r.kind}) · ${r.annotator}`], ["status", `<span class="${st === "pass" ? "pass" : st === "fail" ? "fail" : "excl"}">${st}</span>` +
    (r.exclusions ? ` (${esc(r.exclusions)})` : "")], ["endpoint E", `${num(r.E).toFixed(3)} (${r.E_kind || "–"}) ≤ ${num(r.tolE_wide).toFixed(3)} widened · ${num(r.tolE_plain).toFixed(3)} plain`],
    ["profile H", `${num(r.H).toFixed(3)} (${r.H_kind || "–"}) ≤ ${Number.isFinite(num(r.tolH_wide)) ? num(r.tolH_wide).toFixed(3) : "–"} widened`],
    ["pass widened / plain", `${r.pass_wide} / ${r.pass_plain}`], ["bin", r.bin_wide || "–"], ["split fraction α", num(r.alpha).toFixed(3) + (bool(r.alpha_defined) ? "" : " (undefined)")],
    ["‖ΔℓI‖ ‖ΔℓR‖ ‖ΔℓL‖", `${num(r.nI).toFixed(3)} · ${num(r.nR).toFixed(3)} · ${num(r.nL).toFixed(3)}`],
    ["H_I H_R H_L", `${num(r.H_I).toFixed(3)} · ${num(r.H_R).toFixed(3)} · ${num(r.H_L).toFixed(3)}`],
    ["se_noise · se_tex", `${num(r.se_noise).toFixed(4)} · ${num(r.se_tex).toFixed(3)}`], ["flags", f.join(", ") || "none"],
    ["length", `${num(r.length_px).toFixed(1)} px`]];
  $("nums").innerHTML = `<table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>`;
  crop("cropA", +r.ax, +r.ay, +r.ar); crop("cropB", +r.bx, +r.by, +r.br);
  if (R.prof) {
    const PAI = vec(r.PA_I), PBI = vec(r.PB_I), PAR = vec(r.PA_R), PBR = vec(r.PB_R);
    const PAL = PAI.map((v, c) => v - PAR[c]), PBL = PBI.map((v, c) => v - PBR[c]);
    plot("plotI", "p_I (input)", R.prof.t, R.prof.I, PAI, PBI, num(r.H_I));
    plot("plotR", "p_R (reflectance)", R.prof.t, R.prof.R, PAR, PBR, num(r.H_R));
    plot("plotL", "p_L (illumination)", R.prof.t, R.prof.L, PAL, PBL, num(r.H_L));
  }
}

function renderList() {
  const v = visible(), all = R.rows.filter(r => r.method === R.method);
  const c = s => all.filter(r => status(r) === s).length;
  $("counts").textContent = `${R.stem}: ${all.length} pairs · pass ${c("pass")} · fail ${c("fail")} · excluded ${c("excluded")} · b ${c("b")} · shown ${v.length}`;
  $("list").innerHTML = v.map(r => `<div class="row${R.sel && r.id === R.sel.id ? " sel" : ""}" data-id="${r.id}"><span style="color:${color(r)}">${r.type}</span>` +
    `<span>${status(r)}${r.bin_wide ? " · " + r.bin_wide : ""}</span><span class="dim">${r.id.slice(-7)}</span><span>${num(r.E).toFixed(2)}</span></div>`).join("");
  $("list").querySelectorAll(".row").forEach(el => el.onclick = () => select(v.find(r => r.id === el.dataset.id)));
}

function renderAll() { renderSel(); renderList(); draw(); }
window.__rev = {R, ready: () => !!(R.img && R.rows), scr: (x, y) => { const b = cv().getBoundingClientRect(), q = toScr(x + .5, y + .5); return [b.left + q.x, b.top + q.y]; }};
init();
