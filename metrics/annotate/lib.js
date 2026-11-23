// Pure helpers for the annotation tool (no DOM). Same definitions as metrics/core (patches.py, pair_metrics.py):
// patches are squares (2r+1)^2 (or discs), centred on integer pixels, x = column, y = row; out-of-bounds pixels count
// as invalid; medians are of the cached ℓI (computed by the training-loader hook in prepare.py, never here).
"use strict";
(function (root) {
  const L = {};

  L.median = function (a) {
    if (!a.length) return NaN;
    const s = Float64Array.from(a).sort();
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };

  L.offsets = function (r, shape) {
    const out = [];
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++)
      if (shape !== "disc" || dx * dx + dy * dy <= r * r) out.push([dx, dy]);
    return out;
  };

  // img = {W, H, lI: Float32Array(H*W*3), valid: Uint8Array(H*W), raw: Uint16Array(H*W*3), noise: [[g, r2] x3]}
  L.patchStats = function (img, x, y, r, P) {
    const off = L.offsets(r, P.shape), ch = [[], [], []], Is = [[], [], []];
    let n = 0;
    for (const [dx, dy] of off) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= img.W || yy >= img.H) continue;
      const k = yy * img.W + xx;
      if (!img.valid[k]) continue;
      n++;
      for (let c = 0; c < 3; c++) { ch[c].push(img.lI[3 * k + c]); Is[c].push(Math.max(img.raw[3 * k + c], 1)); }
    }
    const med = ch.map(L.median);
    const se = ch.map((v, c) => n ? P.se_median_factor * P.mad_to_sigma * L.median(v.map(t => Math.abs(t - med[c]))) / Math.sqrt(n) : NaN);
    const seNoise = Is.map((v, c) => {
      if (!n) return NaN;
      const [g, r2] = img.noise[c];
      let s = 0;
      for (const I of v) s += (g * I + r2) / (I * I);
      return P.se_median_factor * Math.sqrt(s / n / n);
    });
    return {med, se, seNoise, n, nTotal: off.length, frac: n / off.length, ok: n / off.length >= P.min_valid_frac};
  };

  L.combined = (a, b, key) => Math.sqrt((a[key][0] ** 2 + b[key][0] ** 2 + a[key][1] ** 2 + b[key][1] ** 2 + a[key][2] ** 2 + b[key][2] ** 2) / 3);
  L.cnorm = v => Math.sqrt((v[0] ** 2 + v[1] ** 2 + v[2] ** 2) / 3);
  L.brightness = v => (v[0] + v[1] + v[2]) / 3;
  L.len = p => Math.hypot(p.B.x - p.A.x, p.B.y - p.A.y);

  // §3: lit end = the end with the brighter b(P(ℓI))
  L.inferLitEnd = function (img, A, B, P) {
    const a = L.patchStats(img, A.x, A.y, A.r, P), b = L.patchStats(img, B.x, B.y, B.r, P);
    if (!a.n || !b.n) return "A";
    return L.brightness(a.med) >= L.brightness(b.med) ? "A" : "B";
  };

  // Old tool's auto-sizer (C3): largest r <= max_r with both patches fully valid and on a plateau (each patch moved
  // plateau_shift px further from the other changes its median by <= plateau_tol of the step |ΔℓI|).
  L.autosize = function (img, A, B, P, AS) {
    const dx = A.x - B.x, dy = A.y - B.y, len = Math.hypot(dx, dy) || 1, sh = AS.plateau_shift_px;
    const cl = (v, n) => Math.min(Math.max(v, 0), n - 1);
    const d3 = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    for (let r = AS.max_r; r >= 1; r--) {
      const a = L.patchStats(img, A.x, A.y, r, P), b = L.patchStats(img, B.x, B.y, r, P);
      if (a.n !== a.nTotal || b.n !== b.nTotal) continue;
      const a2 = L.patchStats(img, cl(Math.round(A.x + sh * dx / len), img.W), cl(Math.round(A.y + sh * dy / len), img.H), r, P);
      const b2 = L.patchStats(img, cl(Math.round(B.x - sh * dx / len), img.W), cl(Math.round(B.y - sh * dy / len), img.H), r, P);
      const step = d3(a.med, b.med);
      if (!(step > 0) || !a2.n || !b2.n) continue;
      if (d3(a2.med, a.med) / step <= AS.plateau_tol && d3(b2.med, b.med) / step <= AS.plateau_tol) return r;
    }
    return null;
  };

  // Live warnings (never block): length limits by type, patch validity, low SNR (sensor-noise model), out of bounds.
  L.warnings = function (img, p, C) {
    const w = [], P = C.patch, len = L.len(p);
    if (C.near_types.includes(p.type) && len > C.near_max_px) w.push(`near pair longer than ${C.near_max_px} px (${len.toFixed(0)} px)`);
    if (C.distant_types.includes(p.type) && len < C.distant_min_px) w.push(`distant pair shorter than ${C.distant_min_px} px (${len.toFixed(0)} px)`);
    if (p.type === "s" && p.pool === "random" && (len < C.s_partner_px[0] || len > C.s_partner_px[1]))
      w.push(`random s partner should be ${C.s_partner_px[0]}–${C.s_partner_px[1]} px away (${len.toFixed(0)} px)`);
    for (const [nm, e] of [["A", p.A], ["B", p.B]]) {
      if (e.x - e.r < 0 || e.y - e.r < 0 || e.x + e.r >= img.W || e.y + e.r >= img.H) w.push(`${nm} patch out of bounds`);
    }
    const a = L.patchStats(img, p.A.x, p.A.y, p.A.r, P), b = L.patchStats(img, p.B.x, p.B.y, p.B.r, P);
    for (const [nm, s] of [["A", a], ["B", b]]) if (!s.ok) w.push(`${nm} patch ${(100 * s.frac).toFixed(0)}% valid (< ${100 * P.min_valid_frac}%)`);
    const q = L.combined(a, b, "seNoise");
    if (!(q <= C.tau / C.snr_divisor)) w.push(`low SNR (noise SE ${q.toFixed(3)} > tau/${C.snr_divisor})`);
    return {warnings: w, a, b, seTex: L.combined(a, b, "se"), seNoise: q, dI: L.cnorm([0, 1, 2].map(c => b.med[c] - a.med[c]))};
  };

  L.newId = function (file) {
    let mx = 0;
    for (const p of file.pairs) mx = Math.max(mx, +p.id.slice(2));
    for (const t of file.deleted_pairs || []) mx = Math.max(mx, +t.id.slice(2));
    return "t=" + Math.max(Date.now(), mx + 1);
  };

  L.nowIso = () => new Date().toISOString();

  // Record an edit on an existing pair: keeps annotator / created, sets last_edited_by / edited, appends history.
  L.recordEdit = function (p, before, who, op) {
    const ch = {}, bf = {};
    for (const k of Object.keys(Object.assign({}, before, p))) {
      if (["history", "edited", "last_edited_by", "legacy"].includes(k)) continue;
      if (JSON.stringify(before[k]) !== JSON.stringify(p[k])) { bf[k] = before[k]; ch[k] = p[k]; }
    }
    if (!Object.keys(ch).length) return false;
    p.last_edited_by = who;
    p.edited = L.nowIso();
    (p.history = p.history || []).push({op, at: p.edited, by: who, before: bf, after: ch});
    return true;
  };

  // Tone map: linear raw -> RGBA. view = {ev, gamma, log}; ref = 99.5th percentile (common.tone); log view maps the
  // ℓI range [lo, hi] linearly to [0, 1].
  L.render = function (img, view, out) {
    const n = img.W * img.H, g = 1 / view.gamma, k = Math.pow(2, view.ev) / img.tone_ref;
    const [lo, hi] = img.lI_range, span = Math.max(hi - lo, 1e-6);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        let v = view.log ? (img.lI[3 * i + c] - lo) / span + view.ev / 8 : Math.pow(Math.min(Math.max(img.raw[3 * i + c] * k, 0), 1), g);
        out[4 * i + c] = Math.round(Math.min(Math.max(v, 0), 1) * 255);
      }
      out[4 * i + 3] = 255;
    }
    return out;
  };

  root.AnnLib = L;
  if (typeof module !== "undefined") module.exports = L;
})(typeof window !== "undefined" ? window : globalThis);
