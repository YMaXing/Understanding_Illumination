// Pair Annotator v1 (metrics/annotate). Keyboard-first; autosaves every change to the server (schema-validated there).
// Never loads predictions. Definitions: metrics/DESIGN.md; helpers: lib.js.
"use strict";
const L = window.AnnLib;
const $ = id => document.getElementById(id);
const TYPE_COLOR = {i: "#ffb020", m: "#4fc3ff", b: "#d36bff", s: "#6fd08c", d: "#ff6f91"};
const NAME_KEY = "metrics_annotator_id";     // the current tag's stable id (schema 1.2); names are resolved for display
const S = {
  man: null, C: null, images: [], idx: 0, stem: null, img: null, file: null, etag: null,
  view: {ev: 0, gamma: 2.2, log: false}, x4: false, ovl: {illum: false, material: false, geom: false, regions: false}, bad: false,
  ovlImg: {}, badCanvas: null, base: null, zoom: 1, panX: 0, panY: 0,
  type: "m", r: 4, placing: null, selected: null, activeEnd: "A", drag: null, pan: null, seed: null, mouse: null,
  undo: [], name: null, annotators: [], filters: {type: "", conf: "", loss: "", ann: ""},
  saveTimer: null, saving: null, pendingSave: false, conflict: false, modal: null, msg: "",
};

// ------------------------------------------------------------------ server
async function api(path, opt = {}) {
  const r = await fetch(path, opt);
  const ct = r.headers.get("Content-Type") || "";
  const body = ct.includes("json") ? await r.json() : await r.arrayBuffer();
  return {ok: r.ok, status: r.status, etag: r.headers.get("ETag"), body};
}

// S.name holds the current tag's id; S.tags the registry [{id, name, created, history}]; nameOf(id) the current name.
function setTags(tags) { S.tags = tags; S.annotators = tags.map(t => t.id); }
function nameOf(id) { const t = (S.tags || []).find(t => t.id === id); return t ? t.name : id; }

async function loadAnnotators() {
  setTags((await api("/api/annotators")).body.annotators);
  const stored = localStorage.getItem(NAME_KEY);
  S.name = S.annotators.includes(stored) ? stored : null;
  renderNameSel();
}

async function addAnnotator(name) {
  let r;
  try { r = await api("/api/annotators", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({name})}); }
  catch (err) { return {error: `could not reach the server (${err.message || err}); is metrics/annotate/run.sh still running?`}; }
  if (!r.ok) return {error: (r.body && r.body.errors ? r.body.errors.join("; ") : `server error ${r.status}`)};
  setTags(r.body.annotators);
  setName(r.body.tag.id);
  return {name: r.body.tag.name, id: r.body.tag.id};
}

// Rename the current tag: annotators.json only (pairs store the id, so no annotation file changes).
async function renameTag() {
  if (needName()) return;
  const id = S.name, old = nameOf(id);
  let u;
  try { u = (await api(`/api/annotators/${encodeURIComponent(id)}/usage`)).body; } catch (e) { u = {pairs: "?", images: "?"}; }
  let v = old, err = "";
  for (;;) {
    v = await askText(`Rename the name tag "${old}" (${u.pairs} pairs on ${u.images} images carry it; they will show the new name)`, v,
                      {okLabel: "Rename", error: err, emptyError: "A name tag cannot be empty."});
    if (!v) return renderNameSel();
    if (v === old) return renderNameSel();
    const ok = await ask(`Rename "${old}" to "${v}"? Only the name changes; the tag's id (${id}) and every pair stay as they are.`,
                         [{key: "1", label: "rename", value: "yes"}, {key: "2", label: "back", value: "no"}], "yes");
    if (ok !== "yes") { err = ""; continue; }
    const r = await api("/api/annotators/rename", {method: "POST", headers: {"Content-Type": "application/json"},
                                                   body: JSON.stringify({id, name: v, by: S.name})});
    if (r.ok) { setTags(r.body.annotators); renderNameSel(); refreshAnnFilter(); renderAll(); flash(`renamed to ${v}`); return; }
    err = (r.body.errors || [r.status]).join("; ");
  }
}

function setName(n) {
  S.name = n;
  if (n) localStorage.setItem(NAME_KEY, n); else localStorage.removeItem(NAME_KEY);
  renderNameSel();
}

function renderNameSel() {
  const sel = $("namesel");
  sel.innerHTML = `<option value="">— choose your name tag —</option>` +
    (S.tags || []).map(t => `<option value="${esc(t.id)}"${t.id === S.name ? " selected" : ""}>${esc(t.name)}</option>`).join("") +
    `<option value="__new">Add new…</option>` + (S.name ? `<option value="__rename">Rename this tag…</option>` : "");
  sel.classList.toggle("missing", !S.name);
}

function scheduleSave() {
  S.dirty = true;
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(save, 250);
  status("unsaved…");
}

async function save() {
  if (S.conflict) return;
  if (S.saving) { S.pendingSave = true; return S.saving; }
  S.saving = (async () => {
    S.dirty = false;
    const r = await api(`/api/ann/${S.stem}`, {method: "PUT", headers: {"Content-Type": "application/json", "If-Match": S.etag},
                                                body: JSON.stringify(S.file)});
    if (r.ok) { S.etag = r.body.etag; status(`saved ${new Date().toLocaleTimeString()}`, "ok"); }
    else if (r.status === 409) { S.conflict = true; banner("This image's file changed elsewhere (another tab or machine). " +
      "Your last change was NOT saved. Reload the page to continue (press F5)."); status("conflict", "warn"); }
    else { banner("The server rejected the save (nothing was written): " + (r.body.errors || [r.status]).join("; ")); status("save rejected", "warn"); }
  })();
  await S.saving;
  S.saving = null;
  if (S.pendingSave) { S.pendingSave = false; return save(); }
}

async function flush() {
  clearTimeout(S.saveTimer);
  if (S.dirty) await save();
  if (S.saving) await S.saving;
}

// ------------------------------------------------------------------ images
// The image list shows the images assigned to the current tag (toolbar "mine only"); all images otherwise.
function rebuildImageList() {
  const mine = $("mineonly").checked && S.name ? S.all.filter(m => m.assigned_to === S.name) : S.all;
  if ($("mineonly").checked && S.name && !mine.length) flash("no images are assigned to your tag: showing all images");
  S.images = mine.length ? mine : S.all;
  $("imgsel").innerHTML = S.images.map(m => `<option value="${m.stem}">${(m.split || "?").toUpperCase()} · ${m.stem}` +
    `${m.reflective_advisory ? " · reflective" : ""}${m.assigned_to && m.assigned_to !== S.name ? ` · ${esc(nameOf(m.assigned_to))}` : ""}</option>`).join("");
  if (S.stem) { const k = S.images.findIndex(m => m.stem === S.stem); if (k >= 0) { S.idx = k; $("imgsel").value = S.stem; } }
}

async function openImage(i, opts = {}) {
  await flush();
  const m = S.images[(i + S.images.length) % S.images.length];
  let relabel = false;
  if (!opts.confirmed && S.name && m.assigned_to && m.assigned_to !== S.name) {
    const v = await ask(`${m.stem} is assigned to ${nameOf(m.assigned_to)}, not to you. Open it anyway?`,
                        [{key: "1", label: "open it (I will annotate it)", value: "open"},
                         {key: "2", label: "open it for blind relabel only", value: "relabel"},
                         {key: "3", label: "stay on the current image", value: "stay"}], "stay");
    if (v == null || v === "stay") { if (S.stem) $("imgsel").value = S.stem; return; }
    relabel = v === "relabel";
  }
  S.idx = (i + S.images.length) % S.images.length;
  S.stem = m.stem;
  $("imgsel").value = m.stem;
  status("loading image…");
  const [meta, raw, lI, valid, ann] = await Promise.all([
    api(`/api/img/${m.stem}/meta.json`), api(`/api/img/${m.stem}/raw.u16.gz`), api(`/api/img/${m.stem}/lI.f32.gz`),
    api(`/api/img/${m.stem}/valid.u8.gz`), api(`/api/ann/${m.stem}`)]);
  const md = meta.body;
  S.img = {W: md.W, H: md.H, raw: new Uint16Array(raw.body), lI: new Float32Array(lI.body), valid: new Uint8Array(valid.body),
           noise: md.noise, tone_ref: md.tone_ref, lI_range: md.lI_range, meta: md};
  S.file = ann.body; S.etag = ann.etag; S.conflict = false;
  S.file.deleted_pairs = S.file.deleted_pairs || [];
  S.selected = null; S.placing = null; S.seed = null; S.undo = [];
  S.ovlImg = {};
  for (const o of md.overlays) { const im = new Image(); im.src = `/api/img/${m.stem}/${o.name}.png`; im.onload = draw; S.ovlImg[o.name] = im; }
  document.querySelectorAll("#ovl input[data-ovl]").forEach(el => {
    const k = el.dataset.ovl; el.disabled = !(k in S.ovlImg); if (el.disabled) { el.checked = false; S.ovl[k] = false; }
    el.parentElement.title = el.disabled ? (k === "regions" ? "n/a: no region source" : "n/a: map not available for this image") : "";
  });
  const sb = document.querySelector('#types button[data-type="s"]');
  // Greyed, not disabled: a disabled button swallows clicks and hides its tooltip, so the reason would never show.
  const off = !targetedAvailable();
  sb.classList.toggle("off", off); sb.setAttribute("aria-disabled", String(off));
  sb.title = off ? TARGETED_OFF : "targeted s (maintainer-only): needs an overlay on; random s comes from the seed button";
  if (S.type === "s" && off) setType("m");
  S.badCanvas = null;
  renderBase();
  fit();
  status(`loaded ${m.stem} · ${(S.file.split || "?").toUpperCase()} split`, "ok");
  refreshAnnFilter();
  renderAll();
  if (relabel) return enterRelabel();
  setTimeout(checkPending, 0);
}

function renderBase() {
  const {W, H} = S.img;
  const c = S.base || document.createElement("canvas");
  c.width = W; c.height = H;
  const ctx = c.getContext("2d"), id = ctx.createImageData(W, H);
  L.render(S.img, {ev: S.view.ev + (S.x4 ? 2 : 0), gamma: S.view.gamma, log: S.view.log}, id.data);
  ctx.putImageData(id, 0, 0);
  S.base = c;
}

function badCanvas() {
  if (S.badCanvas) return S.badCanvas;
  const {W, H, valid} = S.img, c = document.createElement("canvas");
  c.width = W; c.height = H;
  const ctx = c.getContext("2d"), id = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) if (!valid[i]) { id.data[4 * i] = 255; id.data[4 * i + 3] = 200; }
  ctx.putImageData(id, 0, 0);
  return (S.badCanvas = c);
}

// ------------------------------------------------------------------ view
function cv() { return $("cv"); }
function fit() {
  const el = cv(), W = el.clientWidth, H = el.clientHeight;
  S.zoom = Math.min(W / S.img.W, H / S.img.H);
  S.panX = -(W / S.zoom - S.img.W) / 2; S.panY = -(H / S.zoom - S.img.H) / 2;
  draw();
}
function toImg(sx, sy) { return {x: sx / S.zoom + S.panX, y: sy / S.zoom + S.panY}; }
function toScr(x, y) { return {x: (x - S.panX) * S.zoom, y: (y - S.panY) * S.zoom}; }
function center(x, y, z) {
  const el = cv();
  if (z) S.zoom = z;
  S.panX = x + 0.5 - el.clientWidth / S.zoom / 2; S.panY = y + 0.5 - el.clientHeight / S.zoom / 2;
  draw();
}
function anyOverlay() { return Object.entries(S.ovl).some(([k, v]) => v && k in S.ovlImg); }

function draw() {
  if (!S.img) return;
  const el = cv(), dpr = window.devicePixelRatio || 1;
  if (el.width !== el.clientWidth * dpr) { el.width = el.clientWidth * dpr; el.height = el.clientHeight * dpr; }
  const ctx = el.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, el.width, el.height);
  ctx.setTransform(S.zoom * dpr, 0, 0, S.zoom * dpr, -S.panX * S.zoom * dpr, -S.panY * S.zoom * dpr);
  ctx.imageSmoothingEnabled = S.zoom < 1;
  ctx.drawImage(S.base, 0, 0);
  for (const [k, on] of Object.entries(S.ovl)) if (on && S.ovlImg[k] && S.ovlImg[k].complete) ctx.drawImage(S.ovlImg[k], 0, 0);
  if (S.bad) ctx.drawImage(badCanvas(), 0, 0);
  const lw = 1.5 / S.zoom;
  const vis = visiblePairs();
  if (S.relabel) {
    const q = S.relabel.queue[S.relabel.i];
    if (q) drawPair(ctx, Object.assign({type: "x", confidence: "sure"}, q), true, lw);
    hud();
    return;
  }
  for (const p of S.file.pairs) {
    const sel = p.id === S.selected, faded = !vis.includes(p);
    ctx.globalAlpha = faded ? 0.25 : 1;
    drawPair(ctx, p, sel, lw);
  }
  ctx.globalAlpha = 1;
  if (S.placing) {
    const a = S.placing.A;
    drawPatch(ctx, a.x, a.y, S.r, TYPE_COLOR[S.placing.type], lw * 1.5, false);
    if (S.mouse) {
      const m = S.mouse;
      ctx.strokeStyle = TYPE_COLOR[S.placing.type]; ctx.lineWidth = lw; ctx.setLineDash([4 / S.zoom, 3 / S.zoom]);
      ctx.beginPath(); ctx.moveTo(a.x + 0.5, a.y + 0.5); ctx.lineTo(Math.floor(m.x) + 0.5, Math.floor(m.y) + 0.5); ctx.stroke();
      ctx.setLineDash([]);
      drawPatch(ctx, Math.floor(m.x), Math.floor(m.y), S.r, TYPE_COLOR[S.placing.type], lw, true);
    }
  }
  if (S.seed) {
    const s = S.seed;
    ctx.strokeStyle = "#ff3"; ctx.lineWidth = lw;
    ctx.beginPath(); ctx.arc(s.x0 + 0.5, s.y0 + 0.5, S.C.nudge_max_px, 0, 2 * Math.PI); ctx.stroke();
    drawPatch(ctx, s.x, s.y, S.r, "#ff3", lw * 1.5, false);
  }
  hud();
}

function drawPatch(ctx, x, y, r, color, lw, dashed) {
  ctx.strokeStyle = color; ctx.lineWidth = lw;
  if (dashed) ctx.setLineDash([3 / S.zoom, 2 / S.zoom]);
  if (S.C.patch.shape === "disc") { ctx.beginPath(); ctx.arc(x + 0.5, y + 0.5, r + 0.5, 0, 2 * Math.PI); ctx.stroke(); }
  else ctx.strokeRect(x - r, y - r, 2 * r + 1, 2 * r + 1);
  ctx.setLineDash([]);
}

function drawPair(ctx, p, sel, lw) {
  const col = sel ? "#fff" : TYPE_COLOR[p.type], dashed = p.confidence === "unsure";
  ctx.strokeStyle = col; ctx.lineWidth = sel ? lw * 2 : lw;
  if (dashed) ctx.setLineDash([4 / S.zoom, 3 / S.zoom]);
  ctx.beginPath(); ctx.moveTo(p.A.x + 0.5, p.A.y + 0.5); ctx.lineTo(p.B.x + 0.5, p.B.y + 0.5); ctx.stroke();
  ctx.setLineDash([]);
  drawPatch(ctx, p.A.x, p.A.y, p.A.r, col, sel ? lw * 2 : lw, dashed);
  drawPatch(ctx, p.B.x, p.B.y, p.B.r, col, sel ? lw * 2 : lw, dashed);
  if (S.zoom >= 2) {
    ctx.fillStyle = col; ctx.font = `${11 / S.zoom}px system-ui`;
    ctx.fillText("A", p.A.x + p.A.r + 1.5, p.A.y - p.A.r); ctx.fillText("B", p.B.x + p.B.r + 1.5, p.B.y - p.B.r);
    if (p.type === "i" && p.lit_end) { const e = p[p.lit_end]; ctx.fillText("lit", e.x - e.r, e.y + e.r + 12 / S.zoom); }
  }
}

function hud() {
  const m = S.mouse, img = S.img;
  let t = `${S.stem} · zoom ${S.zoom.toFixed(2)}×`;
  if (m && m.x >= 0 && m.y >= 0 && m.x < img.W && m.y < img.H) {
    const x = Math.floor(m.x), y = Math.floor(m.y), k = y * img.W + x;
    t += ` · x ${x} y ${y} · I ${img.raw[3 * k]}/${img.raw[3 * k + 1]}/${img.raw[3 * k + 2]}${img.valid[k] ? "" : " (invalid)"}`;
  }
  if (S.placing) t += S.placing.seed != null ? " · random s: click the partner B (20–80 px; Esc abandons with a reason)" : ` · placing ${S.placing.type}: click B (Esc cancels)`;
  if (S.seed) t += " · seed: Enter accept · arrows nudge ≤ 10 px · Delete reject · Esc abandon (with a reason)";
  if (S.msg) t += ` · ${S.msg}`;
  $("hud").textContent = t;
}

// ------------------------------------------------------------------ pairs
function pairById(id) { return S.file.pairs.find(p => p.id === id); }
function snapshot() {
  return JSON.stringify({pairs: S.file.pairs, deleted_pairs: S.file.deleted_pairs, exhausted: S.file.exhausted || null,
    seeds: S.file.s_seed_log.map(e => ({outcome: e.outcome, reason: e.reason, final_x: e.final_x, final_y: e.final_y,
                                         pair_id: e.pair_id, resolved_at: e.resolved_at, resolved_by: e.resolved_by}))});
}
function pushUndo() { S.undo.push(snapshot()); if (S.undo.length > 200) S.undo.shift(); }

function undo() {
  if (!S.undo.length) return flash("nothing to undo");
  const s = JSON.parse(S.undo.pop());
  S.file.pairs = s.pairs; S.file.deleted_pairs = s.deleted_pairs;
  if (s.exhausted) S.file.exhausted = s.exhausted; else delete S.file.exhausted;
  S.file.s_seed_log.forEach((e, k) => {
    const o = s.seeds[k];
    for (const f of ["outcome", "reason", "final_x", "final_y", "pair_id", "resolved_at", "resolved_by"]) {
      if (o) { if (o[f] === undefined) delete e[f]; else e[f] = o[f]; }
    }
    if (!o && e.pair_id && !pairById(e.pair_id)) { e.outcome = "undone"; }      // seed proposed after the snapshot
  });
  if (S.selected && !pairById(S.selected)) S.selected = null;
  scheduleSave(); renderAll(); flash("undone");
  if (!S.seed && !S.placing) setTimeout(checkPending, 0);
}

function needName() {
  if (S.name) return false;
  flash("Choose your name tag first (toolbar: name). Pairs are never anonymous.");
  banner("Choose your name tag (toolbar → name) before placing or editing pairs.", true);
  $("namesel").focus();
  return true;
}

async function createPair(type, A, B, extra = {}) {
  if (needName()) return null;
  if (type === "s" && extra.pool !== "random" && !anyOverlay()) {
    flash("A targeted s pair needs an overlay on (keys 1–3); random s pairs come from the seed button.");
    return null;
  }
  const r = S.r;
  const p = {id: L.newId(S.file), image: S.stem, type, A: {x: A.x, y: A.y, r: A.r ?? r}, B: {x: B.x, y: B.y, r: B.r ?? r},
             confidence: "sure", used_in_loss: false, overlay_seen: anyOverlay(), annotator: S.name,
             created: L.nowIso(), note: "", schema_version: S.file.schema_version};
  if (type === "i") {
    const inf = L.inferLitEnd(S.img, p.A, p.B, S.C.patch);
    const v = await ask(`i pair: lit end (inferred: ${inf}, the brighter patch)`, [
      {key: "a", label: "A is lit", value: "A"}, {key: "b", label: "B is lit", value: "B"}], inf);
    if (v == null) return null;
    p.lit_end = v;
  } else if (type === "s") {
    const v = await ask("s pair: lighting on both ends", [{key: "1", label: "lit", value: "lit"}, {key: "2", label: "shadow", value: "shadow"}], null);
    if (v == null) return null;
    p.lighting = v; p.pool = extra.pool || "targeted";
    if (p.pool === "targeted") {
      const g = await askText("targeted s: which known gap or failure-review pair is this near? (gap_ref)", "");
      if (!g) return null;
      p.gap_ref = g;
    }
  } else if (type === "d") {
    const opts = [{key: "1", label: "same", value: "same"}, {key: "2", label: "different", value: "different"}, {key: "3", label: "unsure", value: "unsure"}];
    const mat = await ask("d pair: material on the two ends", opts, null);
    if (mat == null) return null;
    const ill = await ask("d pair: illumination on the two ends", opts, null);
    if (ill == null) return null;
    p.material = mat; p.illumination = ill;
  }
  pushUndo();
  S.file.pairs.push(p);
  S.selected = p.id;
  scheduleSave(); renderAll();
  return p;
}

function editPair(op, fn) {
  const p = pairById(S.selected);
  if (!p) return flash("select a pair first");
  if (needName()) return;
  const before = JSON.parse(JSON.stringify(p)), snap = snapshot();
  fn(p);
  if (L.recordEdit(p, before, S.name, op)) { S.undo.push(snap); scheduleSave(); }
  renderAll();
}

function deleteSelected() {
  const p = pairById(S.selected);
  if (!p) return flash("select a pair first");
  if (needName()) return;
  pushUndo();
  S.file.pairs = S.file.pairs.filter(q => q.id !== p.id);
  S.file.deleted_pairs.push({id: p.id, deleted_at: L.nowIso(), deleted_by: S.name, pair: p});
  S.selected = null;
  scheduleSave(); renderAll(); flash(`deleted ${p.id} (u undoes)`);
}

// ------------------------------------------------------------------ random s pool
async function proposeSeed() {
  if (needName()) return;
  if (S.seed || (S.placing && S.placing.seed != null)) return flash("finish the current seed first");
  if (S.file.s_seed_log.some(e => e.outcome === "pending")) return checkPending();
  await flush();
  const r = await api(`/api/seed/${S.stem}`, {method: "POST", headers: {"Content-Type": "application/json", "If-Match": S.etag},
                                               body: JSON.stringify({annotator: S.name})});
  if (!r.ok) { if (r.status === 409) { S.conflict = true; banner("File changed elsewhere; reload."); } else flash((r.body.errors || []).join("; ")); return; }
  S.file.s_seed_log = r.body.file.s_seed_log; S.etag = r.body.etag;
  const e = r.body.seed;
  S.seed = {index: e.index, x0: e.x, y0: e.y, x: e.x, y: e.y};
  S.placing = null;
  center(e.x, e.y, Math.max(S.zoom, 3));
  renderAll();
}

function seedEntry(k) { return S.file.s_seed_log[k]; }

// Abandoning a seed is always logged with a reason. Returns false (and keeps the seed pending) if cancelled.
async function abandonSeed(k, prompt) {
  const reason = await askText(prompt || "Abandon this seed: why? (logged)", "", {emptyError: "Type a reason (or Esc to keep the seed)."});
  if (!reason) return false;
  pushUndo();
  resolveSeed(k, "abandoned", {reason});
  return true;
}

// A seed left pending (page closed, image switched) is offered again when the image is opened: resume or abandon.
async function checkPending() {
  if (S.relabel || S.seed || S.modal) return;
  const e = S.file.s_seed_log.find(x => x.outcome === "pending");
  if (!e) return;
  if (!S.name) { flash("this image has a pending random s seed: choose your name tag, then press r"); return; }
  const v = await ask(`A random s seed proposed ${e.proposed_at.slice(0, 16).replace("T", " ")} by ${nameOf(e.proposed_by)} is still pending`,
                      [{key: "1", label: "resume it (accept, nudge or reject as usual)", value: "resume"},
                       {key: "2", label: "abandon it (asks for a reason)", value: "abandon"}], "resume");
  if (v === "resume") {
    S.seed = {index: e.index, x0: e.x, y0: e.y, x: e.x, y: e.y};
    center(e.x, e.y, Math.max(S.zoom, 3)); renderAll();
  } else if (v === "abandon") {
    if (await abandonSeed(e.index, "Abandon the pending seed: why? (logged)")) { renderAll(); return checkPending(); }
  }
}
function resolveSeed(k, outcome, extra = {}) {
  const e = seedEntry(k);
  Object.assign(e, {outcome, resolved_at: L.nowIso(), resolved_by: S.name}, extra);
  scheduleSave();
}

function nudgeSeed(dx, dy) {
  const s = S.seed, nx = s.x + dx, ny = s.y + dy;
  if (Math.hypot(nx - s.x0, ny - s.y0) > S.C.nudge_max_px) return flash(`nudges stay within ${S.C.nudge_max_px} px of the seed`);
  if (nx < 0 || ny < 0 || nx >= S.img.W || ny >= S.img.H) return;
  s.x = nx; s.y = ny; draw();
}

async function rejectSeed() {
  const reason = await askText("Reject this seed: why? (logged)", "");
  if (!reason) return;
  pushUndo();
  resolveSeed(S.seed.index, "rejected", {reason});
  S.seed = null; renderAll();
}

function acceptSeed() {
  const s = S.seed;
  S.placing = {type: "s", A: {x: s.x, y: s.y}, seed: s.index, nudged: s.x !== s.x0 || s.y !== s.y0};
  S.seed = null; setType("s"); draw();
}

// ------------------------------------------------------------------ placing / editing with the mouse
function hit(sx, sy) {
  let best = null;
  for (const p of visiblePairs()) for (const e of ["A", "B"]) {
    const q = toScr(p[e].x + 0.5, p[e].y + 0.5), d = Math.hypot(q.x - sx, q.y - sy);
    const rad = Math.max(8, (p[e].r + 0.5) * S.zoom);
    if (d <= rad && (!best || d < best.d)) best = {p, end: e, d};
  }
  return best;
}

async function onImageClick(x, y) {
  x = Math.floor(x); y = Math.floor(y);
  if (S.modal || S.relabel) return;
  if (S.seed) return;
  if (!S.placing) {
    if (needName()) return;
    if (S.type === "s" && !anyOverlay()) return flash("s pairs: use the random s seed button (r), or turn an overlay on for a targeted s pair");
    S.placing = {type: S.type, A: {x, y}};
    draw(); renderWarn();
    return;
  }
  const pl = S.placing;
  S.placing = null;
  if (pl.seed != null) {
    const p = await createPair("s", pl.A, {x, y}, {pool: "random"});
    if (p) resolveSeed(pl.seed, pl.nudged ? "nudged" : "accepted", {final_x: pl.A.x, final_y: pl.A.y, pair_id: p.id});
    else if (!(await abandonSeed(pl.seed, "The random s pair was cancelled: why abandon the seed? (logged; Esc keeps it pending)"))) {
      const e = seedEntry(pl.seed); S.seed = {index: e.index, x0: e.x, y0: e.y, x: pl.A.x, y: pl.A.y};
    }
    renderAll();
    return;
  }
  await createPair(pl.type, pl.A, {x, y});
}

function startDrag(h) { S.drag = {id: h.p.id, end: h.end, before: JSON.parse(JSON.stringify(h.p)), snap: snapshot()}; S.selected = h.p.id; S.activeEnd = h.end; }
function moveDrag(x, y) { const p = pairById(S.drag.id); p[S.drag.end].x = Math.floor(x); p[S.drag.end].y = Math.floor(y); draw(); }
function endDrag() {
  const d = S.drag; S.drag = null;
  const p = pairById(d.id);
  if (!S.name) { Object.assign(p, d.before); needName(); renderAll(); return; }
  if (L.recordEdit(p, d.before, S.name, "drag")) { S.undo.push(d.snap); scheduleSave(); }
  renderAll();
}

function bindCanvas() {
  const el = cv();
  el.addEventListener("contextmenu", e => e.preventDefault());
  el.addEventListener("mousedown", e => {
    const r = el.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
    if (e.button === 2 || e.button === 1 || S.space) { S.pan = {sx, sy, px: S.panX, py: S.panY}; return; }
    if (S.seed) {   // drag the seed marker to nudge
      const q = toImg(sx, sy); S.seedDrag = true; nudgeTo(q); return;
    }
    const h = !S.placing && !S.relabel && hit(sx, sy);
    if (h) { startDrag(h); renderAll(); return; }
    const q = toImg(sx, sy);
    S.downAt = {sx, sy, x: q.x, y: q.y};
  });
  window.addEventListener("mousemove", e => {
    const r = el.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
    S.mouse = toImg(sx, sy);
    if (S.pan) { S.panX = S.pan.px - (sx - S.pan.sx) / S.zoom; S.panY = S.pan.py - (sy - S.pan.sy) / S.zoom; draw(); return; }
    if (S.drag) { moveDrag(S.mouse.x, S.mouse.y); return; }
    if (S.seedDrag) { nudgeTo(S.mouse); return; }
    if (S.placing) { draw(); renderWarn(); } else hud();
  });
  window.addEventListener("mouseup", e => {
    if (S.pan) { S.pan = null; return; }
    if (S.drag) { endDrag(); return; }
    if (S.seedDrag) { S.seedDrag = false; return; }
    if (S.downAt) { const d = S.downAt; S.downAt = null; onImageClick(d.x, d.y); }
  });
  el.addEventListener("wheel", e => {
    e.preventDefault();
    const r = el.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top, q = toImg(sx, sy);
    S.zoom = Math.min(Math.max(S.zoom * (e.deltaY < 0 ? 1.25 : 0.8), 0.1), 40);
    S.panX = q.x - sx / S.zoom; S.panY = q.y - sy / S.zoom; draw();
  }, {passive: false});
  window.addEventListener("resize", draw);
}

function nudgeTo(q) {
  const s = S.seed, dx = Math.floor(q.x) - s.x0, dy = Math.floor(q.y) - s.y0, d = Math.hypot(dx, dy), m = S.C.nudge_max_px;
  const k = d > m ? m / d : 1;
  s.x = s.x0 + Math.round(dx * k); s.y = s.y0 + Math.round(dy * k);
  if (Math.hypot(s.x - s.x0, s.y - s.y0) > m) { s.x = s.x0 + Math.trunc(dx * k); s.y = s.y0 + Math.trunc(dy * k); }
  draw();
}

// ------------------------------------------------------------------ modal prompts (keyboard-first)
function ask(title, options, def, opts = {}) {
  return new Promise(res => {
    $("modal").classList.toggle("corner", !!opts.corner);
    $("m_title").textContent = title;
    $("m_body").innerHTML = options.map(o => `<button class="opt${o.value === def ? " def" : ""}" data-v="${o.value}">` +
      `<kbd>${o.key}</kbd> ${esc(o.label)}</button>`).join("");
    $("m_hint").textContent = (def != null ? "Enter accepts the highlighted choice · " : "") + "Esc cancels";
    $("m_err").textContent = "";
    const box = $("modal");
    box.hidden = false;
    S.modal = "choice";
    const done = v => { box.hidden = true; box.onkeydown = null; S.modal = null; releaseFocus(); res(v); };
    $("m_body").querySelectorAll("button").forEach(b => b.onclick = () => done(b.dataset.v));
    // The dialog's own keys. A key it handles stops here: otherwise the same keydown would reach the global hotkeys
    // right after the dialog closes (Enter answering "resume" would also accept the resumed seed).
    box.onkeydown = e => {
      const stop = () => { e.preventDefault(); e.stopPropagation(); };
      if (e.key === "Escape") { stop(); return done(null); }
      if (e.key === "Enter" && def != null) { stop(); return done(def); }
      const o = options.find(o => o.key === e.key.toLowerCase());
      if (o) { stop(); done(o.value); }
    };
    box.focus();
  });
}

// Text dialog. Typing goes to the field untouched (only Enter / Escape are handled, on the field itself).
// opts.onSubmit(value) may return a promise of an error message: the dialog then stays open and shows it.
function askText(title, def, opts = {}) {
  return new Promise(res => {
    $("modal").classList.toggle("corner", !!opts.corner);
    $("m_title").textContent = title;
    $("m_body").innerHTML = `<input type="text" id="m_text" name="tool_text_${Date.now()}" autocomplete="off" data-1p-ignore ` +
      `data-lpignore="true" data-form-type="other" spellcheck="false" value="${esc(def || "")}">` +
      `<div class="btns"><button id="m_ok">${esc(opts.okLabel || "Save")}</button> <button id="m_cancel">Cancel</button></div>`;
    $("m_hint").textContent = "Enter saves · Esc cancels";
    $("m_err").textContent = opts.error || "";
    const box = $("modal"), inp = $("m_text");
    box.hidden = false;
    S.modal = "text";
    let busy = false;
    const done = v => { box.hidden = true; S.modal = null; releaseFocus(); res(v); };
    const submit = async () => {
      if (busy) return;
      const v = inp.value.trim();
      if (!v) { $("m_err").textContent = opts.emptyError || "Type something first (or Esc to cancel)."; return; }
      if (opts.onSubmit) {
        busy = true; $("m_err").textContent = "saving…";
        const err = await opts.onSubmit(v);
        busy = false;
        if (err) { $("m_err").textContent = err; inp.focus(); return; }
      }
      done(v);
    };
    inp.addEventListener("keydown", e => {
      if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); submit(); }
      else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(null); }
    });
    $("m_ok").onclick = submit;
    $("m_cancel").onclick = () => done(null);
    inp.focus();
    inp.select();
  });
}

// ------------------------------------------------------------------ keyboard
const KEYS = [
  ["i m b s d", "pair type for the next pair (s: targeted, needs an overlay; random s comes from r)"],
  ["click, click", "place A, then B (type-specific fields are asked)"], ["drag an end", "move it (logged edit)"],
  ["r", "random s seed: Enter accept · arrows nudge ≤ 10 px · Delete reject (with a reason) · Esc abandon (with a reason); a pending seed is offered again when the image is opened"],
  ["u or z", "undo"], ["q", "toggle confidence sure / unsure (selected pair)"], ["t", "toggle lit end (selected i pair)"],
  ["[ ]", "patch half-size r − / + (selected pair, both ends; else the size for new pairs)"],
  ["Alt+[ ]", "r − / + of the active end (the end last clicked)"], ["g", "auto-size the selected pair (largest r ≤ 4 on a plateau)"],
  ["w", "swap A and B"], ["N", "note for the selected pair"], ["Delete / Backspace", "delete the selected pair"],
  ["e", "×4 exposure"], ["l", "log view"], ["1 2 3", "overlays: illumination / material / geometric edges (4 regions: n/a)"],
  ["x", "invalid pixels"], ["n / p", "next / previous image"], ["f", "fit view"], ["wheel, right-drag, space+drag", "zoom, pan"],
  ["R (Shift+r)", "blind relabel mode for this image (Esc leaves it)"], ["Esc", "cancel placing / deselect"], ["?", "this help"]];

function openHelp() { $("help").hidden = false; $("help").focus(); }
const HOTKEYS = new Set([..."imbsduzqtwgNelxnpfrR1234[]?", "Delete", "Backspace", "Escape", "Enter", "PageUp", "PageDown",
                         "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);
// Targeted s is maintainer-only: it needs overlay maps, which only maintainers have. Without them the control is off.
const TARGETED_OFF = "This s button is for targeted s pairs (maintainer only; they need overlay maps, which this machine does not have). " +
                     "For random s pairs, click \u201crandom s seed\u201d or press r.";
function targetedAvailable() { return Object.keys(S.ovlImg || {}).length > 0; }
function setType(t) {
  if (t === "s" && !targetedAvailable()) { flash(TARGETED_OFF); banner(TARGETED_OFF, true); hintSeed(); return; }
  S.type = t; document.querySelectorAll("#types button").forEach(b => b.classList.toggle("on", b.dataset.type === t)); }

// A closed dialog must not keep keyboard focus (its hidden field would swallow the next hotkey until the browser's
// focus fix-up runs).
function releaseFocus() { const a = document.activeElement; if (a && a !== document.body) a.blur(); }
function dialogOpen() { return !$("modal").hidden || !$("help").hidden || S.modal != null; }
function isEditable(t) {
  return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
}

function onKey(e) {
  if (dialogOpen()) return;                  // dialogs handle their own keys; typed text is never intercepted
  if (isEditable(e.target)) { if (e.key === "Escape") e.target.blur(); return; }
  if (e.key === " ") { S.space = true; e.preventDefault(); return; }
  // A hotkey's own character must never reach a field the action focuses (N opens the note field: without this the
  // "N" itself would be typed into it).
  if (HOTKEYS.has(e.key) && !e.ctrlKey && !e.metaKey) e.preventDefault();
  if (S.seed) {
    const mv = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[e.key];
    if (mv) { e.preventDefault(); return nudgeSeed(...mv); }
    if (e.key === "Enter") return acceptSeed();
    if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); return rejectSeed(); }
    if (e.key === "Escape") { const k = S.seed.index; return abandonSeed(k).then(ok => { if (ok) S.seed = null; renderAll(); }); }
    return;
  }
  const k = e.key;
  if ("imbsd".includes(k) && k.length === 1 && !e.ctrlKey && !e.metaKey) return setType(k);
  if (k === "u" || k === "z") return undo();
  if (k === "Escape") {
    if (S.placing && S.placing.seed != null) {
      const pl = S.placing; S.placing = null;
      return abandonSeed(pl.seed).then(ok => {
        if (!ok) { const e = seedEntry(pl.seed); S.seed = {index: e.index, x0: e.x, y0: e.y, x: pl.A.x, y: pl.A.y}; }
        renderAll();
      });
    }
    if (S.placing) S.placing = null; else S.selected = null;
    return renderAll();
  }
  if (S.relabel) { if (k === "Escape") exitRelabel(); return; }
  if (k === "R") return enterRelabel();
  if (k === "r") return proposeSeed();
  if (k === "q") return editPair("confidence", p => { p.confidence = p.confidence === "sure" ? "unsure" : "sure"; });
  if (k === "t") return editPair("lit_end", p => { if (p.type === "i") p.lit_end = p.lit_end === "A" ? "B" : "A"; });
  if (k === "w") return editPair("swap", p => { [p.A, p.B] = [p.B, p.A]; if (p.lit_end) p.lit_end = p.lit_end === "A" ? "B" : "A"; });
  if (k === "g") return editPair("autosize", p => {
    const r = L.autosize(S.img, p.A, p.B, S.C.patch, S.C.autosize);
    if (r == null) flash("no r ≤ 4 keeps both patches valid and on a plateau"); else { p.A.r = r; p.B.r = r; flash(`auto-size r = ${r}`); }
  });
  if (k === "[" || k === "]") {
    const d = k === "[" ? -1 : 1, clamp = v => Math.min(Math.max(v + d, 1), 25);
    if (!S.selected) { S.r = clamp(S.r); return renderAll(); }
    return editPair("r", p => { if (e.altKey) p[S.activeEnd].r = clamp(p[S.activeEnd].r); else { p.A.r = clamp(p.A.r); p.B.r = clamp(p.B.r); } });
  }
  if (k === "N") return (async () => { const p = pairById(S.selected); if (!p) return flash("select a pair first");
    const v = await askText("note", p.note); if (v != null) editPair("note", q => { q.note = v; }); })();
  if (k === "Delete" || k === "Backspace") { e.preventDefault(); return deleteSelected(); }
  if (k === "e") { S.x4 = !S.x4; renderBase(); return draw(); }
  if (k === "l") { S.view.log = !S.view.log; $("logv").checked = S.view.log; renderBase(); return draw(); }
  if (k === "x") { S.bad = !S.bad; $("badv").checked = S.bad; return draw(); }
  if ("1234".includes(k) && k.length === 1) {
    const name = ["illum", "material", "geom", "regions"][+k - 1];
    if (!(name in S.ovlImg)) return flash(name === "regions" ? "n/a: no region source" : "n/a: no such map for this image");
    S.ovl[name] = !S.ovl[name];
    document.querySelector(`#ovl input[data-ovl=${name}]`).checked = S.ovl[name];
    return draw();
  }
  if (k === "n" || k === "PageDown") return openImage(S.idx + 1);
  if (k === "p" || k === "PageUp") return openImage(S.idx - 1);
  if (k === "f") return fit();
  if (k === "?") return openHelp();
}

// ------------------------------------------------------------------ side panel
function visiblePairs() {
  const f = S.filters;
  return S.file.pairs.filter(p => (!f.type || p.type === f.type) && (!f.conf || p.confidence === f.conf) &&
                                  (!f.loss || String(!!p.used_in_loss) === f.loss) && (!f.ann || p.annotator === f.ann));
}

function refreshAnnFilter() {
  const ids = [...new Set(S.file.pairs.map(p => p.annotator))].sort((a, b) => nameOf(a).localeCompare(nameOf(b)));
  const cur = S.filters.ann;
  $("f_ann").innerHTML = `<option value="">all annotators</option>` +
    ids.map(n => `<option value="${esc(n)}"${n === cur ? " selected" : ""}>${esc(nameOf(n))}</option>`).join("");
}

function renderList() {
  const vis = visiblePairs(), all = S.file.pairs;
  const cnt = t => all.filter(p => p.type === t).length;
  const byAnn = {};
  for (const p of all) (byAnn[p.annotator] = byAnn[p.annotator] || {})[p.type] = ((byAnn[p.annotator] || {})[p.type] || 0) + 1;
  $("progress").innerHTML = progressHtml(all);
  $("progress").querySelectorAll("button.exh").forEach(b => b.onclick = () => toggleExhausted(b.dataset.k));
  $("counts").innerHTML = `this image: ${["i", "m", "b", "s", "d"].map(t => `${t} ${cnt(t)}`).join(" · ")} · shown ${vis.length}<br>` +
    Object.entries(byAnn).map(([a, c]) => `${esc(nameOf(a))}: ${Object.entries(c).map(([t, n]) => `${t} ${n}`).join(" ")}`).join("<br>");
  $("list").innerHTML = vis.slice().reverse().map(p => `<div class="row${p.id === S.selected ? " sel" : ""}" data-id="${p.id}">` +
    `<span style="color:${TYPE_COLOR[p.type]}">${p.type}</span><span>${esc(nameOf(p.annotator))}${p.confidence === "unsure" ? " ?" : ""}</span>` +
    `<span class="dim">${p.id.slice(-7)}</span><span>${L.len(p).toFixed(0)}px</span></div>`).join("");
  $("list").querySelectorAll(".row").forEach(r => r.onclick = () => {
    const p = pairById(r.dataset.id); S.selected = p.id;
    center((p.A.x + p.B.x) / 2, (p.A.y + p.B.y) / 2, Math.max(S.zoom, 3)); renderAll();
  });
}

// Per-image targets (config annotate.targets): i / m / b = the dev means; random s and d ranges; targeted s: as found.
function progressHtml(all) {
  const T = S.man.targets || {}, n = f => all.filter(f).length, X = S.file.exhausted || {};
  const bar = (k, have, lo, hi, key) => {
    const ex = key && X[key], tgt = hi || lo, pct = ex ? 100 : Math.min(100, tgt ? 100 * have / tgt : 0), done = ex || have >= lo;
    const btn = key ? `<button class="exh${ex ? " on" : ""}" data-k="${key}" title="${ex ? esc(`exhausted (${nameOf(X[key].by)}): ${X[key].note}`) :
      "no more good pairs of this type on this image? mark it exhausted (counts as met)"}">${ex ? "exhausted" : "mark exhausted"}</button>` : "";
    return `<div class="prog"><span class="pl">${k}</span><span class="pb"><span style="width:${pct}%" class="${done ? "done" : ""}"></span></span>` +
      `<span class="pn">${have} / ${hi ? `${lo}–${hi}` : lo}</span>${btn}</div>`;
  };
  const dAll = n(p => p.type === "d"), dSM = n(p => p.type === "d" && p.material === "same" && p.illumination === "different");
  const frac = dAll ? dSM / dAll : 0, need = T.d_same_material_diff_illum_min_frac || 0.5;
  return `<div class="prog sub">image split: <b>${esc((S.file.split || "?").toUpperCase())}</b> (same targets on every split)</div>` +
    bar("i", n(p => p.type === "i"), T.i, null, "i") + bar("m", n(p => p.type === "m"), T.m, null, "m") + bar("b", n(p => p.type === "b"), T.b, null, "b") +
    bar("s random", n(p => p.type === "s" && p.pool === "random"), ...(T.s_random || [12, 15]), "s_random") +
    `<div class="prog"><span class="pl">s targeted</span><span class="pb dimbar"></span><span class="pn" title="targeted s is maintainer-only">${n(p => p.type === "s" && p.pool === "targeted") || ""} maint. only</span></div>` +
    bar("d", dAll, ...(T.d || [12, 15]), "d") +
    `<div class="prog sub ${dAll && frac < need ? "warnc" : ""}">d same material, different light: ${dSM} of ${dAll} ` +
    `(${dAll ? (100 * frac).toFixed(0) : 0}%; at least ${(100 * need).toFixed(0)}%)</div>`;
}

// Targets are guidance: a type can be marked exhausted on an image (with a note); it then counts as met.
async function toggleExhausted(k) {
  if (needName()) return;
  const X = S.file.exhausted || {};
  if (X[k]) {
    const v = await ask(`Remove the "exhausted" mark for ${k}? (set by ${nameOf(X[k].by)}: ${X[k].note})`,
                        [{key: "1", label: "remove the mark", value: "rm"}, {key: "2", label: "keep it", value: "keep"}], "keep");
    if (v !== "rm") return;
    pushUndo();
    delete X[k];
    if (!Object.keys(X).length) delete S.file.exhausted;
  } else {
    const note = await askText(`Mark ${k} exhausted on this image: why are there no more good ${k} pairs? (required)`, "",
                               {emptyError: "A short note is required (e.g. 'no other shadow edges on flat surfaces')."});
    if (!note) return;
    pushUndo();
    S.file.exhausted = Object.assign(X, {[k]: {by: S.name, at: L.nowIso(), note}});
  }
  scheduleSave(); renderAll();
}

// ------------------------------------------------------------------ blind relabel mode: geometry only (the server never
// sends the original type, labels or annotator); results go to a separate relabels file, never to the pair file.
async function enterRelabel() {
  if (needName()) return;
  await flush();
  const own = await ask("Blind relabel: which pairs?", [{key: "1", label: "pairs placed by other annotators", value: "0"},
                                                         {key: "2", label: "include my own pairs (test-retest)", value: "1"}], "0");
  if (own == null) return;
  const [q, f] = await Promise.all([api(`/api/relabel/queue/${S.stem}?annotator=${encodeURIComponent(S.name)}&own=${own}`),
                                    api(`/api/relabel/${S.stem}`)]);
  S.relabel = {queue: q.body.queue, total: q.body.total, done: q.body.done, i: 0, etag: f.etag, own: own === "1"};
  S.placing = null; S.selected = null;
  for (const k of Object.keys(S.ovl)) S.ovl[k] = false;
  document.querySelectorAll("#ovl input").forEach(el => { el.checked = false; });
  document.body.classList.add("relabel");
  renderAll();
  relabelLoop();
}

function exitRelabel() {
  S.relabel = null;
  document.body.classList.remove("relabel");
  renderAll();
}

async function relabelLoop() {
  while (S.relabel) {
    const R = S.relabel, q = R.queue[R.i];
    if (!q) { flash(`blind relabel: nothing left on this image (${R.done} of ${R.total} done)`); exitRelabel(); return; }
    center((q.A.x + q.B.x) / 2, (q.A.y + q.B.y) / 2, Math.max(S.zoom, 3));
    $("relabelbar").textContent = `BLIND RELABEL · ${S.stem} · pair ${R.done + 1} of ${R.total} · other pairs and the list are hidden`;
    const c = {corner: true};
    const type = await ask("What type is this pair?", [{key: "i", label: "i: illumination edge (same material)", value: "i"},
      {key: "m", label: "m: material edge (same light)", value: "m"}, {key: "b", label: "b: both at once", value: "b"},
      {key: "s", label: "s: no edge (same material, same light)", value: "s"}, {key: "d", label: "d: distant pair", value: "d"},
      {key: "k", label: "skip this pair for now", value: "skip"}], null, c);
    if (type == null) { exitRelabel(); return; }
    if (type === "skip") { R.queue.push(R.queue.splice(R.i, 1)[0]); continue; }
    const e = {pair_id: q.pair_id, relabeler: S.name, at: L.nowIso(), type};
    if (type === "i") { const v = await ask("Which end is lit?", [{key: "a", label: "A", value: "A"}, {key: "b", label: "B", value: "B"}], null, c); if (v == null) continue; e.lit_end = v; }
    if (type === "s") { const v = await ask("Lighting on both ends", [{key: "1", label: "lit", value: "lit"}, {key: "2", label: "shadow", value: "shadow"}], null, c); if (v == null) continue; e.lighting = v; }
    if (type === "d") {
      const o = [{key: "1", label: "same", value: "same"}, {key: "2", label: "different", value: "different"}, {key: "3", label: "unsure", value: "unsure"}];
      const mt = await ask("Material on the two ends", o, null, c); if (mt == null) continue;
      const il = await ask("Illumination on the two ends", o, null, c); if (il == null) continue;
      e.material = mt; e.illumination = il;
    }
    const cf = await ask("How sure are you?", [{key: "1", label: "sure", value: "sure"}, {key: "2", label: "unsure", value: "unsure"}], "sure", c);
    if (cf == null) continue;
    e.confidence = cf;
    const r = await api(`/api/relabel/${S.stem}`, {method: "POST", headers: {"Content-Type": "application/json", "If-Match": R.etag}, body: JSON.stringify(e)});
    if (!r.ok) { banner("Relabel not saved: " + (r.body.errors || [r.body.error || r.status]).join("; ")); exitRelabel(); return; }
    R.etag = r.body.etag; R.queue.splice(R.i, 1); R.done++;
    status(`relabel saved (${R.done} of ${R.total})`, "ok");
  }
}

function renderEditor() {
  const p = pairById(S.selected);
  if (!p) { $("edit").innerHTML = `<span class="dim">none (click a pair)</span><br>next pair: type <b>${S.type}</b>, r ${S.r}`; return; }
  const f = [["id", p.id], ["type", p.type], ["annotator", nameOf(p.annotator)], ["confidence", p.confidence], ["used in a loss", String(!!p.used_in_loss)],
             ["A", `${p.A.x}, ${p.A.y} r ${p.A.r}`], ["B", `${p.B.x}, ${p.B.y} r ${p.B.r}`], ["length", `${L.len(p).toFixed(1)} px`],
             ["overlay_seen", String(p.overlay_seen)], ["created", p.created]];
  if (p.type === "i") f.push(["lit_end", p.lit_end]);
  if (p.type === "s") f.push(["lighting", p.lighting], ["pool", p.pool], ...(p.gap_ref ? [["gap_ref", p.gap_ref]] : []));
  if (p.type === "d") f.push(["material", p.material], ["illumination", p.illumination]);
  if (p.last_edited_by) f.push(["last edited", `${nameOf(p.last_edited_by)} ${p.edited}`]);
  if (p.note) f.push(["note", p.note]);
  $("edit").innerHTML = f.map(([k, v]) => `<span class="k">${k}</span> ${esc(String(v))}`).join("<br>");
}

function renderWarn() {
  let p = pairById(S.selected);
  if (S.placing && S.mouse) p = {type: S.placing.type, pool: S.placing.seed != null ? "random" : undefined,
                                 A: {...S.placing.A, r: S.r}, B: {x: Math.floor(S.mouse.x), y: Math.floor(S.mouse.y), r: S.r}};
  if (!p) { $("warn").innerHTML = `<span class="dim">–</span>`; return; }
  const w = L.warnings(S.img, p, S.C);
  $("warn").innerHTML = (w.warnings.length ? w.warnings.map(x => `<div class="w">⚠ ${esc(x)}</div>`).join("") : `<span class="dim">no warnings</span>`) +
    `<div class="dim">||ΔℓI|| ${w.dI.toFixed(3)} · se_noise ${w.seNoise.toFixed(4)} · se_tex ${w.seTex.toFixed(3)}</div>`;
}

function renderAll() { $("rval").textContent = S.r; renderEditor(); renderWarn(); renderList(); draw(); }

// ------------------------------------------------------------------ misc
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c])); }
function status(t, cls) { const el = $("save"); el.textContent = t; el.className = cls || "dim"; }
function hintSeed() { const b = $("seed"); b.classList.add("hint"); clearTimeout(S.hintT); S.hintT = setTimeout(() => b.classList.remove("hint"), 3000); }
function flash(t) { S.msg = t; hud(); clearTimeout(S.msgT); S.msgT = setTimeout(() => { S.msg = ""; hud(); }, 4000); }
function banner(t, info) { const b = $("banner"); b.textContent = t; b.className = info ? "info" : ""; b.hidden = false;
  clearTimeout(S.banT); if (info) S.banT = setTimeout(() => { b.hidden = true; }, 5000); }

function bindToolbar() {
  $("prev").onclick = () => openImage(S.idx - 1);
  $("next").onclick = () => openImage(S.idx + 1);
  $("imgsel").onchange = e => openImage(S.images.findIndex(m => m.stem === e.target.value));
  $("namesel").onchange = async e => {
    const v = e.target.value;
    e.target.blur();
    if (v === "__new") {
      const n = await askText("Add your name tag (as it should appear in the data)", "", {
        okLabel: "Add name tag", emptyError: "Type your name first.",
        onSubmit: async name => (await addAnnotator(name)).error || null});
      renderNameSel();
      if (n) flash(`name tag: ${nameOf(S.name)}`);
    } else if (v === "__rename") { await renameTag(); }
    else { setName(v || null); rebuildImageList(); }
  };
  document.querySelectorAll("#types button").forEach(b => b.onclick = () => setType(b.dataset.type));
  $("seed").onclick = proposeSeed;
  $("relabelb").onclick = enterRelabel;
  $("ev").oninput = e => { S.view.ev = +e.target.value; renderBase(); draw(); };
  $("gamma").onchange = e => { S.view.gamma = Math.min(Math.max(+e.target.value || 2.2, 1), 3); renderBase(); draw(); };
  $("logv").onchange = e => { S.view.log = e.target.checked; renderBase(); draw(); };
  $("badv").onchange = e => { S.bad = e.target.checked; draw(); };
  document.querySelectorAll("#ovl input[data-ovl]").forEach(el => el.onchange = () => { S.ovl[el.dataset.ovl] = el.checked; draw(); });
  ["type", "conf", "loss", "ann"].forEach(k => $(`f_${k}`).onchange = e => { S.filters[k] = e.target.value; renderAll(); });
  $("helpb").onclick = openHelp;
  $("help").onkeydown = e => { if (e.key === "Escape" || e.key === "?") { e.preventDefault(); e.stopPropagation(); $("help").hidden = true; releaseFocus(); } };
  $("help").onclick = e => { if (e.target.id === "help") { $("help").hidden = true; releaseFocus(); } };
  $("helptxt").innerHTML = `<table>${KEYS.map(([k, v]) => `<tr><td><kbd>${esc(k)}</kbd></td><td>${esc(v)}</td></tr>`).join("")}</table>`;
  window.addEventListener("keydown", onKey);
  window.addEventListener("keyup", e => { if (e.key === " ") S.space = false; });
  window.addEventListener("beforeunload", e => { if (S.dirty || S.saving) { e.preventDefault(); e.returnValue = ""; } });
}

async function init() {
  S.man = (await api("/api/manifest")).body;
  S.C = S.man.constants;
  S.r = S.C.patch.default_r;
  S.all = S.man.images;
  S.images = S.all;
  bindToolbar(); bindCanvas(); setType("m");
  await loadAnnotators();
  $("mineonly").onchange = () => { rebuildImageList(); renderAll(); };
  rebuildImageList();
  const want = new URLSearchParams(location.search).get("image");
  if (want && !S.images.some(m => m.stem === want)) { $("mineonly").checked = false; rebuildImageList(); }
  await openImage(Math.max(0, S.images.findIndex(m => m.stem === want)));
}

// ------------------------------------------------------------------ test hook (read-only helpers for annotate/selftest.py,
// which drives the page with real input events over the Chrome DevTools Protocol)
window.__ann = {
  S, L,
  scr: (x, y) => { const r = cv().getBoundingClientRect(), q = toScr(x + 0.5, y + 0.5); return [r.left + q.x, r.top + q.y]; },
  ready: () => !!(S.img && S.file && !S.saving && !S.dirty),
  nameOf: id => nameOf(id),
  progress: () => $("progress").textContent,
  pending: () => S.file.s_seed_log.filter(e => e.outcome === "pending").length,
  modal: () => !$("modal").hidden, err: () => $("m_err").textContent, title: () => $("m_title").textContent,
};

init();
