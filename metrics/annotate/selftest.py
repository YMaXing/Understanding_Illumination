#!/usr/bin/env python3
"""Headless-browser self-test of the annotation tool with REAL input events (Chrome DevTools Protocol).

    metrics/run.sh annotate/selftest.py [--image taiwei_cui_080]

Copies one image's annotation file and annotators.json into metrics/runs/annotate_selftest/annotations/ (the real
files are never touched) and serves that copy on a free port. annotate/cdp_driver.ps1 (Windows PowerShell; WSL's NAT
networking cannot reach Chrome's debugging port) launches its own headless Chrome with a fresh profile and drives the
page: key presses via Input.dispatchKeyEvent (the name tag is typed one key at a time), text via Input.insertText
(gap_ref, rejection reason, note), clicks and drags via Input.dispatchMouseEvent. The <select> for "Add new..." is
set by script (only text fields need real typing). Checks are made on the page state and on the server's files;
then on disk: the copy validates, backups exist, the real annotation file is unchanged. Writes selftest.txt,
results.json and screenshot.png next to the copy.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = HERE.parent
sys.path.insert(0, str(M))
sys.path.insert(0, str(HERE))
from core import schema as SC  # noqa: E402
import server as SV  # noqa: E402

PS_STD = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"     # standard location (fallback)


def powershell():
    from core import config as K
    return (K.load_config().get("tools") or {}).get("powershell") or PS_STD
NAME = "Selftest Bot"
NAME_ID = "selftest_bot"              # the stable id the server derives from the name (schema 1.2)
NO_OVL = "adke_neel_026"                # a draw54 image: no overlay maps on any machine yet


def winpath(p):
    return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True).stdout.strip()


def run_driver(steps, out, port_dbg=9333, width=1600, height=1000):
    sp, rp = out / "steps.json", out / "results.json"
    sp.write_text(json.dumps(steps, indent=1), encoding="utf-8")
    r = subprocess.run([powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", winpath(HERE / "cdp_driver.ps1"),
                        "-Steps", winpath(sp), "-Out", winpath(rp), "-Profile", winpath(out / "chrome"),
                        "-Port", str(port_dbg), "-Width", str(width), "-Height", str(height)] + chrome_arg(),
                       capture_output=True, text=True, timeout=600)
    if not rp.exists():
        raise SystemExit(f"driver wrote no results: {r.stdout[-2000:]} {r.stderr[-2000:]}")
    return json.loads(rp.read_text(encoding="utf-8-sig"))


def chrome_arg():
    from core import config as K
    c = (K.load_config().get("tools") or {}).get("chrome")
    if not c:
        return []
    w = subprocess.run(["wslpath", "-w", c], capture_output=True, text=True).stdout.strip() if c.startswith("/mnt/") else c
    return ["-Chrome", w]


class Steps(list):
    def ready(self):
        self.append({"wait": "window.__ann && __ann.ready()", "timeout_ms": 20000})

    def key(self, k, mods=0):
        self.append({"key": k, "mods": mods})

    def typekeys(self, s):                         # one real key event per character
        for ch in s:
            self.key(ch)

    def click(self, x, y):
        self.append({"click_js": f"__ann.scr({x}, {y})"})

    def click_js(self, js):
        self.append({"click_js": js})

    def ev(self, label, js):
        self.append({"eval": js, "label": label})

    def wait(self, js, ms=15000):
        self.append({"wait": js, "timeout_ms": ms})


def build(url, shot):
    s = Steps()
    _url = url
    s.append({"nav": url})
    s.ready()
    n0 = "__ann.S.file.pairs.length"
    s.ev("n0", n0)
    # 1. no name tag: placing is blocked
    s.key("m"); s.click(100, 100); s.click(130, 100); s.append({"sleep": 300})
    s.ev("blocked_no_name", f"({n0}) === __ann.S.file.pairs.length && !__ann.S.placing && !document.getElementById('banner').hidden")
    # 2. add a name tag by typing: first a too-long one (server error shown in the dialog), then the real one
    sel_new = "(()=>{const s=document.getElementById('namesel'); s.value='__new'; s.dispatchEvent(new Event('change')); return true})()"
    s.ev("open_add", sel_new); s.wait("__ann.modal()")
    s.ev("field_attrs", "(()=>{const i=document.getElementById('m_text'); return [i.getAttribute('autocomplete'), i.hasAttribute('data-1p-ignore'), i.getAttribute('data-lpignore'), document.activeElement===i]})()")
    s.typekeys("x" * 70); s.key("Enter")
    s.wait("__ann.err().includes('too long')")
    s.ev("error_in_dialog", "[__ann.modal(), __ann.err()]")
    s.key("Escape"); s.wait("!__ann.modal()")
    s.ev("open_add2", sel_new); s.wait("__ann.modal()")
    s.typekeys("  Selftest   Bot ")                # spaces, Shift+S / Shift+B, all as key events
    s.ev("typed_value", "document.getElementById('m_text').value")
    s.ev("type_unchanged_while_typing", "__ann.S.type")
    s.key("Enter"); s.wait("!__ann.modal()")
    s.ev("name_after_add", "[__ann.S.name, document.getElementById('namesel').value, __ann.S.annotators.includes('selftest_bot'), __ann.nameOf(__ann.S.name)]")
    # 3. place an m pair under the new tag, undo it
    s.key("m"); s.click(100, 100); s.click(130, 100); s.ready()
    s.ev("m_pair", "__ann.S.file.pairs[__ann.S.file.pairs.length-1]")
    s.key("u"); s.ready()
    s.ev("after_undo_n", "__ann.S.file.pairs.length")
    # 4. i pair: lit-end dialog, Enter accepts the inferred end
    s.key("i"); s.click(200, 150); s.click(200, 180); s.wait("__ann.modal()")
    s.ev("i_title", "__ann.title()"); s.key("Enter"); s.ready()
    s.ev("i_pair", "(()=>{const p=__ann.S.file.pairs.find(q=>q.id===__ann.S.selected); return [p, __ann.L.inferLitEnd(__ann.S.img, p.A, p.B, __ann.S.C.patch)]})()")
    s.key("q"); s.ready()
    s.ev("i_after_q", "__ann.S.file.pairs.find(q=>q.id===__ann.S.selected)")
    # 5. overlay on -> overlay_seen; targeted s needs an overlay; gap_ref typed
    s.key("3"); s.key("m"); s.click(300, 120); s.click(330, 120); s.ready(); s.key("3")
    s.ev("ovl_pair", "__ann.S.file.pairs.find(q=>q.id===__ann.S.selected)")
    s.key("s"); s.click(50, 50); s.append({"sleep": 200})
    s.ev("targeted_blocked", "!__ann.S.placing")
    s.key("3"); s.click(50, 50); s.click(80, 50); s.wait("__ann.modal()"); s.key("2")
    s.wait("__ann.modal() && document.getElementById('m_text')")
    s.append({"type": "gap near the post base"}); s.key("Enter"); s.ready(); s.key("3")
    s.ev("targeted_pair", "__ann.S.file.pairs.find(q=>q.id===__ann.S.selected)")
    # 6. hotkeys never fire inside a text dialog (note field): letters land in the field
    s.key("N"); s.wait("__ann.modal()")
    s.typekeys("imbsdur")                           # no clean-up first: the opening "N" must not appear in the field
    s.ev("note_field_hotkeys", "[document.getElementById('m_text').value, __ann.S.type, !!__ann.S.seed]")
    s.key("Enter"); s.ready()
    s.ev("note_saved", "__ann.S.file.pairs.find(q=>q.id===__ann.S.selected).note")
    # 7. random s: propose, nudge (clamped), accept, partner, lighting
    s.ev("nlog", "__ann.S.file.s_seed_log.length")
    s.key("r"); s.wait("!!__ann.S.seed")
    s.ev("seed0", "[__ann.S.seed.x0, __ann.S.seed.y0]")
    for _ in range(3):
        s.key("ArrowRight")
    for _ in range(20):
        s.key("ArrowDown")
    s.ev("seed_nudged", "[__ann.S.seed.x - __ann.S.seed.x0, __ann.S.seed.y - __ann.S.seed.y0]")
    s.key("Enter")
    s.click_js("__ann.scr(Math.min(__ann.S.placing.A.x + 30, __ann.S.img.W - 6), __ann.S.placing.A.y)")
    s.wait("__ann.modal()"); s.key("1"); s.ready()
    s.ev("seed_entry", "__ann.S.file.s_seed_log[__ann.S.file.s_seed_log.length-1]")
    s.ev("seed_pair", "(()=>{const e=__ann.S.file.s_seed_log[__ann.S.file.s_seed_log.length-1]; return __ann.S.file.pairs.find(q=>q.id===e.pair_id)})()")
    # 8. reject a seed with a typed reason
    s.key("r"); s.wait("!!__ann.S.seed"); s.key("Delete"); s.wait("__ann.modal()")
    s.append({"type": "on a specular highlight"}); s.key("Enter"); s.ready()
    s.ev("rejected_entry", "__ann.S.file.s_seed_log[__ann.S.file.s_seed_log.length-1]")
    # 8b. a pending seed survives a reload and is offered again: abandon (with a reason), then resume + abandon
    s.key("r"); s.wait("!!__ann.S.seed")
    s.append({"nav": "__RELOAD__"})
    s.wait("__ann.modal() && __ann.title().includes('assigned to')")
    s.ev("assigned_warning", "[__ann.title(), document.querySelectorAll('#m_body button').length]")
    s.key("1"); s.ready()
    s.wait("__ann.modal() && __ann.title().includes('still pending')")
    s.key("2"); s.wait("__ann.modal() && document.getElementById('m_text')")
    s.append({"type": "left the image to check the guide"}); s.key("Enter"); s.ready()
    s.ev("pending_abandoned", "__ann.S.file.s_seed_log[__ann.S.file.s_seed_log.length-1]")
    s.key("r"); s.wait("!!__ann.S.seed")
    s.append({"nav": "__RELOAD__"})
    s.wait("__ann.modal() && __ann.title().includes('assigned to')"); s.key("1"); s.ready()
    s.wait("__ann.modal() && __ann.title().includes('still pending')")
    s.key("Enter"); s.wait("!!__ann.S.seed"); s.append({"sleep": 300})
    s.ev("pending_resumed", "[__ann.S.seed && __ann.S.seed.index, __ann.S.file.s_seed_log.length - 1, !!__ann.S.placing]")
    s.key("Escape"); s.wait("__ann.modal() && document.getElementById('m_text')")
    s.append({"type": "resumed, then not usable"}); s.key("Enter"); s.ready()
    s.ev("pending_left", "[__ann.pending(), __ann.S.file.s_seed_log[__ann.S.file.s_seed_log.length-1]]")
    # 8c. exhausted marks (note required), counted as met; removable
    s.click_js("(()=>{const b=document.querySelector('#progress button.exh[data-k=\"b\"]'); const r=b.getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]})()")
    s.wait("__ann.modal() && document.getElementById('m_text')"); s.key("Enter")
    s.ev("exh_empty_refused", "[__ann.modal(), __ann.err()]")
    s.append({"type": "only two edges where both change"}); s.key("Enter"); s.ready()
    s.ev("exh_set", "[__ann.S.file.exhausted, __ann.progress()]")
    s.click_js("(()=>{const b=document.querySelector('#progress button.exh[data-k=\"b\"]'); const r=b.getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]})()")
    s.wait("__ann.modal()"); s.key("1"); s.ready()
    s.ev("exh_removed", "__ann.S.file.exhausted || null")
    s.ev("split_label", "[document.getElementById('imgsel').selectedOptions[0].textContent, __ann.progress().includes('image split')]")
    # 9. d pair, delete, drag, g / w / ], help
    s.key("f")                                      # the seed step zoomed in: fit the view before fixed-position clicks
    s.key("d"); s.click(40, 40); s.click_js("__ann.scr(Math.min(240, __ann.S.img.W-10), Math.min(160, __ann.S.img.H-10))")
    s.wait("__ann.modal()"); s.key("1"); s.wait("__ann.title().includes('illumination')"); s.key("2"); s.ready()
    s.ev("d_pair", "__ann.S.file.pairs.find(q=>q.id===__ann.S.selected)")
    s.ev("d_id", "__ann.S.selected")
    s.key("Delete"); s.ready()
    s.ev("tomb", "__ann.S.file.deleted_pairs[__ann.S.file.deleted_pairs.length-1]")
    s.ev("drag_target", "(()=>{const p=__ann.S.file.pairs.find(q=>q.annotator==='selftest_bot' && q.type==='m'); return p.id})()")
    s.key("f")
    s.append({"drag_js": "(()=>{const p=__ann.S.file.pairs.find(q=>q.annotator==='selftest_bot' && q.type==='m'); "
                         "const a=__ann.scr(p.B.x, p.B.y), b=__ann.scr(p.B.x+4, p.B.y); return [a[0],a[1],b[0],b[1]]})()"})
    s.ready()
    s.ev("dragged", "__ann.S.file.pairs.find(q=>q.annotator==='selftest_bot' && q.type==='m')")
    s.key("g"); s.key("w"); s.key("]"); s.ready()
    s.ev("edits_ok", "document.getElementById('banner').hidden || !document.getElementById('banner').textContent.includes('rejected')")
    s.key("?"); s.wait("!document.getElementById('help').hidden"); s.key("Escape")
    s.ev("help_closed", "document.getElementById('help').hidden")
    # 10. server refusals, and no request outside the tool's API
    s.ev("stale", "fetch('/api/ann/'+__ann.S.stem,{method:'PUT',headers:{'Content-Type':'application/json','If-Match':'0'.repeat(64)},body:JSON.stringify(__ann.S.file)}).then(r=>r.status)")
    s.ev("invalid", "(()=>{const b=JSON.parse(JSON.stringify(__ann.S.file)); delete b.pairs[0].annotator; return fetch('/api/ann/'+__ann.S.stem,{method:'PUT',headers:{'Content-Type':'application/json','If-Match':__ann.S.etag},body:JSON.stringify(b)}).then(r=>r.status)})()")
    # 10b. rename the current tag by typing; annotation file untouched (same ETag); a colliding name is refused
    s.ev("etag_before", "fetch('/api/ann/'+__ann.S.stem).then(r=>r.headers.get('ETag'))")
    sel_ren = "(()=>{const s=document.getElementById('namesel'); s.value='__rename'; s.dispatchEvent(new Event('change')); return true})()"
    s.ev("open_rename", sel_ren); s.wait("__ann.modal() && document.getElementById('m_text')")
    s.ev("rename_title", "__ann.title()")
    s.typekeys("yueren")                             # the dialog pre-selects the old name, so typing replaces it
    s.append({"sleep": 150}); s.key("Enter")            # collides (case-insensitively) with the tag "Yueren"
    s.wait("__ann.modal() && __ann.title().includes('Rename \"')"); s.key("Enter")
    s.wait("__ann.modal() && __ann.err().includes('already used')")
    s.ev("rename_collision", "[__ann.err(), __ann.nameOf(__ann.S.name)]")
    s.key("Escape"); s.wait("!__ann.modal()")
    s.ev("open_rename2", sel_ren); s.wait("__ann.modal() && document.getElementById('m_text')")
    s.typekeys("Selftest Robot"); s.append({"sleep": 150}); s.key("Enter")
    s.wait("__ann.modal() && __ann.title().includes('Rename \"')"); s.key("Enter")
    s.wait("!__ann.modal() && __ann.nameOf(__ann.S.name) === 'Selftest Robot'")
    s.ev("after_rename", "[__ann.S.name, __ann.nameOf(__ann.S.name), document.getElementById('namesel').selectedOptions[0].textContent, "
                         "[...document.querySelectorAll('#list .row')].some(r => r.textContent.includes('Selftest Robot')), "
                         "[...document.querySelectorAll('#list .row')].some(r => r.textContent.includes('Selftest Bot'))]")
    s.ev("etag_after", "fetch('/api/ann/'+__ann.S.stem).then(r=>r.headers.get('ETag'))")
    # 10c. another annotator's image opened for blind relabel needs no further confirmation
    s.append({"nav": "__RELOAD__"})
    s.wait("__ann.modal() && __ann.title().includes('assigned to')"); s.key("2")
    s.wait("__ann.modal() && (__ann.title().includes('which pairs') || __ann.title().includes('What type'))")
    s.ev("relabel_exempt", "[__ann.title(), !!__ann.S.relabel || __ann.title().includes('which pairs')]")
    s.key("Escape"); s.wait("!__ann.modal()")
    # 11. progress against targets
    s.ev("progress", "__ann.progress()")
    # 12. blind relabel mode: Selftest Bot relabels Yueren's pairs (geometry only), answers by key
    s.ev("pairs_before_relabel", "JSON.stringify(__ann.S.file.pairs)")
    s.key("Escape")
    s.key("R"); s.wait("__ann.modal() && __ann.title().includes('which pairs')"); s.key("Enter")
    s.wait("__ann.modal() && __ann.title().includes('What type')")
    s.ev("relabel_ui", "[!!__ann.S.relabel, document.body.classList.contains('relabel'), getComputedStyle(document.querySelector('#side > section')).visibility, "
                       "Object.keys(__ann.S.relabel.queue[0]).sort().join(','), Object.values(__ann.S.ovl).some(v=>v), __ann.S.relabel.total]")
    s.ev("relabel_first", "__ann.S.relabel.queue[0].pair_id")
    s.key("m"); s.wait("__ann.title().includes('How sure')"); s.key("Enter")
    s.wait("__ann.S.relabel && __ann.S.relabel.done === 1 && __ann.title().includes('What type')")
    s.key("i"); s.wait("__ann.title().includes('lit')"); s.key("a"); s.wait("__ann.title().includes('How sure')"); s.key("2")
    s.wait("__ann.S.relabel && __ann.S.relabel.done === 2 && __ann.title().includes('What type')")
    s.key("d"); s.wait("__ann.title().includes('Material')"); s.key("1"); s.wait("__ann.title().includes('Illumination')"); s.key("2")
    s.wait("__ann.title().includes('How sure')"); s.key("Enter")
    s.wait("__ann.S.relabel && __ann.S.relabel.done === 3 && __ann.title().includes('What type')")
    s.key("Escape"); s.wait("!__ann.S.relabel")
    s.ev("relabel_exit", "[document.body.classList.contains('relabel'), __ann.modal()]")
    s.ev("relabel_file", "fetch('/api/relabel/'+__ann.S.stem).then(r=>r.json())")
    s.ev("pairs_after_relabel", "fetch('/api/ann/'+__ann.S.stem).then(r=>r.json()).then(f=>JSON.stringify(f.pairs))")
    s.ev("urls", "performance.getEntriesByType('resource').map(r=>new URL(r.name).pathname)")
    s.append({"shot": shot})
    # 12b. an image without overlay maps (an annotator's machine): the greyed s button, clicked with the mouse, says why
    #      and points to the seed button; the seed button then works with no overlays
    s.append({"nav": url.split("?")[0] + "?image=" + NO_OVL})
    s.wait("__ann.modal() && __ann.title().includes('assigned to')"); s.key("1"); s.ready()
    btn = lambda sel: f"(()=>{{const r=document.querySelector('{sel}').getBoundingClientRect(); return [r.x+r.width/2, r.y+r.height/2]}})()"
    s.ev("s_off_state", "(()=>{const b=document.querySelector('#types button[data-type=\"s\"]'); "
                        "return [Object.keys(__ann.S.ovlImg).length, b.classList.contains('off'), b.disabled, __ann.S.type]})()")
    s.click_js(btn('#types button[data-type="s"]')); s.append({"sleep": 300})
    s.ev("s_off_click", "[__ann.S.type, !document.getElementById('banner').hidden, document.getElementById('banner').textContent, "
                        "document.getElementById('seed').classList.contains('hint')]")
    s.append({"shot": shot.replace("screenshot.png", "screenshot_s_off.png")})
    s.click_js(btn("#seed")); s.wait("!!__ann.S.seed")
    s.ev("seed_no_ovl", "[!!__ann.S.seed, __ann.S.file.s_seed_log[__ann.S.file.s_seed_log.length-1].outcome]")
    s.key("Escape"); s.wait("__ann.modal() && document.getElementById('m_text')")
    s.typekeys("selftest"); s.key("Enter"); s.wait("!__ann.modal()")
    s.ready()                                      # saved: leaving with unsaved changes opens the browser's own prompt
    s.ev("seed_no_ovl_after", "__ann.S.file.s_seed_log[__ann.S.file.s_seed_log.length-1]")
    # 13. review page (separate page; server started with --review)
    s.append({"nav": url.replace("/?image=", "/review.html?run=baselines_v1&image=")})
    s.wait("window.__rev && __rev.ready()", 30000)
    s.ev("review_counts", "document.getElementById('counts').textContent")
    s.click_js("(()=>{const r=__rev.R.rows.find(r=>r.method===__rev.R.method && r.type==='i'); return __rev.scr(+r.ax,+r.ay)})()")
    s.wait("!!__rev.R.prof && document.querySelectorAll('.plots svg').length === 3")
    s.ev("review_sel", "[__rev.R.sel.type, document.getElementById('nums').textContent.includes('endpoint E'), document.querySelectorAll('.plots svg polyline').length, "
                       "document.getElementById('plotR').textContent.includes('H =')]")
    s.ev("review_filter", "(()=>{const s=document.getElementById('f_status'); s.value='fail'; s.dispatchEvent(new Event('change')); "
                          "return [...document.querySelectorAll('#list .row')].length})()")
    s.ev("review_fail_rows", "__rev.R.rows.filter(r=>r.method===__rev.R.method && r.kind!=='b' && !r.exclusions && r.pass_wide!=='True').length")
    s.append({"shot": shot.replace("screenshot.png", "review.png")})
    for st in s:
        if st.get("nav") == "__RELOAD__":
            st["nav"] = _url
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="taiwei_cui_080")
    a = ap.parse_args()
    out = M / "runs" / "annotate_selftest"
    ann = out / "annotations"
    if out.exists():
        shutil.rmtree(out)
    ann.mkdir(parents=True)
    real = M / "annotations" / f"{a.image}.json"
    real_sha = hashlib.sha256(real.read_bytes()).hexdigest()
    shutil.copy2(real, ann / real.name)
    shutil.copy2(M / "annotations" / "annotators.json", ann / "annotators.json")
    shutil.copy2(M / "annotations" / f"{NO_OVL}.json", ann / f"{NO_OVL}.json")
    srv = SV.make_server(HERE / "cache" / "manifest.json", "127.0.0.1", 0, annotations=str(ann), review=str(M / "runs"))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        res = run_driver(build(f"http://localhost:{port}/?image={a.image}", winpath(out / "screenshot.png")), out)
    finally:
        srv.shutdown()
    V = {r["label"]: r.get("value") for r in res["results"] if r.get("label")}
    disk_file = json.loads((ann / real.name).read_text())
    disk_ann = json.loads((ann / "annotators.json").read_text())["annotators"]
    chk = []

    def ok(name, cond, info=None):
        chk.append(f"{'ok  ' if cond else 'FAIL'} {name}" + ("" if cond or info is None else f" :: {json.dumps(info)[:400]}"))

    ok("driver ran every step", res["status"] == "done", res["status"])
    g = V.get
    ok("no name tag blocks pair creation", g("blocked_no_name") is True)
    ok("dialog field: autocomplete off, 1Password + LastPass ignore, focused", g("field_attrs") == ["off", True, "true", True], g("field_attrs"))
    ok("server error (too long) shown in the dialog, dialog stays open", bool(g("error_in_dialog")) and g("error_in_dialog")[0] and "too long" in g("error_in_dialog")[1], g("error_in_dialog"))
    ok("typed characters appear in the name field (real key events)", g("typed_value") == "  Selftest   Bot ", g("typed_value"))
    ok("typing letters in the dialog does not change the pair type", g("type_unchanged_while_typing") == "m", g("type_unchanged_while_typing"))
    ok("new name tag saved, in the dropdown and current", g("name_after_add") == [NAME_ID, NAME_ID, True, NAME] and NAME_ID in [x["id"] for x in disk_ann], [g("name_after_add"), disk_ann])
    mp = g("m_pair") or {}
    ok("pair placed under the new tag (annotator, used_in_loss false, overlay_seen false)", mp.get("annotator") == NAME_ID and mp.get("used_in_loss") is False and "split" not in mp and mp.get("overlay_seen") is False and mp.get("type") == "m", mp)
    ok("u undoes it", g("after_undo_n") == g("n0"), [g("after_undo_n"), g("n0")])
    ip = (g("i_pair") or [{}, None])
    ok("i pair: lit-end dialog; Enter accepts the inferred brighter end", "lit end" in (g("i_title") or "") and ip[0].get("lit_end") == ip[1], ip)
    q = g("i_after_q") or {}
    ok("q toggles confidence; annotator kept; last_edited_by + history", q.get("confidence") == "unsure" and q.get("annotator") == NAME_ID and q.get("last_edited_by") == NAME_ID and q.get("history", [{}])[-1].get("op") == "confidence", q)
    ok("overlay on at creation -> overlay_seen true", (g("ovl_pair") or {}).get("overlay_seen") is True, g("ovl_pair"))
    ok("targeted s without an overlay is blocked", g("targeted_blocked") is True)
    tp = g("targeted_pair") or {}
    ok("targeted s: lighting key + gap_ref typed (insertText)", tp.get("pool") == "targeted" and tp.get("lighting") == "shadow" and tp.get("gap_ref") == "gap near the post base", tp)
    nf = g("note_field_hotkeys") or []
    ok("hotkey letters typed into the note field go to the field, not to the tool", nf[:1] == ["imbsdur"] and nf[1] == "s" and nf[2] is False, nf)
    ok("note saved", g("note_saved") == "imbsdur", g("note_saved"))
    sn = g("seed_nudged") or [0, 0]
    ok("seed nudge clamps at 10 px", sn[0] == 3 and (sn[0] ** 2 + sn[1] ** 2) ** 0.5 <= 10, sn)
    se, spair = g("seed_entry") or {}, g("seed_pair") or {}
    ok("random s: pool random, lighting lit; log nudged with final point and pair id", se.get("outcome") == "nudged" and se.get("final_x") == g("seed0")[0] + 3 and spair.get("pool") == "random" and spair.get("lighting") == "lit" and se.get("resolved_by") == NAME_ID, [se, spair])
    rj = g("rejected_entry") or {}
    ok("rejected seed: typed reason logged", rj.get("outcome") == "rejected" and rj.get("reason") == "on a specular highlight", rj)
    dp = g("d_pair") or {}
    pa = g("pending_abandoned") or {}
    ok("pending seed after reload: offered again; abandon asks a reason (logged)", pa.get("outcome") == "abandoned" and pa.get("reason") == "left the image to check the guide", pa)
    pr, pl = g("pending_resumed") or [None, None], g("pending_left") or [None, {}]
    ok("pending seed: resume restores it (the Enter is not also an accept); Esc abandons with a reason; none left pending",
       pr[0] == pr[1] and pr[2] is False and pl[0] == 0 and pl[1].get("reason") == "resumed, then not usable", [pr, pl])
    ee, es = g("exh_empty_refused") or [None, ""], g("exh_set") or [{}, ""]
    ok("exhausted mark: empty note refused, note saved, progress shows it as met", ee[0] is True and "required" in ee[1].lower() or "note" in (ee[1] or "").lower())
    ok("exhausted mark saved with name and note; removable", (es[0] or {}).get("b", {}).get("note") == "only two edges where both change" and es[0]["b"]["by"] == NAME_ID
       and "exhausted" in es[1] and g("exh_removed") is None, [es, g("exh_removed")])
    ok("image picker and progress show the image's data split", (g("split_label") or [""])[0].split(" · ")[0] in ("TRAIN", "VAL", "TEST") and g("split_label")[1] is True, g("split_label"))
    so, sc = g("s_off_state") or [None] * 4, g("s_off_click") or [None] * 4
    ok("no overlays: the s button is greyed but clickable (not disabled)", so[0] == 0 and so[1] is True and so[2] is False, so)
    ok("no overlays: clicking s keeps the type, shows why in the banner and highlights the seed button",
       sc[0] == so[3] and sc[0] != "s" and sc[1] is True and "random s seed" in (sc[2] or "") and sc[3] is True, sc)
    sn, sa = g("seed_no_ovl") or [False, None], g("seed_no_ovl_after") or {}
    ok("no overlays: the random s seed button proposes a seed (then abandoned with a reason)",
       sn == [True, "pending"] and sa.get("outcome") == "abandoned" and sa.get("reason") == "selftest", [sn, sa])
    aw = g("assigned_warning") or ["", 0]
    ok("opening another annotator's image warns and asks (open / blind relabel / stay)", "assigned to Yueren" in aw[0] and aw[1] == 3, aw)
    rc = g("rename_collision") or ["", ""]
    ok("rename: a name colliding case-insensitively with another tag is refused in the dialog", "already used" in rc[0] and rc[1] == NAME, rc)
    ar = g("after_rename") or []
    ok("rename by typing: id kept, new name in the selector and the pair list, old name gone",
       ar[:3] == [NAME_ID, "Selftest Robot", "Selftest Robot"] and ar[3] is True and ar[4] is False, ar)
    ok("rename shows how many pairs carry the tag", "pairs on" in (g("rename_title") or ""), g("rename_title"))
    ok("rename changes no annotation file (same ETag = same bytes)", g("etag_before") and g("etag_before") == g("etag_after"), [g("etag_before"), g("etag_after")])
    ok("blind relabel of another annotator's image needs no further confirmation", (g("relabel_exempt") or [None, False])[1] is True, g("relabel_exempt"))
    ok("d pair: material + illumination keys", dp.get("material") == "same" and dp.get("illumination") == "different", dp)
    ok("delete leaves a tombstone", (g("tomb") or {}).get("id") == g("d_id") and (g("tomb") or {}).get("deleted_by") == NAME_ID)
    dg = g("dragged") or {}
    ok("drag (real mouse events) logs a 'drag' edit", any(h.get("op") == "drag" for h in dg.get("history", [])), dg.get("history"))
    ok("g / w / ] edits saved", g("edits_ok") is True)
    ok("? opens help, Escape closes it", g("help_closed") is True)
    ok("stale write -> 409, invalid write -> 400", g("stale") == 409 and g("invalid") == 400, [g("stale"), g("invalid")])
    urls = g("urls") or []
    ok("no request outside the tool's API (never loads predictions or runs)",
       bool(urls) and all(u.startswith(("/api/manifest", "/api/images", "/api/annotators", "/api/img/", "/api/ann/", "/api/seed/", "/api/relabel/")) or u in ("/lib.js", "/app.js", "/style.css") for u in urls), urls)
    ok("progress panel shows the targets (i 33 = dev median, random s 12–15, d same-material share)", "/ 33" in (g("progress") or "") and "12–15" in (g("progress") or "")
       and "same material, different light" in (g("progress") or ""), g("progress"))
    ru = g("relabel_ui") or []
    ok("blind relabel: list hidden, geometry only (no type / labels / annotator), overlays off", ru[:5] == [True, True, "hidden", "A,B,pair_id", False] and ru[5] > 0, ru)
    rf = (g("relabel_file") or {}).get("relabels", [])
    ok("blind relabel: three keyed answers saved to the relabels file (m sure; i lit A unsure; d same / different)",
       [(e["type"], e.get("lit_end"), e.get("material"), e.get("illumination"), e["confidence"], e["relabeler"]) for e in rf] ==
       [("m", None, None, None, "sure", NAME_ID), ("i", "A", None, None, "unsure", NAME_ID), ("d", None, "same", "different", "sure", NAME_ID)]
       and rf[0]["pair_id"] == g("relabel_first") and all(e["own_pair"] is False for e in rf), rf)
    ok("blind relabel never changes the pair file; Esc leaves the mode", g("pairs_before_relabel") == g("pairs_after_relabel") and g("relabel_exit") == [False, False], g("relabel_exit"))
    ok("review page: run loaded, pair selected by click, three profile plots with H marked", "pairs" in (g("review_counts") or "") and
       (g("review_sel") or [None])[0] == "i" and g("review_sel")[1] is True and g("review_sel")[2] == 9 and g("review_sel")[3] is True, [g("review_counts"), g("review_sel")])
    ok("review page: the fail filter lists exactly the failing eligible pairs", g("review_filter") == g("review_fail_rows"), [g("review_filter"), g("review_fail_rows")])
    ok("the self-test's file validates on disk", SC.validate_file(disk_file) == [])
    ok("timestamped backups written", any((ann / ".backups" / a.image).glob("*.json")))
    ok("the real annotation file is unchanged", hashlib.sha256(real.read_bytes()).hexdigest() == real_sha)
    passed = all(c.startswith("ok") for c in chk)
    txt = "\n".join(chk) + f"\nANNOTATE SELFTEST {'PASS' if passed else 'FAIL'} ({sum(c.startswith('ok') for c in chk)}/{len(chk)})"
    (out / "selftest.txt").write_text(txt + "\n")
    print(txt)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
