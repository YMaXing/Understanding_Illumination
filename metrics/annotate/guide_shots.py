#!/usr/bin/env python3
"""Screenshots for docs/ANNOTATION_GUIDE.md, taken by the headless-Chrome harness (annotate/cdp_driver.ps1).

    metrics/run.sh annotate/guide_shots.py

Runs the real tool on a throwaway copy of taiwei_cui_080's annotations (metrics/runs/guide_shots/; the real files are
never touched). Dialogs are opened with real key presses and clicks; labels are drawn on top of the page (numbered
boxes) just before each capture. The pair-type examples are demonstration pairs drawn into the page state for the
picture only (never saved). Writes metrics/docs/img/*.png|jpg.
"""

import json
import shutil
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = HERE.parent
sys.path.insert(0, str(M))
sys.path.insert(0, str(HERE))
import server as SV  # noqa: E402
import selftest as ST  # noqa: E402

STEM = "taiwei_cui_080"
LAB = r"""
window.__lab = {
  clear() { document.querySelectorAll('.__lab').forEach(e => e.remove()); },
  box(r, n, text, color, pos) {
    color = color || '#ffb020';
    const d = document.createElement('div'); d.className = '__lab';
    Object.assign(d.style, {position: 'fixed', left: (r.left - 3) + 'px', top: (r.top - 3) + 'px', width: (r.width + 6) + 'px',
      height: (r.height + 6) + 'px', border: `3px solid ${color}`, borderRadius: '5px', zIndex: 60, pointerEvents: 'none'});
    const t = document.createElement('div'); t.textContent = n + (text ? '  ' + text : '');
    Object.assign(t.style, {position: 'absolute', left: '-3px', background: color, color: '#111', font: 'bold 13px system-ui',
      padding: '1px 7px', borderRadius: '3px', whiteSpace: 'nowrap'});
    pos = pos === true ? 'below' : (pos || 'above');
    if (pos === 'below') t.style.bottom = '-24px'; else if (pos === 'inside') { t.style.top = '4px'; t.style.left = '4px'; }
    else if (pos === 'inside-right') { t.style.top = '4px'; t.style.left = 'auto'; t.style.right = '4px'; } else t.style.top = '-24px';
    d.appendChild(t); document.body.appendChild(d);
  },
  el(sel, n, text, color, pos) { const e = document.querySelector(sel); if (e) this.box(e.getBoundingClientRect(), n, text, color, pos); },
  img(x0, y0, x1, y1, n, text, color, below) { const a = __ann.scr(x0, y0), b = __ann.scr(x1, y1);
    this.box({left: Math.min(a[0], b[0]), top: Math.min(a[1], b[1]), width: Math.abs(b[0] - a[0]), height: Math.abs(b[1] - a[1])}, n, text, color, below); },
  hideBanner() { const b = document.getElementById('banner'); if (b) b.hidden = true; },
  note(x, y, text, color) { const d = document.createElement('div'); d.className = '__lab'; d.textContent = text;
    Object.assign(d.style, {position: 'fixed', left: x + 'px', top: y + 'px', background: color || '#ffb020', color: '#111',
      font: 'bold 14px system-ui', padding: '3px 8px', borderRadius: '4px', zIndex: 60, maxWidth: '520px'}); document.body.appendChild(d); },
};
true
"""
MODAL_CLIP = "(()=>{const r=document.querySelector('#modal .box').getBoundingClientRect(); return [r.left-12, r.top-12, r.width+24, r.height+24]})()"
HELP_CLIP = "(()=>{const r=document.querySelector('#help .box').getBoundingClientRect(); return [r.left-12, r.top-12, r.width+24, r.height+24]})()"
STAGE_CLIP = "(()=>{const r=document.getElementById('stage').getBoundingClientRect(); return [r.left, r.top, r.width, r.height]})()"

# Demonstration pairs (image x, y), chosen by eye on taiwei_cui_080 (docs: why each is good or bad)
EX = {
    "i": dict(good=dict(A=(700, 588), B=(700, 646), r=4), bad=dict(A=(655, 585), B=(655, 610), r=4),
              view=(680, 615, 4.0), good_txt="GOOD i: lit grass to shadowed grass, both patches well clear of the edge",
              bad_txt="BAD i: the B patch sits in the soft shadow edge (penumbra)"),
    "m": dict(good=dict(A=(454, 775), B=(495, 764), r=4), bad=dict(A=(454, 775), B=(476, 771), r=4),
              view=(472, 772, 5.0), good_txt="GOOD m: stone slab to grass, same sunlight", bad_txt="BAD m: the B patch touches the slab's edge"),
    "b": dict(good=dict(A=(455, 622), B=(505, 612), r=3), bad=dict(A=(600, 455), B=(600, 615), r=3),
              view=(530, 560, 2.6), good_txt="GOOD b: lit metal foot to shadowed grass: one edge where both change",
              bad_txt="BAD b: pebble edge and shadow edge are separate (make an m and an i pair instead)"),
    "s_random": dict(good=dict(A=(640, 530), B=(680, 545), r=4), bad=dict(A=(740, 600), B=(740, 665), r=4),
                     view=(690, 590, 3.2), good_txt="GOOD random s: lit grass, no edge between the ends",
                     bad_txt="BAD random s: crosses the shadow edge (that is an i pair)"),
    "s_targeted": dict(good=dict(A=(100, 307), B=(140, 316), r=3), bad=dict(A=(120, 312), B=(120, 352), r=3),
                       view=(120, 330, 4.0), ovl=True, good_txt="GOOD targeted s: lit pavement along the shadow edge, overlay on",
                       bad_txt="BAD targeted s: both ends lit, but it crosses the shadow stripe"),
    "d": dict(good=dict(A=(620, 520), B=(750, 655), r=4), bad=dict(A=(650, 400), B=(750, 655), r=4),
              view=(690, 540, 2.2), good_txt="GOOD d: green grass in sun vs the same grass in shadow, far apart",
              bad_txt="BAD d: the A patch covers many small pebbles (not one material)"),
}


def demo_js(kind, ex):
    t = "s" if kind.startswith("s_") else kind
    pairs = []
    for k in ("good", "bad"):
        e = ex[k]
        p = dict(id=f"t={9000000000000 + len(pairs)}", image=STEM, type=t, A=dict(x=e["A"][0], y=e["A"][1], r=e["r"]),
                 B=dict(x=e["B"][0], y=e["B"][1], r=e["r"]), confidence="sure", used_in_loss=False, overlay_seen=False,
                 annotator="Guide example", created="2026-11-18T00:00:00Z", note="", schema_version="1.1")
        if t == "i":
            p["lit_end"] = "A"
        if t == "s":
            p.update(lighting="lit", pool=kind[2:])
        if t == "d":
            p.update(material="same", illumination="different")
        pairs.append(p)
    vx, vy, z = ex["view"]
    return (f"(()=>{{const S=__ann.S; S.file.pairs=S.file.pairs.filter(p=>p.annotator!=='Guide example').concat({json.dumps(pairs)}); "
            f"S.filters.ann='Guide example'; S.selected=null; S.ovl.illum={'true' if ex.get('ovl') else 'false'}; "
            f"const el=document.getElementById('cv'); S.zoom={z}; S.panX={vx}+0.5-el.clientWidth/S.zoom/2; S.panY={vy}+0.5-el.clientHeight/S.zoom/2; "
            f"__lab.clear(); return true}})()")


def build(url, img):
    s = ST.Steps()
    P = lambda name: str(img / name)
    shot = lambda name, clip=None, fmt="png": s.append(dict({"shot": ST.winpath(img / name), "format": fmt}, **({"quality": 85} if fmt == "jpeg" else {}),
                                                           **({"clip_js": clip} if clip else {})))
    s.append({"nav": url}); s.ready()
    s.ev("lab", LAB)
    # no name tag yet: the banner
    s.key("m"); s.click(300, 520); s.append({"sleep": 300})
    s.ev("l0", "__lab.el('#banner','!','',  '#ff7a59', true); __lab.el('#namesel','1','choose your name tag here','#ffb020', true); true")
    shot("banner_no_name.png", "[0,0,1920,150]")
    s.ev("c0", "__lab.clear()")
    # name-tag dialog (real typing)
    s.ev("open_add", "(()=>{const s=document.getElementById('namesel'); s.value='__new'; s.dispatchEvent(new Event('change')); return true})()")
    s.wait("__ann.modal()"); s.typekeys("Ana Lopez"); shot("dialog_name.png", MODAL_CLIP)
    s.key("Enter"); s.wait("!__ann.modal()")
    # interface tour
    s.ev("hb", "__lab.hideBanner(); true")
    s.ev("tour", """(()=>{__lab.clear(); const g=i=>`#bar .grp:nth-of-type(${i})`;
      __lab.el(g(1),'1','image picker',null,true); __lab.el(g(2),'2','name tag',null,true); __lab.el('#types','3','pair type',null,true);
      __lab.el(g(4),'4','patch size, random s, relabel',null,true); __lab.el(g(5),'5','exposure / gamma / log view',null,true);
      __lab.el('#ovl','6','overlays (off by default)','#4fc3ff',true); __lab.el('#save','7','save status',null,true);
      __lab.el('#stage','8','image canvas: wheel = zoom, right-drag = pan',null,'inside-right'); __lab.el('#hud','9','status bar',null,'above');
      const sec=document.querySelectorAll('#side > section'); __lab.box(sec[0].getBoundingClientRect(),'10','selected pair',null,'inside-right');
      __lab.box(sec[1].getBoundingClientRect(),'11','warnings',null,'inside-right'); __lab.box(sec[2].getBoundingClientRect(),'12','progress against targets',null,'inside-right');
      __lab.el('#counts','13','counts',null,'inside-right'); __lab.el('.filters','14','filters',null,'inside-right'); __lab.el('#list','15','pair list (click to jump)',null,'inside-right'); return true})()""")
    shot("tour.jpg", None, "jpeg")
    s.ev("c1", "__lab.clear()")
    # canvas close-up
    s.ev("zoom", "(()=>{const S=__ann.S, el=document.getElementById('cv'); S.zoom=5; S.panX=700-el.clientWidth/10; S.panY=625-el.clientHeight/10; "
                 "S.selected=null; document.dispatchEvent(new Event('x')); return true})()")
    s.key("f"); s.ev("zoom2", "(()=>{const S=__ann.S, el=document.getElementById('cv'); S.zoom=2.4; S.panX=470-el.clientWidth/4.8; S.panY=400-el.clientHeight/4.8;"
                             " const p=S.file.pairs.filter(q=>q.A.r>=3 && q.A.x>380 && q.A.x<560 && q.A.y>300 && q.A.y<500).sort((a,b)=>b.A.r-a.A.r)[0] || S.file.pairs[0];"
                             " window.__p=p; return true})()")
    s.key("e"); s.key("e")                                     # redraw via a harmless toggle pair
    s.ev("lab_canvas", """(()=>{const p=window.__p; __lab.img(p.A.x-p.A.r-0.5,p.A.y-p.A.r-0.5,p.A.x+p.A.r+0.5,p.A.y+p.A.r+0.5,'A','patch square, (2r+1) px','#fff');
      __lab.img(p.B.x-p.B.r-0.5,p.B.y-p.B.r-0.5,p.B.x+p.B.r+0.5,p.B.y+p.B.r+0.5,'B','','#fff',true);
      const st=document.getElementById('stage').getBoundingClientRect();
      __lab.note(st.left+14, st.top+14, 'pair colours:  i orange · m blue · b purple · s green · d pink · selected white · dashed = unsure · faded = filtered out','#e6e6e6'); return true})()""")
    shot("canvas.jpg", STAGE_CLIP, "jpeg")
    s.ev("c2", "__lab.clear()")
    # live warnings while placing
    s.key("f"); s.key("i"); s.click(700, 595)
    s.append({"move_js": "__ann.scr(700, 700)"}); s.append({"sleep": 200})
    s.ev("lab_warn", "(()=>{const sec=document.querySelectorAll('#side > section'); __lab.box(sec[1].getBoundingClientRect(),'!','live warnings','#ff7a59'); "
                     "__lab.el('#hud','','status bar: what to do next','#ff7a59'); return true})()")
    shot("warnings.jpg", None, "jpeg")
    s.ev("c3", "__lab.clear()"); s.key("Escape")
    # dialogs
    s.key("i"); s.click(700, 595); s.click(700, 652); s.wait("__ann.modal()"); shot("dialog_lit_end.png", MODAL_CLIP); s.key("Escape"); s.wait("!__ann.modal()")
    s.key("3"); s.key("s"); s.click(100, 307); s.click(140, 316); s.wait("__ann.modal()"); shot("dialog_s_lighting.png", MODAL_CLIP)
    s.key("1"); s.wait("__ann.modal() && document.getElementById('m_text')"); s.append({"type": "failure review F10"}); shot("dialog_gap_ref.png", MODAL_CLIP)
    s.key("Escape"); s.wait("!__ann.modal()"); s.key("3")
    s.key("d"); s.click(620, 520); s.click(750, 655); s.wait("__ann.modal()"); shot("dialog_d_material.png", MODAL_CLIP)
    s.key("1"); s.wait("__ann.title().includes('llumination')"); shot("dialog_d_illumination.png", MODAL_CLIP); s.key("Escape"); s.wait("!__ann.modal()")
    s.ev("sel", "(()=>{__ann.S.selected=__ann.S.file.pairs[0].id; return true})()")
    s.key("N"); s.wait("__ann.modal()"); s.append({"type": "half of the B patch is on a drain grate"}); shot("dialog_note.png", MODAL_CLIP); s.key("Escape"); s.wait("!__ann.modal()")
    s.key("r"); s.wait("!!__ann.S.seed"); s.append({"sleep": 200})
    s.ev("lab_seed", "(()=>{const z=__ann.S.seed; __lab.img(z.x0-10,z.y0-10,z.x0+11,z.y0+11,'','proposed seed (yellow); the circle = how far you may nudge (10 px)','#ffef5a'); return true})()")
    shot("seed.jpg", STAGE_CLIP, "jpeg"); s.ev("c4", "__lab.clear()")
    s.key("Delete"); s.wait("__ann.modal()"); s.append({"type": "on a specular highlight"}); shot("dialog_reject.png", MODAL_CLIP)
    s.key("Enter"); s.ready()
    # a pending seed is offered again when the image is reopened
    s.key("r"); s.wait("!!__ann.S.seed"); s.append({"nav": url})
    s.wait("__ann.modal() && __ann.title().includes('assigned to')"); shot("dialog_assigned.png", MODAL_CLIP); s.key("1")
    s.ready(); s.ev("lab2", LAB)
    s.wait("__ann.modal() && __ann.title().includes('still pending')"); shot("dialog_pending_seed.png", MODAL_CLIP)
    s.key("2"); s.wait("__ann.modal() && document.getElementById('m_text')"); s.append({"type": "stopped for the day"})
    shot("dialog_abandon.png", MODAL_CLIP); s.key("Enter"); s.ready()
    # rename the current tag
    s.ev("ren", "(()=>{const s=document.getElementById('namesel'); s.value='__rename'; s.dispatchEvent(new Event('change')); return true})()")
    s.wait("__ann.modal() && document.getElementById('m_text')"); s.typekeys("Ana Lopez Reyes"); shot("dialog_rename.png", MODAL_CLIP)
    s.key("Escape"); s.wait("!__ann.modal()")
    # exhausted mark
    s.click_js("(()=>{const b=document.querySelector('#progress button.exh[data-k=\"b\"]'); const r=b.getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]})()")
    s.wait("__ann.modal() && document.getElementById('m_text')"); s.append({"type": "no other edge where material and shadow change together"})
    shot("dialog_exhausted.png", MODAL_CLIP); s.key("Enter"); s.ready()
    s.key("?"); s.wait("!document.getElementById('help').hidden"); shot("dialog_help.png", HELP_CLIP); s.key("Escape")
    s.key("R"); s.wait("__ann.modal()"); shot("dialog_relabel_which.png", MODAL_CLIP); s.key("Enter")
    s.wait("__ann.modal() && __ann.title().includes('What type')"); shot("relabel.jpg", None, "jpeg"); s.key("Escape"); s.wait("!__ann.S.relabel")
    # progress and pair list panels
    s.ev("panel", "(()=>{const sec=document.querySelectorAll('#side > section'); window.__pr=sec[2].getBoundingClientRect(); window.__li=sec[3].getBoundingClientRect(); return true})()")
    shot("progress.png", "[window.__pr.left-6, window.__pr.top-6, window.__pr.width+12, window.__pr.height+12]")
    shot("pairlist.png", "[window.__li.left-6, window.__li.top-6, window.__li.width+12, Math.min(window.__li.height, 420)+12]")
    # pair-type examples (demonstration pairs, never saved)
    for kind, ex in EX.items():
        s.ev(f"hb_{kind}", "__lab.hideBanner(); true")
        s.ev(f"demo_{kind}", demo_js(kind, ex))
        s.key("e"); s.key("e")
        s.ev(f"lab_{kind}", f"""(()=>{{const pp=__ann.S.file.pairs.filter(p=>p.annotator==='Guide example'), g=pp[0], b=pp[1];
          const bx=p=>[Math.min(p.A.x,p.B.x)-p.A.r-3, Math.min(p.A.y,p.B.y)-p.A.r-3, Math.max(p.A.x,p.B.x)+p.A.r+4, Math.max(p.A.y,p.B.y)+p.A.r+4];
          __lab.img(...bx(g),'GOOD','', '#6fd08c'); __lab.img(...bx(b),'BAD','', '#ff5a4f', true);
          const st=document.getElementById('stage').getBoundingClientRect();
          __lab.note(st.left+14, st.top+14, {json.dumps(ex['good_txt'])}, '#6fd08c'); __lab.note(st.left+14, st.top+48, {json.dumps(ex['bad_txt'])}, '#ff8a80'); return true}})()""")
        shot(f"type_{kind}.jpg", STAGE_CLIP, "jpeg")
        s.ev(f"c_{kind}", "__lab.clear()")
    return s


def main():
    out = M / "runs" / "guide_shots"
    img = M / "docs" / "img"
    if out.exists():
        shutil.rmtree(out)
    (out / "annotations").mkdir(parents=True)
    img.mkdir(parents=True, exist_ok=True)
    shutil.copy2(M / "annotations" / f"{STEM}.json", out / "annotations" / f"{STEM}.json")
    shutil.copy2(M / "annotations" / "annotators.json", out / "annotations" / "annotators.json")
    srv = SV.make_server(HERE / "cache" / "manifest.json", "127.0.0.1", 0, annotations=str(out / "annotations"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        res = ST.run_driver(build(f"http://localhost:{srv.server_address[1]}/?image={STEM}", img), out, width=1920, height=1080)
    finally:
        srv.shutdown()
    print(res["status"])
    print("\n".join(sorted(p.name for p in img.iterdir())))
    return 0 if res["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
