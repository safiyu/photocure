// Burst Collage Builder — Photocure Studio (dependency-free, vanilla JS).
//
// Combines 2–5 frames of a burst into a single square collage with per-cell
// pan/zoom. One committed canvas is the source of truth; the stage shows a
// scaled-down CSS preview of it. No hardcoded canvas size: the user picks the
// export resolution (capped at 3840 px = 4K).
(function (root) {
  "use strict";
  if (root.PCUR_COLLAGE) return;

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const MAX_PX = 3840;
  const icon = (n) => `<svg class="ic sm" aria-hidden="true"><use href="#i-${n}"/></svg>`;
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const LAYOUTS = {
    "2-split-v": (w, h) => [{ x: 0, y: 0, w: w / 2, h: h }, { x: w / 2, y: 0, w: w / 2, h: h }],
    "2-split-h": (w, h) => [{ x: 0, y: 0, w, h: h / 2 }, { x: 0, y: h / 2, w, h: h / 2 }],
    "3-cols": (w, h) => [
      { x: 0, y: 0, w: w / 3, h }, { x: w / 3, y: 0, w: w / 3, h }, { x: (w / 3) * 2, y: 0, w: w / 3, h },
    ],
    "3-rows": (w, h) => [
      { x: 0, y: 0, w, h: h / 3 }, { x: 0, y: h / 3, w, h: h / 3 }, { x: 0, y: (h / 3) * 2, w, h: h / 3 },
    ],
    "3-left-main": (w, h) => [
      { x: 0, y: 0, w: w / 2, h }, { x: w / 2, y: 0, w: w / 2, h: h / 2 }, { x: w / 2, y: h / 2, w: w / 2, h: h / 2 },
    ],
    "3-top-main": (w, h) => [
      { x: 0, y: 0, w, h: h / 2 }, { x: 0, y: h / 2, w: w / 2, h: h / 2 }, { x: w / 2, y: h / 2, w: w / 2, h: h / 2 },
    ],
    "4-grid": (w, h) => [
      { x: 0, y: 0, w: w / 2, h: h / 2 }, { x: w / 2, y: 0, w: w / 2, h: h / 2 },
      { x: 0, y: h / 2, w: w / 2, h: h / 2 }, { x: w / 2, y: h / 2, w: w / 2, h: h / 2 },
    ],
    "4-cols": (w, h) => [
      { x: 0, y: 0, w: w / 4, h }, { x: w / 4, y: 0, w: w / 4, h },
      { x: w / 2, y: 0, w: w / 4, h }, { x: (w / 4) * 3, y: 0, w: w / 4, h },
    ],
    "4-rows": (w, h) => [
      { x: 0, y: 0, w, h: h / 4 }, { x: 0, y: h / 4, w, h: h / 4 },
      { x: 0, y: h / 2, w, h: h / 4 }, { x: 0, y: (h / 4) * 3, w, h: h / 4 },
    ],
    "5-grid-mixed": (w, h) => [
      { x: 0, y: 0, w: w / 2, h: h / 2 }, { x: w / 2, y: 0, w: w / 2, h: h / 2 },
      { x: 0, y: h / 2, w: w / 3, h: h / 2 }, { x: w / 3, y: h / 2, w: w / 3, h: h / 2 },
      { x: (w / 3) * 2, y: h / 2, w: w / 3, h: h / 2 },
    ],
    "5-cols": (w, h) => [
      { x: 0, y: 0, w: w / 5, h }, { x: w / 5, y: 0, w: w / 5, h }, { x: (w / 5) * 2, y: 0, w: w / 5, h },
      { x: (w / 5) * 3, y: 0, w: w / 5, h }, { x: (w / 5) * 4, y: 0, w: w / 5, h },
    ],
  };
  const LAYOUTS_FOR = {
    2: ["2-split-v", "2-split-h"],
    3: ["3-cols", "3-rows", "3-left-main", "3-top-main"],
    4: ["4-grid", "4-cols", "4-rows"],
    5: ["5-grid-mixed", "5-cols"],
  };
  const LAYOUT_LABELS = {
    "2-split-v": "Split ·", "2-split-h": "Split —",
    "3-cols": "3 columns", "3-rows": "3 rows", "3-left-main": "Left main", "3-top-main": "Top main",
    "4-grid": "2×2 grid", "4-cols": "4 columns", "4-rows": "4 rows",
    "5-grid-mixed": "5 mixed", "5-cols": "5 columns",
  };
  const SIZE_CHOICES = [1080, 1600, 2400, MAX_PX];
  const IDENTITY = { zoom: 1, panX: 0, panY: 0 };

  function toast(msg, cls) {
    const box = $("#toasts");
    if (!box) return;
    const n = document.createElement("div");
    n.className = "toast " + (cls || "");
    n.textContent = msg;
    box.appendChild(n);
    setTimeout(() => n.remove(), 3200);
  }

  // Defensive image load: onerror + 15 s timeout.
  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      let done = false;
      const fail = (m) => { if (!done) { done = true; clearTimeout(t); reject(new Error(m)); } };
      const t = setTimeout(() => fail("timed out"), 15000);
      img.onload = () => { if (!done) { done = true; clearTimeout(t); resolve(img); } };
      img.onerror = () => fail("failed to load");
      img.src = url;
    });
  }

  const state = { active: false, selected: [], layout: "2-split-v", size: 0, transforms: [] };

  async function openBurstCollage(members, bestId, opts) {
    opts = opts || {};
    const onSaved = opts.onSaved;
    if (!members || members.length < 2) { toast("A collage needs at least two frames", "err"); return null; }

    // Load every member's oriented pixels (the backend already applied EXIF).
    const loaded = await Promise.all(members.map(async (m) => {
      try {
        const res = await fetch("/api/images/" + m.id + "/full");
        if (!res.ok) throw new Error("HTTP " + res.status);
        const blob = await res.blob();
        const bmp = await createImageBitmap(blob).catch(async () => {
          const img = await loadImage(URL.createObjectURL(blob));
          return img;
        });
        return { m, src: bmp, bmp: !("close" in bmp) };
      } catch (e) { return { m, src: null, bmp: false, error: e }; }
    }));
    const bad = loaded.filter((l) => !l.src);
    if (bad.length) toast(`Could not load ${bad.length} frame${bad.length > 1 ? "s" : ""}`, "err");
    if (loaded.every((l) => !l.src)) { toast("No frames could be loaded", "err"); return null; }

    // Pre-select: the best shot first, then the next frames up to 3.
    const byId = (id) => members.findIndex((m) => m.id === id);
    const sel = [bestId != null ? byId(bestId) : 0, 1, 2].filter((i, k, a) =>
      i >= 0 && i < loaded.length && a.indexOf(i) === k && i < loaded.length).slice(0, 3);
    if (sel.length < 2) for (let i = 0; i < loaded.length && sel.length < 2; i++) if (sel.indexOf(i) === -1) sel.push(i);
    state.selected = sel;
    state.transforms = loaded.map(() => ({ ...IDENTITY }));
    state.size = 0; // 0 = auto (max member dimension, capped)

    // ---- DOM ----
    const root = document.createElement("div");
    root.className = "pcur-cg";
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.setAttribute("aria-label", "Burst collage builder");
    root.innerHTML = `
      <div class="pcur-cg-back" data-close></div>
      <div class="pcur-cg-dlg">
        <div class="pcur-cg-top">
          <span class="pcur-cg-title">Burst collage</span>
          <span class="pcur-cg-count" data-count></span>
          <button type="button" class="pcur-x" data-close aria-label="Close">${icon("x")}</button>
        </div>
        <div class="pcur-cg-body">
          <div class="pcur-cg-side">
            <div class="pcur-cg-sec">Frames <span class="pcur-cg-sub">(2–5)</span></div>
            <div class="pcur-cg-frames" data-frames></div>
            <div class="pcur-cg-sec">Layout</div>
            <div class="pcur-cg-layouts" data-layouts></div>
            <div class="pcur-cg-sec">Size</div>
            <div class="pcur-cg-sizes" data-sizes></div>
          </div>
          <div class="pcur-cg-stage">
            <canvas data-canvas></canvas>
            <div class="pcur-cg-hint">Drag inside a cell to pan · scroll to zoom (1–3×)</div>
          </div>
        </div>
        <div class="pcur-cg-foot">
          <button type="button" class="btn sm secondary" data-dl>${icon("down")}<span>Download</span></button>
          <button type="button" class="btn sm primary" data-save>${icon("check")}<span>Save to Library</span></button>
        </div>
      </div>`;
    document.body.appendChild(root);
    document.body.style.overflow = "hidden";

    const framesEl = root.querySelector("[data-frames]");
    const layoutsEl = root.querySelector("[data-layouts]");
    const sizesEl = root.querySelector("[data-sizes]");
    const canvas = root.querySelector("[data-canvas]");
    const countEl = root.querySelector("[data-count]");
    const saveBtn = root.querySelector("[data-save]");
    const dlBtn = root.querySelector("[data-dl]");
    const ctx = canvas.getContext("2d");
    let closed = false;
    let drag = null;

    function targetSize() {
      if (state.size > 0) return state.size;
      let mx = 0;
      for (const l of loaded) if (l.src) mx = Math.max(mx, l.src.width, l.src.height);
      return Math.min(MAX_PX, Math.max(640, mx));
    }

    function drawCell(i, r) {
      if (!r) return;
      const item = loaded[i];
      if (!item || !item.src) return;
      const img = item.src;
      const t = state.transforms[i] || IDENTITY;
      const rectRatio = r.w / r.h;
      const imgRatio = img.width / img.height;
      let drawW, drawH, drawX, drawY;
      if (imgRatio > rectRatio) {
        drawH = img.height; drawW = img.height * rectRatio; drawX = (img.width - drawW) / 2; drawY = 0;
      } else {
        drawW = img.width; drawH = img.width / rectRatio; drawX = 0; drawY = (img.height - drawH) / 2;
      }
      const zf = Math.max(0.1, t.zoom);
      const zoomedW = drawW / zf, zoomedH = drawH / zf;
      let sx = drawX + (drawW - zoomedW) / 2 - t.panX * drawW;
      let sy = drawY + (drawH - zoomedH) / 2 - t.panY * drawH;
      // Clamp the source window so the destination cell stays covered.
      const cx0 = Math.max(drawX, sx), cx1 = Math.min(drawX + drawW, sx + zoomedW);
      const cy0 = Math.max(drawY, sy), cy1 = Math.min(drawY + drawH, sy + zoomedH);
      if (cx1 - cx0 <= 0.5 || cy1 - cy0 <= 0.5) {
        ctx.drawImage(img, r.x, r.y, r.w, r.h);
        return;
      }
      const dx = (cx0 - sx) / zoomedW * r.w, dy = (cy0 - sy) / zoomedH * r.h;
      const dw = (cx1 - cx0) / zoomedW * r.w, dh = (cy1 - cy0) / zoomedH * r.h;
      ctx.save();
      ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
      ctx.fillStyle = "#ffffff"; ctx.fillRect(r.x, r.y, r.w, r.h);
      ctx.drawImage(img, cx0, cy0, cx1 - cx0, cy1 - cy0, r.x + dx, r.y + dy, dw, dh);
      ctx.restore();
    }

    function render() {
      const n = state.selected.length;
      const size = targetSize();
      if (canvas.width !== size) canvas.width = size;
      if (canvas.height !== size) canvas.height = size;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, size, size);
      if (n < 2) {
        ctx.fillStyle = "#98a2b3";
        ctx.font = `${Math.round(size / 22)}px system-ui, sans-serif`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("Select at least two frames", size / 2, size / 2);
        countEl.textContent = n + " selected";
        return;
      }
      const rects = (LAYOUTS[state.layout] || LAYOUTS["2-split-v"])(size, size);
      state.selected.forEach((selIdx, i) => drawCell(i, rects[i]));
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = Math.max(2, Math.round(size / 300));
      for (const r of rects.slice(0, n)) ctx.strokeRect(r.x, r.y, r.w, r.h);
      countEl.textContent = n + " selected";
    }

    // ---- frame selector ----
    function buildFrames() {
      framesEl.innerHTML = "";
      loaded.forEach((l, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "pcur-cg-frame" + (state.selected.indexOf(i) !== -1 ? " on" : "");
        b.dataset.idx = String(i);
        b.title = l.m.filename || ("Frame " + (i + 1));
        b.setAttribute("aria-pressed", state.selected.indexOf(i) !== -1 ? "true" : "false");
        if (l.src) {
          const t = document.createElement("img");
          t.src = "/thumbnail/" + l.m.id + "?tier=medium";
          t.alt = "";
          t.onerror = () => t.remove();
          b.appendChild(t);
        }
        const lab = document.createElement("span");
        lab.textContent = i + 1;
        b.appendChild(lab);
        b.onclick = () => {
          const k = state.selected.indexOf(i);
          if (k !== -1) state.selected.splice(k, 1);
          else {
            if (state.selected.length >= 5) { toast("Up to five frames in one collage", "err"); return; }
            state.selected.push(i); state.selected.sort((a, b2) => a - b2);
          }
          ensureLayout(); buildFrames(); buildLayouts(); render();
        };
        framesEl.appendChild(b);
      });
    }

    function ensureLayout() {
      const n = state.selected.length;
      if (n >= 2 && !(LAYOUTS_FOR[n] || []).includes(state.layout)) state.layout = (LAYOUTS_FOR[n] || [])[0] || "2-split-v";
    }

    function buildLayouts() {
      const n = state.selected.length;
      layoutsEl.innerHTML = "";
      const opts = n >= 2 ? LAYOUTS_FOR[n] : [];
      if (!opts.length) {
        const p = document.createElement("p");
        p.className = "pcur-hint"; p.textContent = "Pick 2–5 frames first.";
        layoutsEl.appendChild(p);
        return;
      }
      for (const id of opts) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "chip" + (state.layout === id ? " on" : "");
        b.textContent = LAYOUT_LABELS[id] || id;
        b.dataset.layout = id;
        b.onclick = () => { state.layout = id; buildLayouts(); render(); };
        layoutsEl.appendChild(b);
      }
    }

    function buildSizes() {
      sizesEl.innerHTML = "";
      const auto = document.createElement("button");
      auto.type = "button";
      auto.className = "chip" + (state.size === 0 ? " on" : "");
      auto.textContent = "Auto";
      auto.onclick = () => { state.size = 0; buildSizes(); render(); };
      sizesEl.appendChild(auto);
      for (const s of SIZE_CHOICES) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "chip" + (state.size === s ? " on" : "");
        b.textContent = s + " px";
        b.onclick = () => { state.size = s; buildSizes(); render(); };
        sizesEl.appendChild(b);
      }
    }

    // ---- per-cell pan/zoom on the stage ----
    function cellAt(ev) {
      const rect = canvas.getBoundingClientRect();
      const n = state.selected.length;
      if (n < 2) return -1;
      const size = targetSize();
      const x = (ev.clientX - rect.left) * (size / rect.width);
      const y = (ev.clientY - rect.top) * (size / rect.height);
      const rects = (LAYOUTS[state.layout] || LAYOUTS["2-split-v"])(size, size);
      for (let i = 0; i < n && i < rects.length; i++) {
        const r = rects[i];
        if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) return i;
      }
      return -1;
    }
    canvas.addEventListener("pointerdown", (ev) => {
      const i = cellAt(ev);
      if (i < 0) return;
      ev.preventDefault();
      canvas.setPointerCapture(ev.pointerId);
      drag = { i, x: ev.clientX, y: ev.clientY, t: { ...state.transforms[i] } };
    });
    canvas.addEventListener("pointermove", (ev) => {
      if (!drag) return;
      const rect = canvas.getBoundingClientRect();
      const size = targetSize();
      const t = state.transforms[drag.i];
      t.panX = clampPan(drag.t.panX - (ev.clientX - drag.x) * (size / rect.width) / size, t.zoom);
      t.panY = clampPan(drag.t.panY - (ev.clientY - drag.y) * (size / rect.height) / size, t.zoom);
      render();
    });
    const endDrag = () => { drag = null; };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("wheel", (ev) => {
      const i = cellAt(ev);
      if (i < 0) return;
      ev.preventDefault();
      const t = state.transforms[i];
      t.zoom = clampZoom(t.zoom * (ev.deltaY < 0 ? 1.12 : 1 / 1.12));
      render();
    }, { passive: false });

    function clampZoom(z) { return Math.min(3, Math.max(1, z)); }
    function clampPan(p, zoom) {
      const m = (zoom - 1) / zoom;
      return Math.min(m, Math.max(-m, p));
    }

    // ---- save / download ----
    function renderBlob(type, quality) {
      const out = document.createElement("canvas");
      out.width = canvas.width; out.height = canvas.height;
      out.getContext("2d").drawImage(canvas, 0, 0);
      return new Promise((resolve, reject) =>
        out.toBlob((b) => (b ? resolve(b) : reject(new Error("render failed"))), type, quality));
    }
    async function saveToLibrary() {
      if (state.selected.length < 2) { toast("Select at least two frames", "err"); return; }
      busy(saveBtn, true);
      try {
        const blob = await renderBlob("image/jpeg", 0.92);
        const fd = new FormData();
        fd.append("file", blob, "collage.jpg");
        fd.append("mode", "copy");
        fd.append("format", "jpg");
        const host = bestId != null && members.find((m) => m.id === bestId) ? bestId : members[0].id;
        const res = await fetch(`/api/images/${host}/edit-save`, { method: "POST", body: fd });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || "save failed");
        toast("Collage saved to the library", "ok");
        if (onSaved) onSaved(data);
        close();
      } catch (e) { toast("Could not save the collage: " + e.message, "err"); }
      finally { busy(saveBtn, false); }
    }
    async function download() {
      if (state.selected.length < 2) { toast("Select at least two frames", "err"); return; }
      busy(dlBtn, true);
      try {
        const blob = await renderBlob("image/png");
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = "burst-collage.png";
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 4000);
      } catch (e) { toast("Download failed: " + e.message, "err"); }
      finally { busy(dlBtn, false); }
    }
    function busy(b, on) { b.disabled = on; b.classList.toggle("busy", on); }

    // ---- lifecycle ----
    function close() {
      if (closed) return;
      closed = true;
      state.active = false;
      window.removeEventListener("keydown", onKey, true);
      document.body.style.overflow = "";
      root.remove();
      for (const l of loaded) if (l.bmp && l.src && l.src.close) { try { l.src.close(); } catch (e) { /* */ } }
    }
    function onKey(ev) { if (ev.key === "Escape") { ev.preventDefault(); close(); } }
    root.querySelector("[data-close]").addEventListener("click", close);
    $$("[data-close]", root).forEach((n) => n.addEventListener("click", close));
    window.addEventListener("keydown", onKey, true);
    saveBtn.addEventListener("click", saveToLibrary);
    dlBtn.addEventListener("click", download);

    state.active = true;
    state.selected = sel.slice();
    ensureLayout();
    buildFrames(); buildLayouts(); buildSizes(); render();
    state._api = {
      set: (o) => { Object.assign(state, o); ensureLayout(); buildFrames(); buildLayouts(); buildSizes(); render(); },
      render,
      state,
    };

    return { close, state, api: state._api };
  }

  state.openBurstCollage = openBurstCollage;
  root.PCUR_COLLAGE = state;
})(typeof window !== "undefined" ? window : globalThis);

export { };
