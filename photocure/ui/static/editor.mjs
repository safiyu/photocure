/* Photocure Photo Editor Studio — dependency-free vanilla JS.
 *
 * Architecture (per docs/implementation_plan.md §1, §4.1):
 *  - Single canvas pipeline: `workC` holds the committed pixels. Every tool bakes
 *    directly into it. There is no parallel "actions" list and no replay, so a
 *    filter can never wipe an earlier draw (Ranthal's dual-track bug).
 *  - Live previews (adjust / filters / blur / pending flip-rotate) are pure CSS on
 *    the stage wrapper; they are baked with a temp canvas + ctx.filter on commit.
 *  - Strict coordinate normalizer: screenToCanvas()/canvasToScreen() derive from
 *    the overlay canvas' live bounding rect, which already folds in fit scale,
 *    zoom, pan (and any pending preview transform).
 *  - Bounded history: at most 10 states stored as compressed Blobs decoded to
 *    ImageBitmap; every temporary object URL is revoked; bitmaps are closed on
 *    close.
 *  - Remaster uses Math.max(1, max-min) plus a 1–99% percentile stretch.
 *  - The client never re-rotates EXIF: /api/images/{id}/full already delivers
 *    upright pixels (orientation=1).
 *  - Defensive image promises: onerror + 15 s timeout + toast on failure.
 *  - Configurable resolution: the working canvas keeps the source size up to a
 *    4K (3840px) cap; exports use the chosen format/quality.
 */
(function (root) {
  "use strict";
  if (root.PCUR_EDITOR) return;

  const MAX_PX = 3840;          // 4K cap for web export
  const HIST_MAX = 10;          // bounded undo states
  const IMG_TIMEOUT = 15000;    // defensive image-load timeout
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (n) => Number(n) || 0;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const icon = (n) => `<svg class="ic sm" aria-hidden="true"><use href="#i-${n}"/></svg>`;

  // 33 filter presets (Classic + Social). Previewed via CSS filter, baked via ctx.filter.
  const FILTERS = [
    ["none", "Original", "none"], ["grayscale", "B&W", "grayscale(100%)"],
    ["sepia", "Sepia", "sepia(100%)"], ["vintage", "Vintage", "sepia(50%) contrast(90%) brightness(90%)"],
    ["warm", "Warm", "sepia(30%) saturate(120%) brightness(105%)"], ["cool", "Cool", "saturate(80%) hue-rotate(20deg) brightness(95%)"],
    ["highContrast", "High Contrast", "contrast(150%) saturate(110%)"], ["noir", "Noir", "grayscale(100%) contrast(120%) brightness(90%)"],
    ["fade", "Fade", "contrast(80%) brightness(110%) saturate(80%)"], ["kodak", "Kodak", "sepia(20%) saturate(160%) contrast(110%) brightness(105%)"],
    ["technicolor", "Technicolor", "saturate(200%) contrast(130%) hue-rotate(-10deg)"], ["polaroid", "Polaroid", "contrast(110%) brightness(110%) grayscale(20%) sepia(20%)"],
    ["dramatic", "Dramatic", "contrast(140%) grayscale(30%) brightness(90%)"], ["golden", "Golden", "sepia(40%) saturate(150%) brightness(110%) contrast(110%)"],
    ["cyberpunk", "Cyberpunk", "hue-rotate(180deg) saturate(200%) contrast(130%)"], ["clarendon", "Clarendon", "sepia(10%) contrast(120%) brightness(125%) saturate(135%)"],
    ["gingham", "Gingham", "sepia(10%) hue-rotate(-10deg) brightness(105%) contrast(110%) saturate(80%)"], ["juno", "Juno", "sepia(30%) contrast(115%) brightness(110%) saturate(140%) hue-rotate(-10deg)"],
    ["lark", "Lark", "contrast(90%) brightness(120%) saturate(110%)"], ["ludwig", "Ludwig", "sepia(10%) contrast(105%) brightness(105%) saturate(180%)"],
    ["valencia", "Valencia", "sepia(25%) contrast(108%) brightness(108%)"], ["moon", "Moon", "grayscale(100%) brightness(110%) contrast(110%)"],
    ["reyes", "Reyes", "sepia(22%) brightness(110%) contrast(85%) saturate(75%)"], ["slumber", "Slumber", "sepia(35%) contrast(125%) saturate(125%)"],
    ["crema", "Crema", "sepia(50%) contrast(125%) saturate(90%) hue-rotate(-2deg)"], ["aden", "Aden", "hue-rotate(-20deg) contrast(90%) saturate(85%) brightness(120%)"],
    ["perpetua", "Perpetua", "contrast(110%) brightness(110%) saturate(110%)"], ["oldCinema", "Old Cinema", "grayscale(90%) contrast(160%) brightness(85%) sepia(20%)"],
    ["retro", "Retro", "sepia(40%) saturate(140%) contrast(110%) brightness(95%)"], ["twilight", "Twilight", "sepia(40%) hue-rotate(180deg) saturate(140%) contrast(110%)"],
    ["sunset", "Sunset", "sepia(30%) saturate(160%) hue-rotate(-10deg) contrast(110%)"], ["forest", "Forest", "sepia(20%) brightness(95%) contrast(110%) saturate(120%) hue-rotate(60deg)"],
    ["rust", "Rust", "sepia(50%) saturate(150%) contrast(130%) hue-rotate(-20deg)"],
  ];

  const STICKER_CATS = [
    { id: "fun", label: "Fun", items: ["😀","😂","😍","😎","🤔","😭","😡","👍","👎","🎉","🔥","❤️","💯","💩","👻","🤖"] },
    { id: "nature", label: "Nature", items: ["🐶","🐱","🐭","🦁","🐸","🌸","🌹","🌻","🌲","🌵","🍎","🍌","🍕","🍔","🍦"] },
    { id: "objects", label: "Objects", items: ["⚽","🏀","🎮","📱","💻","📷","🚗","✈️","🚀","💡","⏰","🎁","🎈","🎵","🕶️"] },
    { id: "signs", label: "Signs", items: ["⚠️","⛔","✅","❌","➡️","⬅️","⬆️","⬇️","🔴","🔵","⭐","💲","❗","❓"] },
    { id: "custom", label: "My Stickers", items: [] },
  ];
  const CUSTOM_STICKER_KEY = "pcur_custom_stickers";

  const TOOLS = [
    ["crop", "Crop", "grid"], ["transform", "Transform", "swap"], ["adjust", "Adjust", "sun"],
    ["filters", "Filters", "cpu"], ["remaster", "Remaster", "image"], ["blur", "Blur / Sharpen", "eye"],
    ["redeye", "Red-Eye", "face"], ["bg", "Remove BG", "layers"], ["draw", "Draw", "pen"],
    ["stickers", "Stickers", "star"], ["idcard", "ID Card", "cal"], ["qr", "QR Code", "copy"],
    ["ocr", "OCR", "search"],
  ];

  // --------------------------------------------------------------- small utils
  function makeCanvas(w, h) {
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(w));
    c.height = Math.max(1, Math.round(h));
    return c;
  }
  function ctx2d(c) {
    const x = c.getContext("2d", { willReadFrequently: true });
    if (!x) throw new Error("Could not get a 2D canvas context");
    return x;
  }
  function toast(msg, cls) {
    const box = $("#toasts");
    if (!box) return;
    const n = document.createElement("div");
    n.className = "toast " + (cls || "");
    n.textContent = msg;
    box.appendChild(n);
    setTimeout(() => { n.classList.add("out"); setTimeout(() => n.remove(), 260); }, 3400);
  }
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }
  // Defensive image promise: explicit onerror + timeout. No EXIF handling here —
  // the backend already delivers oriented pixels.
  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      let done = false;
      const t = setTimeout(() => finish(new Error("Image load timed out — the file may be corrupt")), IMG_TIMEOUT);
      function finish(err) { if (!done) { done = true; clearTimeout(t); err ? reject(err) : resolve(img); } }
      img.onload = () => finish(null);
      img.onerror = () => finish(new Error("Failed to load image"));
      img.src = url;
    });
  }
  function blobToBitmap(blob) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      createImageBitmap(blob).then((b) => { URL.revokeObjectURL(url); resolve(b); })
        .catch((e) => { URL.revokeObjectURL(url); reject(e); });
    });
  }
  function canvasToBlob(c, type, quality) {
    return new Promise((resolve, reject) => {
      let done = false;
      const t = setTimeout(() => { if (!done) { done = true; reject(new Error("Image encode timed out")); } }, 20000);
      try {
        c.toBlob((b) => { if (done) return; done = true; clearTimeout(t); b ? resolve(b) : reject(new Error("Image encode failed")); }, type, quality);
      } catch (e) { if (!done) { done = true; clearTimeout(t); reject(e); } }
    });
  }
  function fitArea(iw, ih, mw, mh) {
    mw = Math.max(1, mw); mh = Math.max(1, mh);
    const ir = iw / ih, ar = mw / mh;
    if (ir > ar) { const w = mw; return { w, h: Math.max(1, w / ir) }; }
    const h = mh; return { w: Math.max(1, h * ir), h };
  }
  function roundRectPath(x, px, py, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    x.beginPath();
    x.moveTo(px + r, py);
    x.arcTo(px + w, py, px + w, py + h, r);
    x.arcTo(px + w, py + h, px, py + h, r);
    x.arcTo(px, py + h, px, py, r);
    x.arcTo(px, py, px + w, py, r);
    x.closePath();
  }
  // Minimal single-page PDF (JPEG embedded, DCTDecode) — no external library.
  async function jpegToPdf(jpegU8, w, h) {
    const enc = new TextEncoder();
    const parts = [];
    const tell = () => parts.reduce((a, p) => a + p.length, 0);
    const offs = [];
    const obj = (n, body) => { offs.push(tell()); parts.push(enc.encode(`${n} 0 obj\n${body}\nendobj\n`)); };
    parts.push(enc.encode("%PDF-1.4\n"));
    const W = (w / 72).toFixed(2), H = (h / 72).toFixed(2);
    obj(1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(3, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${W} ${H}] /Resources << /XObject << /Im1 4 0 R >> /ProcSet [/PDF /ImageC] >> /Contents 5 0 R >>`);
    offs.push(tell());
    parts.push(enc.encode(`4 0 obj\n<< /Type /XObject /Subtype /Image /Width ${w} /Height ${h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpegU8.length} >>\nstream\n`));
    parts.push(jpegU8);
    parts.push(enc.encode("\nendstream\nendobj\n"));
    const content = `q ${W} 0 0 ${H} 0 0 cm /Im1 Do Q`;
    obj(5, `<< /Length ${content.length} >>\nstream\n${content}\nendstream\n`);
    const xref = tell();
    let s = `xref\n0 ${offs.length + 1}\n0000000000 65535 f \n`;
    for (const o of offs) s += String(o).padStart(10, "0") + " 00000 n \n";
    parts.push(enc.encode(s + `trailer\n<< /Size ${offs.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`));
    const total = parts.reduce((a, p) => a + p.length, 0);
    const out = new Uint8Array(total);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return new Blob([out], { type: "application/pdf" });
  }

  // --------------------------------------------------------------- pixel ops
  // Remaster: 1–99% percentile stretch (zero-division guarded) + 15% saturation + mild sharpen.
  function remaster(ctx, w, h) {
    const img = ctx.getImageData(0, 0, w, h);
    const d = img.data;
    const n = w * h;
    const hist = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
    const step = Math.max(1, Math.floor(n / 400000));
    let samples = 0;
    for (let i = 0; i < n; i += step) {
      const p = i * 4;
      hist[0][d[p]]++; hist[1][d[p + 1]]++; hist[2][d[p + 2]]++;
      samples++;
    }
    const pctl = (hst, t) => {
      let acc = 0;
      for (let v = 0; v < 256; v++) { acc += hst[v]; if (acc >= t) return v; }
      return 255;
    };
    const lo = [pctl(hist[0], samples * 0.01), pctl(hist[1], samples * 0.01), pctl(hist[2], samples * 0.01)];
    const hi = [pctl(hist[0], samples * 0.99), pctl(hist[1], samples * 0.99), pctl(hist[2], samples * 0.99)];
    // Math.max(1, …) keeps flat / solid-color images from dividing by zero.
    const scale = [0, 1, 2].map((c) => 255 / Math.max(1, hi[c] - lo[c]));
    for (let i = 0; i < d.length; i += 4) {
      let r = clamp((d[i] - lo[0]) * scale[0], 0, 255);
      let g = clamp((d[i + 1] - lo[1]) * scale[1], 0, 255);
      let b = clamp((d[i + 2] - lo[2]) * scale[2], 0, 255);
      const gray = 0.2989 * r + 0.5870 * g + 0.1140 * b;
      d[i] = clamp(gray + (r - gray) * 1.15, 0, 255);
      d[i + 1] = clamp(gray + (g - gray) * 1.15, 0, 255);
      d[i + 2] = clamp(gray + (b - gray) * 1.15, 0, 255);
    }
    ctx.putImageData(img, 0, 0);
    sharpen(ctx, w, h, 0.5);
  }
  function sharpen(ctx, w, h, amount) {
    if (amount <= 0) return;
    const img = ctx.getImageData(0, 0, w, h);
    const src = new Uint8ClampedArray(img.data);
    const o = img.data;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = (y * w + x) * 4;
        for (let c = 0; c < 3; c++) {
          const v = src[i + c];
          const lap = v * 5 - (src[i - w * 4 + c] + src[i + w * 4 + c] + src[i - 4 + c] + src[i + 4 + c]);
          o[i + c] = clamp(v + (lap - v) * amount, 0, 255);
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  function redEye(ctx, w, h, cx, cy, radius) {
    const x0 = Math.max(0, Math.floor(cx - radius)), y0 = Math.max(0, Math.floor(cy - radius));
    const x1 = Math.min(w, Math.ceil(cx + radius)), y1 = Math.min(h, Math.ceil(cy + radius));
    if (x1 - x0 < 1 || y1 - y0 < 1) return;
    const img = ctx.getImageData(x0, y0, x1 - x0, y1 - y0);
    const d = img.data;
    const r2 = radius * radius;
    for (let y = 0; y < y1 - y0; y++) {
      for (let x = 0; x < x1 - x0; x++) {
        const dx = x + x0 - cx, dy = y + y0 - cy;
        if (dx * dx + dy * dy > r2) continue;
        const i = (y * (x1 - x0) + x) * 4;
        const r = d[i], g = d[i + 1], b = d[i + 2];
        if (r > 60 && r > g * 1.5 && r > b * 1.5) {
          const lum = 0.2989 * r + 0.5870 * g + 0.1140 * b;
          d[i] = clamp(lum * 0.5, 0, 255);
          d[i + 2] = clamp(b + 12, 0, 255);
        }
      }
    }
    ctx.putImageData(img, x0, y0);
  }
  function cartoonOutline(src, stroke) {
    stroke = Math.max(1, Math.round(stroke));
    const pad = stroke * 2;
    const c = makeCanvas(src.width + pad * 2, src.height + pad * 2);
    const x = ctx2d(c);
    for (let a = 0; a < Math.PI * 2; a += Math.PI / 12)
      x.drawImage(src, pad + Math.cos(a) * stroke, pad + Math.sin(a) * stroke);
    const od = x.getImageData(0, 0, c.width, c.height);
    for (let i = 0; i < od.data.length; i += 4)
      if (od.data[i + 3] > 0) { od.data[i] = od.data[i + 1] = od.data[i + 2] = 255; }
    x.putImageData(od, 0, 0);
    x.drawImage(src, pad, pad);
    return c;
  }
  function despeckle(ctx, w, h) {
    const img = ctx.getImageData(0, 0, w, h);
    const d = img.data;
    for (let i = 0; i < d.length; i += 4)
      if (d[i + 3] < 50) d[i + 3] = 0;
      else if (d[i + 3] > 200) d[i + 3] = 255;
    const copy = new Uint8ClampedArray(d);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = (y * w + x) * 4;
        if (copy[i + 3] > 0) {
          let nb = 0;
          if (copy[i - 4 + 3] > 0) nb++;
          if (copy[i + 4 + 3] > 0) nb++;
          if (copy[i - w * 4 + 3] > 0) nb++;
          if (copy[i + w * 4 + 3] > 0) nb++;
          if (nb < 2) d[i + 3] = 0;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  function trimTransparency(c) {
    const x = ctx2d(c);
    const d = x.getImageData(0, 0, c.width, c.height).data;
    let minX = c.width, minY = c.height, maxX = -1, maxY = -1;
    for (let y = 0; y < c.height; y++) {
      for (let px = 0; px < c.width; px++) {
        if (d[(y * c.width + px) * 4 + 3] > 10) {
          if (px < minX) minX = px; if (px > maxX) maxX = px;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return c;
    minX = Math.max(0, minX - 8); minY = Math.max(0, minY - 8);
    maxX = Math.min(c.width - 1, maxX + 8); maxY = Math.min(c.height - 1, maxY + 8);
    const tw = maxX - minX + 1, th = maxY - minY + 1;
    const nc = makeCanvas(tw, th);
    ctx2d(nc).drawImage(c, minX, minY, tw, th, 0, 0, tw, th);
    return nc;
  }
  function emojiToBitmap(ch, size) {
    size = size || 384;
    const c = makeCanvas(size, size);
    const x = ctx2d(c);
    x.font = `${Math.floor(size * 0.8)}px serif`;
    x.textAlign = "center"; x.textBaseline = "middle";
    x.fillText(ch, size / 2, size / 2 + size * 0.06);
    return createImageBitmap(c).catch(() => c);
  }
  function dataUrlToBitmap(url) {
    return loadImage(url).then((img) => createImageBitmap(img).catch(() => img));
  }
  function qrToBitmap(matrix) {
    const n = matrix.length, margin = 4, cell = 12;
    const size = (n + margin * 2) * cell;
    const c = makeCanvas(size, size);
    const x = ctx2d(c);
    x.fillStyle = "#fff"; x.fillRect(0, 0, size, size);
    x.fillStyle = "#000";
    for (let y = 0; y < n; y++)
      for (let px = 0; px < n; px++)
        if (matrix[y][px]) x.fillRect((px + margin) * cell, (y + margin) * cell, cell, cell);
    return createImageBitmap(c).catch(() => c);
  }
  async function loadCustomStickers() {
    try {
      const arr = JSON.parse(localStorage.getItem(CUSTOM_STICKER_KEY) || "[]");
      return Array.isArray(arr) ? arr : [];
    } catch (e) { return []; }
  }
  function saveCustomStickers(arr) {
    try { localStorage.setItem(CUSTOM_STICKER_KEY, JSON.stringify(arr.slice(0, 20))); } catch (e) { /* storage full */ }
  }

  // --------------------------------------------------------------- main entry
  const state = { active: false, histLen: 0 };

  async function openPhotoEditor(imageId, opts) {
    opts = opts || {};
    const onSaved = opts.onSaved;
    const onClose = opts.onClose;
    const baseName = opts.filename || ("photo-" + num(imageId));
    const id = num(imageId);

    // --- load the oriented full-res image (backend already applied EXIF) ---
    let blob;
    try {
      const res = await fetch("/api/images/" + id + "/full");
      if (!res.ok) throw new Error("HTTP " + res.status);
      blob = await res.blob();
    } catch (e) { toast("Could not load the full image: " + e.message, "err"); return; }
    let original;
    try { original = await blobToBitmap(blob); }
    catch (e) { toast("Could not read the image: " + e.message, "err"); return; }

    // 4K cap (configurable resolution — the source size is preserved up to this point)
    let W = original.width, H = original.height;
    if (Math.max(W, H) > MAX_PX) { const f = MAX_PX / Math.max(W, H); W = Math.round(W * f); H = Math.round(H * f); }

    const workC = makeCanvas(W, H);
    ctx2d(workC).drawImage(original, 0, 0, W, H);
    const originalC = makeCanvas(W, H);
    ctx2d(originalC).drawImage(original, 0, 0, W, H);
    const dispC = makeCanvas(W, H);
    const ovC = makeCanvas(W, H);

    const S = {
      tool: "crop", zoom: 1, panX: 0, panY: 0, fit: 0.3,
      brightness: 100, contrast: 100, saturation: 100, filter: "none", blur: 0,
      pendingFlip: null, pendingRotate: 0,
      crop: null, cropAspect: null,
      drawMode: "pen", color: "#ff3b30", brush: 10,
      redRadius: 24,
      // One editable sticker at a time; `base` is the canvas state before it was added,
      // so moving/resizing/rotating re-bakes base + sticker (no dual-track replay).
      sticker: null,
      needsAlpha: false,
      compare: false,
      history: [], hi: -1,
      _drag: null, _live: null, _space: false,
    };
    let closed = false;

    // ---- history: <= 10 compressed states, bitmaps closed on discard ----
    function closeBitmap(b) { try { b && b.close && b.close(); } catch (e) { /* */ } }
    function snapshot() {
      const type = S.needsAlpha ? "image/png" : "image/jpeg";
      const q = S.needsAlpha ? undefined : 0.92;
      return canvasToBlob(workC, type, q).then(blobToBitmap).catch(() => null);
    }
    async function commit() {
      const st = await snapshot();
      if (!st) { syncDisp(); return; }
      S.history = S.history.slice(0, S.hi + 1);
      S.history.push(st);
      while (S.history.length > HIST_MAX) closeBitmap(S.history.shift());
      S.hi = S.history.length - 1;
      state.histLen = S.history.length;
      syncDisp(); renderOverlay(); historyBtns();
    }
    async function restoreState(b) {
      if (!b) return;
      W = b.width; H = b.height;
      workC.width = W; workC.height = H;
      ctx2d(workC).drawImage(b, 0, 0);
      S.sticker = null; S.crop = null;
      resetAdjustPreview();
      relayout(); syncDisp(); renderOverlay();
    }
    async function undo() { if (S.hi > 0) { S.hi--; await restoreState(S.history[S.hi]); historyBtns(); } }
    async function redo() { if (S.hi < S.history.length - 1) { S.hi++; await restoreState(S.history[S.hi]); historyBtns(); } }

    // ---- display: dispC = clean copy of workC; live filter/preview is CSS only ----
    function liveFilter() {
      const f = [];
      if (S.brightness !== 100) f.push(`brightness(${S.brightness}%)`);
      if (S.contrast !== 100) f.push(`contrast(${S.contrast}%)`);
      if (S.saturation !== 100) f.push(`saturate(${S.saturation}%)`);
      const fl = FILTERS.find((z) => z[0] === S.filter);
      if (fl && fl[2] !== "none") f.push(fl[2]);
      if (S.blur > 0) f.push(`blur(${(S.blur / 10).toFixed(2)}px)`);
      return f.length ? f.join(" ") : "none";
    }
    function syncDisp() {
      if (dispC.width !== workC.width || dispC.height !== workC.height) {
        dispC.width = workC.width; dispC.height = workC.height;
      }
      const x = ctx2d(dispC);
      x.clearRect(0, 0, dispC.width, dispC.height);
      x.drawImage(workC, 0, 0);
      applyPreview();
    }
    function applyPreview() {
      wrap.style.filter = liveFilter();
      const z = S.zoom * S.fit;
      const r = stage.getBoundingClientRect();
      const panelW = panel ? (panel.offsetWidth || 0) : 0;
      const availW = Math.max(1, r.width - panelW);
      const ox = (availW - workC.width * z) / 2;
      const oy = (r.height - workC.height * z) / 2;
      let t = `translate(${(ox + S.panX).toFixed(2)}px,${(oy + S.panY).toFixed(2)}px) scale(${z.toFixed(4)})`;
      if (S.pendingFlip === "h") t += " scaleX(-1)";
      if (S.pendingFlip === "v") t += " scaleY(-1)";
      if (S.pendingRotate) t += ` rotate(${S.pendingRotate}deg)`;
      wrap.style.transform = t;
    }
    function resetAdjustPreview() {
      S.brightness = 100; S.contrast = 100; S.saturation = 100;
      S.filter = "none"; S.blur = 0; S.pendingFlip = null; S.pendingRotate = 0;
      applyPreview();
    }
    function bakeFilter() {
      const f = liveFilter();
      const sh = S.sharpenPending; S.sharpenPending = 0;
      if (f && f !== "none") {
        const w = workC.width, h = workC.height;
        const tmp = makeCanvas(w, h);
        const tx = ctx2d(tmp);
        tx.filter = f;
        tx.clearRect(0, 0, w, h);
        tx.drawImage(workC, 0, 0);
        workC.width = w; workC.height = h;
        ctx2d(workC).drawImage(tmp, 0, 0);
      }
      S.brightness = 100; S.contrast = 100; S.saturation = 100;
      S.filter = "none"; S.blur = 0;
      if (sh > 0) sharpen(ctx2d(workC), workC.width, workC.height, sh);
      syncDisp();
    }

    // ---- DOM ----
    const rootEl = document.createElement("div");
    rootEl.className = "pcur-ed";
    // Small preview used by the filter swatches (CSS var --pcur-thumb).
    try {
      const tw = 96;
      const th = Math.max(1, Math.round((H / Math.max(1, W)) * tw));
      const tc = makeCanvas(tw, th);
      ctx2d(tc).drawImage(workC, 0, 0, tw, th);
      rootEl.style.setProperty("--pcur-thumb", `url(${tc.toDataURL("image/jpeg", 0.7)})`);
    } catch (e) { /* preview is cosmetic */ }
    rootEl.setAttribute("role", "dialog");
    rootEl.setAttribute("aria-modal", "true");
    rootEl.setAttribute("aria-label", "Photo editor");

    const topbar = document.createElement("div");
    topbar.className = "pcur-top";
    const titleEl = document.createElement("div");
    titleEl.className = "pcur-title";
    titleEl.textContent = baseName;
    titleEl.title = baseName;
    const undoBtn = mkBtn(icon("undo") + "<span>Undo</span>", "pcur-hist");
    const redoBtn = mkBtn("<span>Redo</span>" + icon("right"), "pcur-hist");
    const compareBtn = mkBtn("Compare", ""); compareBtn.title = "Hold to compare with the original";
    const dlBtn = mkBtn(icon("down") + "<span>Download</span>", "btn sm secondary");
    const saveBtn = mkBtn(icon("check") + "<span>Save</span>", "btn sm primary");
    const closeBtn = mkBtn(icon("x"), "pcur-x");
    closeBtn.title = "Close (Esc)";
    closeBtn.setAttribute("aria-label", "Close editor");
    topbar.append(titleEl, grow(), undoBtn, redoBtn, compareBtn, dlBtn, saveBtn, closeBtn);

    const stage = document.createElement("div");
    stage.className = "pcur-stage";
    const wrap = document.createElement("div");
    wrap.className = "pcur-wrap";
    dispC.className = "pcur-disp";
    ovC.className = "pcur-ov";
    wrap.append(dispC, ovC);
    stage.appendChild(wrap);

    const toolbar = document.createElement("div");
    toolbar.className = "pcur-tools";
    toolbar.setAttribute("role", "toolbar");
    toolbar.setAttribute("aria-label", "Editor tools");
    TOOLS.forEach(([tid, label, ic], i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "pcur-tool";
      b.dataset.tool = tid;
      b.title = `${label} (${i + 1})`;
      b.innerHTML = icon(ic) + `<span>${esc(label)}</span>`;
      b.onclick = () => selectTool(tid);
      toolbar.appendChild(b);
    });

    const panel = document.createElement("div");
    panel.className = "pcur-panel";
    const panelBody = document.createElement("div");
    panelBody.className = "pcur-panelbody";
    panel.appendChild(panelBody);

    rootEl.append(topbar, stage, toolbar, panel);

    function grow() { const n = document.createElement("span"); n.className = "grow"; return n; }
    function mkBtn(html, cls) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = cls || "btn sm";
      b.innerHTML = html;
      return b;
    }

    // ---- geometry: fit scale + single transform on the wrapper ----
    function relayout() {
      const r = stage.getBoundingClientRect();
      const panelW = panel ? (panel.offsetWidth || 0) : 0;
      const availW = Math.max(1, r.width - panelW);
      if (availW > 4 && r.height > 4)
        S.fit = Math.min(availW / workC.width, r.height / workC.height) * 0.94;
      wrap.style.width = workC.width + "px";
      wrap.style.height = workC.height + "px";
      dispC.width = workC.width; dispC.height = workC.height;
      ovC.width = workC.width; ovC.height = workC.height;
      dispC.style.width = ovC.style.width = "100%";
      dispC.style.height = ovC.style.height = "100%";
      applyPreview();
    }
    // Strict screen -> canvas mapping (accounts for fit, zoom, pan, preview transform).
    function screenToCanvas(clientX, clientY) {
      const r = ovC.getBoundingClientRect();
      return {
        x: (clientX - r.left) / Math.max(1, r.width) * workC.width,
        y: (clientY - r.top) / Math.max(1, r.height) * workC.height,
      };
    }
    function canvasToScreen(cx, cy) {
      const r = ovC.getBoundingClientRect();
      return { x: r.left + cx / workC.width * r.width, y: r.top + cy / workC.height * r.height };
    }

    // ---- overlay (crop box, live stroke, sticker handles) ----
    function strokePath(x, st, Wc, Hc) {
      const pts = st.pts;
      if (!pts.length) return;
      const lw = Math.max(1, st.size * (Wc / 1000));
      x.save();
      x.lineCap = "round"; x.lineJoin = "round"; x.lineWidth = lw;
      if (st.mode === "highlighter") { x.globalAlpha = 0.45; x.globalCompositeOperation = "multiply"; }
      else if (st.mode === "eraser") x.globalCompositeOperation = "destination-out";
      x.strokeStyle = st.color; x.fillStyle = st.color;
      x.beginPath();
      x.moveTo(pts[0].x * Wc, pts[0].y * Hc);
      for (let i = 1; i < pts.length; i++) x.lineTo(pts[i].x * Wc, pts[i].y * Hc);
      if (pts.length === 1) { x.arc(pts[0].x * Wc, pts[0].y * Hc, lw / 2, 0, Math.PI * 2); x.fill(); }
      else x.stroke();
      x.restore();
    }
    function drawSticker(x, sk, Wc, Hc) {
      const img = sk.bitmap;
      if (!img) return;
      const bw = img.width || 1, bh = img.height || 1;
      const w = sk.w * Wc, h = w * (bh / bw);
      const cx = sk.x * Wc, cy = sk.y * Hc;
      x.save();
      x.translate(cx, cy);
      if (sk.rot) x.rotate(sk.rot * Math.PI / 180);
      if (sk.kind === "qr") {
        const pad = w * 0.09;
        x.fillStyle = "#fff";
        roundRectPath(x, -w / 2 - pad, -h / 2 - pad, w + pad * 2, h + pad * 2, w * 0.05);
        x.fill();
      }
      x.drawImage(img, -w / 2, -h / 2, w, h);
      x.restore();
    }
    function bakeSticker() {
      const sk = S.sticker;
      const x = ctx2d(workC);
      x.clearRect(0, 0, workC.width, workC.height);
      if (sk && sk.base) {
        x.drawImage(sk.base, 0, 0, workC.width, workC.height);
        drawSticker(x, sk, workC.width, workC.height);
      }
    }
    function renderOverlay() {
      const x = ctx2d(ovC);
      x.clearRect(0, 0, ovC.width, ovC.height);
      const Wc = workC.width, Hc = workC.height;
      if (S.tool === "stickers" && S.sticker) {
        const sk = S.sticker;
        const w = sk.w * Wc, h = w * ((sk.bitmap.height || 1) / (sk.bitmap.width || 1));
        const cx = sk.x * Wc, cy = sk.y * Hc;
        x.save();
        x.translate(cx, cy);
        if (sk.rot) x.rotate(sk.rot * Math.PI / 180);
        x.strokeStyle = "var(--accent, #4f8cff)";
        x.strokeStyle = "#4f8cff";
        x.lineWidth = Math.max(1, Wc / 500);
        x.setLineDash([w / 16, w / 16]);
        x.strokeRect(-w / 2, -h / 2, w, h);
        x.setLineDash([]);
        const hr = Math.max(6, w / 12);
        x.fillStyle = "#fff"; x.strokeStyle = "#4f8cff";
        x.beginPath(); x.arc(w / 2, -h / 2, hr, 0, Math.PI * 2); x.fill(); x.stroke();   // resize (corner)
        x.beginPath(); x.arc(0, -h / 2 - w / 5, hr, 0, Math.PI * 2); x.fill(); x.stroke();  // rotate (top)
        x.restore();
      }
      if (S.tool === "crop" && S.crop) {
        const c = S.crop;
        x.save();
        x.fillStyle = "rgba(0,0,0,.55)";
        x.beginPath();
        x.rect(0, 0, Wc, Hc);
        x.rect(c.x, c.y, c.w, c.h);
        x.fill("evenodd");
        x.strokeStyle = "#fff";
        x.lineWidth = Math.max(1, Wc / 600);
        x.strokeRect(c.x, c.y, c.w, c.h);
        x.restore();
      }
      if (S._live) strokePath(x, S._live, Wc, Hc);
    }

    // ---- pointer interactions ----
    function norm(p) { return { x: p.x / workC.width, y: p.y / workC.height }; }
    function stickerMetrics(sk) {
      const bw = (sk.bitmap.width || 1), bh = (sk.bitmap.height || 1);
      const w = sk.w * workC.width, h = w * (bh / bw);
      return { cx: sk.x * workC.width, cy: sk.y * workC.height, w, h };
    }
    function hitStickerHandle(p) {
      const sk = S.sticker;
      if (!sk || S.tool !== "stickers") return null;
      const m = stickerMetrics(sk);
      let lx = p.x - m.cx;
      let ly = p.y - m.cy;
      if (sk.rot) {
        const rad = -sk.rot * Math.PI / 180;
        const cos = Math.cos(rad), sin = Math.sin(rad);
        const rx = lx * cos - ly * sin;
        const ry = lx * sin + ly * cos;
        lx = rx; ly = ry;
      }
      const rad = Math.max(14, workC.width / 40);
      const d1 = Math.hypot(lx - m.w / 2, ly - (-m.h / 2));
      if (d1 < rad) return "resize";
      const d2 = Math.hypot(lx, ly - (-m.h / 2 - m.w / 5));
      if (d2 < rad) return "rotate";
      if (lx > -m.w / 2 && lx < m.w / 2 && ly > -m.h / 2 && ly < m.h / 2) return "move";
      return null;
    }
    function cropHandle(p) {
      const c = S.crop;
      if (!c) return null;
      const t = Math.max(12, workC.width / 50);
      const pts = {
        nw: [c.x, c.y], ne: [c.x + c.w, c.y], sw: [c.x, c.y + c.h], se: [c.x + c.w, c.y + c.h],
        n: [c.x + c.w / 2, c.y], s: [c.x + c.w / 2, c.y + c.h],
        e: [c.x + c.w, c.y + c.h / 2], w: [c.x, c.y + c.h / 2],
      };
      for (const k in pts)
        if (Math.abs(p.x - pts[k][0]) < t && Math.abs(p.y - pts[k][1]) < t) return k;
      if (p.x > c.x && p.x < c.x + c.w && p.y > c.y && p.y < c.y + c.h) return "inside";
      return null;
    }
    function resizeCrop(p, h) {
      const c = S.crop;
      const right = c.x + c.w, bottom = c.y + c.h;
      if (h === "e") c.w = clamp(p.x - c.x, 8, workC.width - c.x);
      if (h === "w") { c.x = clamp(p.x, 0, right - 8); c.w = right - c.x; }
      if (h === "s") c.h = clamp(p.y - c.y, 8, workC.height - c.y);
      if (h === "n") { c.y = clamp(p.y, 0, bottom - 8); c.h = bottom - c.y; }
      if (h === "nw") { c.x = clamp(p.x, 0, right - 8); c.y = clamp(p.y, 0, bottom - 8); c.w = right - c.x; c.h = bottom - c.y; }
      if (h === "ne") { c.w = clamp(p.x - c.x, 8, workC.width - c.x); c.y = clamp(p.y, 0, bottom - 8); c.h = bottom - c.y; }
      if (h === "sw") { c.x = clamp(p.x, 0, right - 8); c.w = right - c.x; c.h = clamp(p.y - c.y, 8, workC.height - c.y); }
      if (h === "se") { c.w = clamp(p.x - c.x, 8, workC.width - c.x); c.h = clamp(p.y - c.y, 8, workC.height - c.y); }
    }
    function applyAspect() {
      const c = S.crop;
      if (!c || !S.cropAspect) return;
      const ar = S.cropAspect;
      const cx = c.x + c.w / 2, cy = c.y + c.h / 2;
      let w = c.w, h = w / ar;
      if (h > c.h * 1.6 || h < c.h * 0.62) { h = c.h; w = h * ar; }
      c.w = w; c.h = h;
      c.x = clamp(cx - w / 2, 0, workC.width - w);
      c.y = clamp(cy - h / 2, 0, workC.height - h);
      c.w = Math.min(c.w, workC.width - c.x);
      c.h = Math.min(c.h, workC.height - c.y);
    }

    function onPointerDown(ev) {
      if (closed) return;
      ev.preventDefault();
      if (S._space || ev.button === 2 || ev.button === 1) {
        S._drag = { kind: "pan", sx: ev.clientX, sy: ev.clientY, px: S.panX, py: S.panY };
        stage.style.cursor = "grabbing";
        return;
      }
      if (ev.button !== 0) return;
      const p = screenToCanvas(ev.clientX, ev.clientY);
      if (S.tool === "crop") {
        const h = cropHandle(p);
        if (h === "inside") S._drag = { kind: "cropmove", ox: p.x - S.crop.x, oy: p.y - S.crop.y };
        else if (h) S._drag = { kind: "crophandle", h };
        else {
          S.crop = { x: p.x, y: p.y, w: 0, h: 0, sx: p.x, sy: p.y };
          S._drag = { kind: "cropnew" };
        }
        renderOverlay();
      } else if (S.tool === "redeye") {
        redEye(ctx2d(workC), workC.width, workC.height, p.x, p.y, S.redRadius);
        syncDisp();
        commit();
      } else if (S.tool === "draw") {
        S._live = { mode: S.drawMode, color: S.color, size: S.brush, pts: [norm(p)] };
        renderOverlay();
      } else if (S.tool === "stickers" && S.sticker) {
        const h = hitStickerHandle(p);
        const m = stickerMetrics(S.sticker);
        if (h === "resize") {
          S._drag = { kind: "resize", initialDist: Math.hypot(p.x - m.cx, p.y - m.cy), initialW: S.sticker.w };
        } else if (h === "rotate") {
          S._drag = { kind: "rotate", initialAngle: Math.atan2(p.y - m.cy, p.x - m.cx), initialRot: S.sticker.rot || 0 };
        } else if (h === "move") {
          S._drag = { kind: "move", ox: p.x - m.cx, oy: p.y - m.cy };
        } else {
          S.sticker = null;
          renderOverlay();
        }
      }
    }
    function onPointerMove(ev) {
      if (closed) return;
      const d = S._drag;
      if (d && d.kind === "pan") {
        S.panX = d.px + (ev.clientX - d.sx);
        S.panY = d.py + (ev.clientY - d.sy);
        applyPreview();
        return;
      }
      if (d) {
        const p = screenToCanvas(ev.clientX, ev.clientY);
        if (d.kind === "cropnew") {
          const c = S.crop;
          c.x = Math.min(c.sx, p.x); c.y = Math.min(c.sy, p.y);
          c.w = Math.abs(p.x - c.sx); c.h = Math.abs(p.y - c.sy);
          applyAspect();
        } else if (d.kind === "cropmove") {
          S.crop.x = clamp(p.x - d.ox, 0, workC.width - S.crop.w);
          S.crop.y = clamp(p.y - d.oy, 0, workC.height - S.crop.h);
        } else if (d.kind === "crophandle") {
          resizeCrop(p, d.h);
          applyAspect();
        } else if (d.kind === "move" && S.sticker) {
          S.sticker.x = clamp((p.x - d.ox) / workC.width, 0.02, 0.98);
          S.sticker.y = clamp((p.y - d.oy) / workC.height, 0.02, 0.98);
          bakeSticker(); syncDisp(); renderOverlay();
        } else if (d.kind === "resize" && S.sticker) {
          const m = stickerMetrics(S.sticker);
          const curDist = Math.hypot(p.x - m.cx, p.y - m.cy);
          S.sticker.w = clamp(d.initialW * (curDist / Math.max(1, d.initialDist)), 0.02, 1.3);
          bakeSticker(); syncDisp(); renderOverlay();
        } else if (d.kind === "rotate" && S.sticker) {
          const m = stickerMetrics(S.sticker);
          const curAngle = Math.atan2(p.y - m.cy, p.x - m.cx);
          S.sticker.rot = (d.initialRot + (curAngle - d.initialAngle) * 180 / Math.PI) % 360;
          bakeSticker(); syncDisp(); renderOverlay();
        }
        return;
      }
      if (S._live) {
        const p = screenToCanvas(ev.clientX, ev.clientY);
        S._live.pts.push(norm(p));
        renderOverlay();
      }
    }
    function onPointerUp() {
      if (closed) return;
      const d = S._drag;
      S._drag = null;
      stage.style.cursor = "";
      if (S._live) {
        const st = S._live;
        S._live = null;
        if (st.pts.length) {
          strokePath(ctx2d(workC), st, workC.width, workC.height);
          syncDisp();
          commit();
        }
        return;
      }
      if (d && (d.kind === "move" || d.kind === "resize" || d.kind === "rotate")) commit();
    }
    function onWheel(ev) {
      if (closed) return;
      ev.preventDefault();
      const f = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
      S.zoom = clamp(S.zoom * f, 0.2, 12);
      applyPreview();
    }

    // ---- tool panels ----
    function selectTool(tid) {
      S.tool = tid;
      resetAdjustPreview();
      if (tid !== "crop") S.crop = null;
      $$(".pcur-tool", toolbar).forEach((b) => b.classList.toggle("active", b.dataset.tool === tid));
      buildPanel();
      renderOverlay();
      relayout();
    }
    function historyBtns() {
      undoBtn.disabled = S.hi <= 0;
      redoBtn.disabled = S.hi >= S.history.length - 1;
    }
    function pEl(tag, cls, html) {
      const n = document.createElement(tag);
      if (cls) n.className = cls;
      if (html != null) n.innerHTML = html;
      return n;
    }
    function pRow(label, node) {
      const r = pEl("div", "pcur-row");
      if (label) r.appendChild(pEl("label", "pcur-lbl", esc(label)));
      if (node) r.appendChild(node);
      return r;
    }
    function pSlider(label, min, max, value, onInput, onCommit, fmt) {
      const box = pEl("div", "pcur-slider");
      const head = pEl("div", "pcur-shead");
      head.appendChild(pEl("label", "pcur-lbl", esc(label)));
      const val = pEl("span", "pcur-val", fmt ? fmt(value) : String(value));
      head.appendChild(val);
      const inp = document.createElement("input");
      inp.type = "range"; inp.min = min; inp.max = max; inp.value = value;
      inp.step = (max - min) / 100;
      inp.addEventListener("input", () => {
        const v = Number(inp.value);
        val.textContent = fmt ? fmt(v) : String(v);
        onInput && onInput(v);
      });
      inp.addEventListener("change", () => onCommit && onCommit(Number(inp.value)));
      box.append(head, inp);
      return box;
    }
    function pBtn(label, cls, fn, title) {
      const b = mkBtn(label, "btn sm " + (cls || ""));
      b.onclick = fn;
      if (title) b.title = title;
      return b;
    }
    function chipRow(items, active, onPick) {
      const r = pEl("div", "pcur-chips");
      items.forEach(([val, label], i) => {
        const b = pEl("button", "chip" + (active === val ? " on" : ""), esc(label));
        b.type = "button";
        b.onclick = () => {
          onPick(val);
          $$(".chip", r).forEach((c, j) => c.classList.toggle("on", j === i));
        };
        r.appendChild(b);
      });
      return r;
    }
    function busy(b, on) {
      b.disabled = on;
      b.classList.toggle("busy", on);
    }

    async function buildPanel() {
      panelBody.innerHTML = "";
      const add = (n) => panelBody.appendChild(n);
      const T = S.tool;

      if (T === "crop") {
        add(chipRow([["free", "Free"], ["1:1", "1:1"], ["4:3", "4:3"], ["16:9", "16:9"], ["3:2", "3:2"], ["9:16", "9:16"]],
          S.cropAspect || "free", (v) => { S.cropAspect = v === "free" ? null : Number(v.split(":")[0]) / Number(v.split(":")[1]); if (S.crop) { applyAspect(); renderOverlay(); } }));
        add(pEl("p", "pcur-hint", "Drag on the photo to choose the crop area. Drag the corners to resize it, the middle to move it."));
        const apply = pBtn(icon("check") + "<span>Apply crop</span>", "primary", async () => {
          const c = S.crop;
          if (!c || c.w < 2 || c.h < 2) { toast("Draw a crop area first", "err"); return; }
          const nc = makeCanvas(c.w, c.h);
          ctx2d(nc).drawImage(workC, c.x, c.y, c.w, c.h, 0, 0, c.w, c.h);
          let nw = nc.width, nh = nc.height;
          if (Math.max(nw, nh) > MAX_PX) { const f = MAX_PX / Math.max(nw, nh); nw = Math.round(nw * f); nh = Math.round(nh * f); }
          workC.width = nw; workC.height = nh;
          ctx2d(workC).drawImage(nc, 0, 0, nw, nh);
          S.crop = null;
          relayout();
          await commit();
          toast("Crop applied", "ok");
        });
        const reset = pBtn("Reset", "ghost", () => { S.crop = null; renderOverlay(); });
        add(pEl("div", "pcur-btns", ""));
        panelBody.lastChild.append(apply, reset);
      }

      else if (T === "transform") {
        const r90 = pEl("div", "pcur-btns");
        r90.append(
          pBtn("⟲ 90°", "", () => rotate90(-90), "Rotate counter-clockwise"),
          pBtn("90° ⟳", "", () => rotate90(90), "Rotate clockwise"),
          pBtn("Flip ↔", "", () => { bakeFlip("h"); toast("Flipped horizontally", "ok"); }, "Flip horizontal"),
          pBtn("Flip ↕", "", () => { bakeFlip("v"); toast("Flipped vertically", "ok"); }, "Flip vertical"),
        );
        add(r90);
        add(pSlider("Fine rotate", -45, 45, 0,
          (v) => { S.pendingRotate = v; applyPreview(); },
          (v) => {
            if (v) rotateBy(v);
            else applyPreview();
            S.pendingRotate = 0;
          }, (v) => Math.round(v) + "°"));
        add(pEl("p", "pcur-hint", "90° and flips bake instantly. Fine rotation previews live and bakes when you release the slider."));
      }

      else if (T === "adjust") {
        const sB = pSlider("Brightness", 50, 150, S.brightness, (v) => { S.brightness = v; applyPreview(); }, null, (v) => v + "%");
        const sC = pSlider("Contrast", 50, 150, S.contrast, (v) => { S.contrast = v; applyPreview(); }, null, (v) => v + "%");
        const sS = pSlider("Saturation", 0, 200, S.saturation, (v) => { S.saturation = v; applyPreview(); }, null, (v) => v + "%");
        add(sB); add(sC); add(sS);
        const resetSliders = () => {
          S.brightness = S.contrast = S.saturation = 100;
          applyPreview();
          for (const box of [sB, sC, sS]) {
            const inp = box.querySelector("input");
            const val = box.querySelector(".pcur-val");
            if (inp) inp.value = "100";
            if (val) val.textContent = "100%";
          }
        };
        const apply = pBtn(icon("check") + "<span>Apply adjustments</span>", "primary", async () => {
          if (S.brightness === 100 && S.contrast === 100 && S.saturation === 100) return;
          bakeFilter();
          await commit();
          resetSliders();
          toast("Adjustments applied", "ok");
        });
        const reset = pBtn("Reset", "ghost", resetSliders);
        const btnBox = pEl("div", "pcur-btns", "");
        btnBox.append(apply, reset);
        add(btnBox);
        add(pEl("p", "pcur-hint", "Adjust sliders preview live. Click 'Apply adjustments' to bake into the photo."));
      }

      else if (T === "filters") {
        const grid = pEl("div", "pcur-filters");
        FILTERS.forEach(([fid, label, css]) => {
          const b = pEl("button", "pcur-fil" + (S.filter === fid ? " on" : ""));
          b.type = "button";
          b.title = label;
          b.innerHTML = `<span class="pcur-fprev" style="background-image:var(--pcur-thumb,none);filter:${css}"></span><span>${esc(label)}</span>`;
          b.onclick = () => {
            S.filter = fid;
            applyPreview();
            $$(".pcur-fil", grid).forEach((z) => z.classList.remove("on"));
            b.classList.add("on");
          };
          grid.appendChild(b);
        });
        add(grid);
        const apply = pBtn(icon("check") + "<span>Apply filter</span>", "primary", () => { bakeFilter(); commit(); toast("Filter applied", "ok"); }, "Bake the selected filter into the photo");
        add(pEl("div", "pcur-btns", "").appendChild(apply));
      }

      else if (T === "remaster") {
        add(pEl("p", "pcur-hint", "Auto-levels with a 1–99% percentile stretch, a +15% saturation boost and a mild sharpen. Safe on flat and solid-color images."));
        const b = pBtn("Remaster", "primary", async () => {
          busy(b, true);
          try {
            await new Promise((r) => setTimeout(r, 30));   // let the busy state paint first
            remaster(ctx2d(workC), workC.width, workC.height);
            syncDisp();
            await commit();
            toast("Remaster applied", "ok");
          } catch (e) { toast("Remaster failed: " + e.message, "err"); }
          busy(b, false);
        });
        add(pEl("div", "pcur-btns", "").appendChild(b));
      }

      else if (T === "blur") {
        const sBlur = pSlider("Blur", 0, 40, S.blur, (v) => { S.blur = v; applyPreview(); }, null, (v) => v + "px");
        const sSharp = pSlider("Sharpen", 0, 100, Math.round((S.sharpenPending || 0) * 100), (v) => { S.sharpenPending = v / 100; }, null, (v) => v + "%");
        add(sBlur); add(sSharp);
        const resetBlur = () => {
          S.blur = 0; S.sharpenPending = 0;
          applyPreview();
          const inps = [sBlur.querySelector("input"), sSharp.querySelector("input")];
          const vals = [sBlur.querySelector(".pcur-val"), sSharp.querySelector(".pcur-val")];
          if (inps[0]) inps[0].value = "0";
          if (vals[0]) vals[0].textContent = "0px";
          if (inps[1]) inps[1].value = "0";
          if (vals[1]) vals[1].textContent = "0%";
        };
        const apply = pBtn(icon("check") + "<span>Apply blur / sharpen</span>", "primary", async () => {
          if (S.blur <= 0 && (!S.sharpenPending || S.sharpenPending <= 0)) return;
          bakeFilter();
          await commit();
          resetBlur();
          toast("Blur / sharpen applied", "ok");
        });
        const reset = pBtn("Reset", "ghost", resetBlur);
        const btnBox = pEl("div", "pcur-btns", "");
        btnBox.append(apply, reset);
        add(btnBox);
        add(pEl("p", "pcur-hint", "Blur previews with the GPU. Click 'Apply blur / sharpen' to bake into the photo."));
      }

      else if (T === "redeye") {
        add(pSlider("Radius", 6, 80, S.redRadius, (v) => { S.redRadius = v; }, null, (v) => v + "px"));
        add(pEl("p", "pcur-hint", "Click directly on a red pupil. Reddish pixels inside the radius are darkened."));
      }

      else if (T === "bg") {
        const opts = { outline: true, despeckle: true, trim: false };
        const cbox = (key, label) => {
          const l = pEl("label", "pcur-chk");
          const i = document.createElement("input");
          i.type = "checkbox"; i.checked = opts[key];
          i.addEventListener("change", () => { opts[key] = i.checked; });
          l.append(i, pEl("span", "", esc(label)));
          return l;
        };
        add(cbox("outline", "White outline (cartoonize)"));
        add(cbox("despeckle", "Clean edges (despeckle)"));
        add(cbox("trim", "Trim to the subject"));
        const b = pBtn("Remove background", "primary", async () => {
          busy(b, true);
          try {
            const b2 = await canvasToBlob(workC, "image/png");
            const fd = new FormData();
            fd.append("file", b2, "cut.png");
            const res = await fetch("/api/editor/remove-bg", { method: "POST", body: fd });
            if (!res.ok) throw new Error(((await res.json().catch(() => ({}))).detail) || "HTTP " + res.status);
            const j = await res.json();
            const img = await loadImage(j.image);
            let outC = makeCanvas(img.width, img.height);
            ctx2d(outC).drawImage(img, 0, 0);
            if (opts.despeckle) despeckle(ctx2d(outC), outC.width, outC.height);
            if (opts.outline) outC = cartoonOutline(outC, Math.max(3, Math.round(outC.width / 250)));
            if (opts.trim) outC = trimTransparency(outC);
            if (Math.max(outC.width, outC.height) > MAX_PX) {
              const f = MAX_PX / Math.max(outC.width, outC.height);
              const t = makeCanvas(outC.width * f, outC.height * f);
              ctx2d(t).drawImage(outC, 0, 0, t.width, t.height);
              outC = t;
            }
            workC.width = outC.width; workC.height = outC.height;
            ctx2d(workC).drawImage(outC, 0, 0);
            S.needsAlpha = true;
            relayout();
            await commit();
            toast("Background removed", "ok");
          } catch (e) { toast("Background removal failed: " + e.message, "err"); }
          busy(b, false);
        });
        add(pEl("div", "pcur-btns", "").appendChild(b));
        add(pEl("p", "pcur-hint", "Runs on the server (rembg when available, GrabCut offline fallback). The result keeps transparency."));
      }

      else if (T === "draw") {
        add(chipRow([["pen", "Pen"], ["highlighter", "Highlighter"], ["eraser", "Eraser"]], S.drawMode, (v) => { S.drawMode = v; }));
        const cr = pEl("div", "pcur-row");
        cr.appendChild(pEl("label", "pcur-lbl", "Color"));
        const ci = document.createElement("input");
        ci.type = "color"; ci.value = S.color;
        ci.addEventListener("input", () => { S.color = ci.value; });
        cr.appendChild(ci);
        add(cr);
        add(pSlider("Brush size", 1, 60, S.brush, (v) => { S.brush = v; }, null, (v) => v + "px"));
        add(pEl("p", "pcur-hint", "Draw straight onto the photo. The highlighter blends with multiply; the eraser clears back to the photo."));
      }

      else if (T === "stickers") {
        let custom = await loadCustomStickers();
        const grid = pEl("div", "pcur-stickers");
        const showCat = async (cid) => {
          grid.innerHTML = "";
          if (cid === "custom") {
            if (!custom.length) grid.appendChild(pEl("p", "pcur-hint", "No custom stickers yet — upload one below."));
            custom.forEach((url) => {
              const b = pEl("button", "pcur-stk");
              b.type = "button";
              b.innerHTML = `<img src="${esc(url)}" alt="Custom sticker">`;
              b.onclick = async () => placeSticker(await dataUrlToBitmap(url), "sticker");
              grid.appendChild(b);
            });
            const up = pBtn(icon("plus") + "<span>Upload</span>", "ghost", () => {
              const inp = document.createElement("input");
              inp.type = "file"; inp.accept = "image/*";
              inp.onchange = () => {
                const f = inp.files && inp.files[0];
                if (!f) return;
                const r = new FileReader();
                r.onload = () => { custom.unshift(String(r.result)); saveCustomStickers(custom); showCat("custom"); toast("Sticker saved", "ok"); };
                r.readAsDataURL(f);
              };
              inp.click();
            });
            grid.appendChild(up);
          } else {
            const cat = STICKER_CATS.find((c) => c.id === cid);
            for (const ch of cat.items) {
              const b = pEl("button", "pcur-stk", esc(ch));
              b.type = "button";
              b.onclick = async () => placeSticker(await emojiToBitmap(ch), "sticker");
              grid.appendChild(b);
            }
          }
        };
        const tabs = chipRow(STICKER_CATS.map((c) => [c.id, c.label]), "fun", (v) => showCat(v));
        add(tabs);
        add(grid);
        showCat("fun");
        add(pEl("p", "pcur-hint", "Click a sticker to drop it in the middle. Then drag to move, the corner handle to resize, the top handle to rotate."));
      }

      else if (T === "idcard") {
        S.idc = S.idc || { dpi: 150, front: null, back: null, fscale: 1, bscale: 1 };
        add(chipRow([[150, "150 DPI (1240×1754)"], [300, "300 DPI (2480×3508)"]], S.idc.dpi,
          (v) => { S.idc.dpi = v; renderID(); }));
        const slot = (which, label) => {
          const r = pEl("div", "pcur-row");
          r.appendChild(pEl("label", "pcur-lbl", esc(label)));
          const r2 = pEl("div", "pcur-btns");
          r2.append(
            pBtn(which === "front" ? "Front photo…" : "Back photo…", "ghost", () => pick(which)),
            pBtn("Use current", "ghost", () => {
              const tmp = makeCanvas(workC.width, workC.height);
              ctx2d(tmp).drawImage(workC, 0, 0);
              createImageBitmap(tmp).then((b) => { (which === "front" ? (S.idc.front = b) : (S.idc.back = b)); renderID(); }).catch(() => toast("Could not capture the current image", "err"));
            }, "Use the current edited photo"),
          );
          r.appendChild(r2);
          return r;
        };
        function pick(which) {
          const inp = document.createElement("input");
          inp.type = "file"; inp.accept = "image/*";
          inp.onchange = () => {
            const f = inp.files && inp.files[0];
            if (!f) return;
            const url = URL.createObjectURL(f);
            loadImage(url).then((img) => {
              createImageBitmap(img).then((b) => { (which === "front" ? (S.idc.front = b) : (S.idc.back = b)); renderID(); }).catch(() => toast("Could not read that image", "err"));
            }).catch((e) => toast(e.message, "err"));
          };
          inp.click();
        }
        add(slot("front", "Front"));
        add(slot("back", "Back"));
        add(pSlider("Front scale", 50, 140, Math.round(S.idc.fscale * 100), (v) => { S.idc.fscale = v / 100; renderID(); }, null, (v) => v + "%"));
        add(pSlider("Back scale", 50, 140, Math.round(S.idc.bscale * 100), (v) => { S.idc.bscale = v / 100; renderID(); }, null, (v) => v + "%"));
        const pc = makeCanvas(465, 657);
        pc.className = "pcur-idc";
        const pw = pEl("div", "pcur-idprev");
        pw.appendChild(pc);
        add(pw);
        const apply = pBtn(icon("check") + "<span>Apply ID card</span>", "primary", async () => {
          if (!S.idc.front && !S.idc.back) { toast("Choose a front and/or back photo first", "err"); return; }
          const c = buildIDCard();
          if (!c) return;
          let cw = c.width, ch = c.height;
          if (Math.max(cw, ch) > MAX_PX) { const f = MAX_PX / Math.max(cw, ch); cw = Math.round(cw * f); ch = Math.round(ch * f); }
          workC.width = cw; workC.height = ch;
          ctx2d(workC).drawImage(c, 0, 0, cw, ch);
          relayout();
          await commit();
          toast("ID card layout applied", "ok");
        });
        add(pEl("div", "pcur-btns", "").appendChild(apply));

        function idDims() {
          return { w: Math.round(1240 * S.idc.dpi / 150), h: Math.round(1754 * S.idc.dpi / 150) };
        }
        function drawID(x, Wc, Hc) {
          const margin = Wc * 0.032, gap = Hc * 0.02;
          const availW = Wc - margin * 2;
          const availH = (Hc - margin * 2 - gap) / 2;
          x.fillStyle = "#fff";
          x.fillRect(0, 0, Wc, Hc);
          const drawHalf = (img, y0, scale) => {
            if (!img) return;
            const d = fitArea(img.width, img.height, availW * scale, availH * scale);
            x.drawImage(img, margin + (availW - d.w) / 2, y0 + (availH - d.h) / 2, d.w, d.h);
          };
          drawHalf(S.idc.front, margin, S.idc.fscale);
          drawHalf(S.idc.back, margin + availH + gap, S.idc.bscale);
          x.fillStyle = "#666";
          x.font = `bold ${Math.round(Wc * 0.016)}px Arial`;
          x.textAlign = "center";
          x.fillText("FRONT", Wc / 2, margin + availH + gap / 2 + Wc * 0.006);
          x.fillText("BACK", Wc / 2, Hc - margin + Wc * 0.02);
          x.strokeStyle = "#e0e0e0";
          x.lineWidth = 1;
          x.strokeRect(margin, margin, availW, availH);
          x.strokeRect(margin, margin + availH + gap, availW, availH);
        }
        function renderID() {
          const d = idDims();
          pc.width = Math.round(d.w / 2.67); pc.height = Math.round(d.h / 2.67);
          drawID(ctx2d(pc), pc.width, pc.height);
        }
        function buildIDCard() {
          const d = idDims();
          const c = makeCanvas(d.w, d.h);
          drawID(ctx2d(c), d.w, d.h);
          return c;
        }
        renderID();
      }

      else if (T === "qr") {
        S.qr = S.qr || { type: "url", fields: {} };
        add(chipRow([["url", "URL"], ["text", "Text"], ["phone", "Phone"], ["email", "Email"], ["sms", "SMS"], ["wifi", "Wi-Fi"], ["vcard", "vCard"], ["geo", "Geo"]],
          S.qr.type, (v) => { S.qr.type = v; S.qr.fields = {}; buildFields(); }));
        const fields = pEl("div", "pcur-qrf");
        add(fields);
        function qrText() {
          const f = S.qr.fields;
          switch (S.qr.type) {
            case "url": return f.value || "";
            case "text": return f.value || "";
            case "phone": return f.value ? "tel:" + f.value : "";
            case "email": return f.value ? `mailto:${f.value}${f.subject ? "?subject=" + encodeURIComponent(f.subject) : ""}` : "";
            case "sms": return f.phone ? `SMSTO:${f.phone}:${f.message || ""}` : "";
            case "wifi": return f.ssid ? `WIFI:T:${f.enc || "WPA"};S:${f.ssid};P:${f.password || ""};;` : "";
            case "vcard": return f.name ? `BEGIN:VCARD\nVERSION:3.0\nFN:${f.name}\nTEL:${f.phone || ""}\nEMAIL:${f.email || ""}\nEND:VCARD` : "";
            case "geo": return f.lat || f.long ? `geo:${f.lat || 0},${f.long || 0}` : "";
          }
          return "";
        }
        const qcv = makeCanvas(200, 200);
        qcv.className = "pcur-qrcv";
        const pw = pEl("div", "pcur-qrprev");
        pw.appendChild(qcv);
        add(pw);
        function paintQR() {
          const x = ctx2d(qcv);
          const txt = qrText();
          x.clearRect(0, 0, 200, 200);
          if (!txt) {
            x.fillStyle = "rgba(128,128,128,.25)";
            x.fillRect(0, 0, 200, 200);
            x.fillStyle = "#888"; x.font = "13px sans-serif"; x.textAlign = "center";
            x.fillText("Type details to generate", 100, 104);
            return;
          }
          try {
            if (!root.PCUR_QR) throw new Error("QR module not loaded");
            const m = root.PCUR_QR.encode(txt);
            qrToBitmap(m).then((bmp) => {
              const x2 = ctx2d(qcv);
              x2.clearRect(0, 0, 200, 200);
              x2.fillStyle = "#fff"; x2.fillRect(0, 0, 200, 200);
              const s = Math.min(196 / bmp.width, 196 / bmp.height);
              x2.drawImage(bmp, (200 - bmp.width * s) / 2, (200 - bmp.height * s) / 2, bmp.width * s, bmp.height * s);
            });
          } catch (e) {
            const x2 = ctx2d(qcv);
            x2.fillStyle = "rgba(128,128,128,.25)"; x2.fillRect(0, 0, 200, 200);
            x2.fillStyle = "#a33"; x2.font = "12px sans-serif"; x2.textAlign = "center";
            x2.fillText("Too much data for QR v10", 100, 104);
          }
        }
        const DEFS = {
          url: [["value", "URL"]], text: [["value", "Text"]], phone: [["value", "Number"]],
          email: [["value", "Address"], ["subject", "Subject"]],
          sms: [["phone", "Number"], ["message", "Message"]],
          wifi: [["ssid", "Network name"], ["password", "Password"], ["enc", "Encryption (WPA/WEPA/nopass)"]],
          vcard: [["name", "Full name"], ["phone", "Phone"], ["email", "Email"]],
          geo: [["lat", "Latitude"], ["long", "Longitude"]],
        };
        function buildFields() {
          fields.innerHTML = "";
          for (const [k, lab] of DEFS[S.qr.type]) {
            const r = pEl("div", "pcur-row");
            r.appendChild(pEl("label", "pcur-lbl", esc(lab)));
            const inp = document.createElement("input");
            inp.type = "text";
            inp.value = S.qr.fields[k] || "";
            inp.placeholder = lab;
            inp.addEventListener("input", () => { S.qr.fields[k] = inp.value; paintQR(); });
            r.appendChild(inp);
            fields.appendChild(r);
          }
          paintQR();
        }
        buildFields();
        const addBtn = pBtn(icon("plus") + "<span>Add to photo</span>", "primary", async () => {
          const txt = qrText();
          if (!txt) { toast("Fill in the QR details first", "err"); return; }
          if (!root.PCUR_QR) { toast("QR module not loaded", "err"); return; }
          let m;
          try { m = root.PCUR_QR.encode(txt); } catch (e) { toast("Too much data for a QR code", "err"); return; }
          const bmp = await qrToBitmap(m);
          placeSticker(bmp, "qr");
          selectTool("stickers");
          toast("QR added — drag it into place", "ok");
        });
        add(pEl("div", "pcur-btns", "").appendChild(addBtn));
      }

      else if (T === "ocr") {
        const b = pBtn("Extract text", "primary", async () => {
          busy(b, true);
          try {
            const b2 = await canvasToBlob(workC, "image/png");
            const fd = new FormData();
            fd.append("file", b2, "ocr.png");
            const res = await fetch("/api/editor/ocr", { method: "POST", body: fd });
            if (!res.ok) throw new Error(((await res.json().catch(() => ({}))).detail) || "HTTP " + res.status);
            const j = await res.json();
            ta.value = j.text || "(no text found)";
            ta.hidden = false;
            copyB.hidden = !j.text;
          } catch (e) { toast("OCR failed: " + e.message, "err"); }
          busy(b, false);
        });
        const ta = document.createElement("textarea");
        ta.className = "pcur-ocr";
        ta.rows = 7;
        ta.placeholder = "Extracted text appears here.";
        ta.hidden = true;
        add(pEl("div", "pcur-btns", "").appendChild(b));
        add(ta);
        const copyB = pBtn(icon("copy") + "<span>Copy to clipboard</span>", "secondary", async () => {
          try {
            await navigator.clipboard.writeText(ta.value);
            toast("Copied to clipboard", "ok");
          } catch (e) {
            ta.select();
            document.execCommand("copy");
            toast("Copied to clipboard", "ok");
          }
        });
        copyB.hidden = true;
        add(copyB);
        add(pEl("p", "pcur-hint", "Runs Tesseract OCR on the current image on the server."));
      }
    }

    // ---- sticker placement (baked; single editable instance) ----
    function placeSticker(bmp, kind) {
      const base = makeCanvas(workC.width, workC.height);
      ctx2d(base).drawImage(workC, 0, 0);
      const sk = { bitmap: bmp, kind: kind || "sticker", x: 0.5, y: 0.5, w: 0.24, rot: 0, base };
      S.sticker = sk;
      bakeSticker();
      syncDisp();
      commit();
    }

    // ---- transform bakes ----
    function rotate90(deg) {
      const tmp = makeCanvas(workC.height, workC.width);
      const x = ctx2d(tmp);
      x.translate(tmp.width / 2, tmp.height / 2);
      x.rotate(deg * Math.PI / 180);
      x.drawImage(workC, -workC.width / 2, -workC.height / 2);
      workC.width = tmp.width; workC.height = tmp.height;
      ctx2d(workC).drawImage(tmp, 0, 0);
      S.sticker = null;
      relayout();
      commit();
    }
    function bakeFlip(dir) {
      if (!dir) return;
      const tmp = makeCanvas(workC.width, workC.height);
      const x = ctx2d(tmp);
      if (dir === "h") { x.translate(tmp.width, 0); x.scale(-1, 1); }
      else { x.translate(0, tmp.height); x.scale(1, -1); }
      x.drawImage(workC, 0, 0);
      workC.width = tmp.width; workC.height = tmp.height;
      ctx2d(workC).drawImage(tmp, 0, 0);
      S.sticker = null;
      relayout();
      commit();
    }
    function rotateBy(deg) {
      const rad = deg * Math.PI / 180;
      const c = Math.abs(Math.cos(rad)), s = Math.abs(Math.sin(rad));
      const nw = Math.round(workC.width * c + workC.height * s);
      const nh = Math.round(workC.width * s + workC.height * c);
      const tmp = makeCanvas(nw, nh);
      const x = ctx2d(tmp);
      x.translate(nw / 2, nh / 2);
      x.rotate(rad);
      x.drawImage(workC, -workC.width / 2, -workC.height / 2);
      let cw = tmp.width, ch = tmp.height;
      if (Math.max(cw, ch) > MAX_PX) { const f = MAX_PX / Math.max(cw, ch); cw = Math.round(cw * f); ch = Math.round(ch * f); }
      workC.width = cw; workC.height = ch;
      ctx2d(workC).drawImage(tmp, 0, 0, cw, ch);
      S.sticker = null;
      relayout();
      commit();
    }

    // ---- export / save ----
    async function exportBlob(type, quality) {
      if (type === "pdf") {
        const jb = await canvasToBlob(workC, "image/jpeg", quality != null ? quality : 0.92);
        const u8 = new Uint8Array(await jb.arrayBuffer());
        return jpegToPdf(u8, workC.width, workC.height);
      }
      const mime = { jpeg: "image/jpeg", png: "image/png", webp: "image/webp" }[type] || "image/jpeg";
      return canvasToBlob(workC, mime, quality != null ? quality : 0.92);
    }
    function extFor(type) { return { jpeg: "jpg", png: "png", webp: "webp", pdf: "pdf" }[type] || "jpg"; }
    async function download(type, quality) {
      try {
        const blob = await exportBlob(type, quality);
        downloadBlob(blob, baseName.replace(/\.[^.]+$/, "") + "-edit." + extFor(type));
        toast("Download started", "ok");
      } catch (e) { toast("Export failed: " + e.message, "err"); }
    }
    function openSaveModal() {
      const m = pEl("div", "pcur-saveback");
      const dlg = pEl("div", "pcur-savedlg");
      dlg.setAttribute("role", "dialog");
      dlg.setAttribute("aria-label", "Save changes");
      dlg.innerHTML = `
        <h3>Save changes</h3>
        <div class="pcur-row"><label class="pcur-lbl">Format</label>
          <select class="pcur-sel">
            <option value="jpeg">JPEG</option><option value="png">PNG</option>
            <option value="webp">WebP</option><option value="pdf">PDF</option>
          </select>
        </div>
        <div class="pcur-slider"><div class="pcur-shead"><label class="pcur-lbl">Quality</label><span class="pcur-val">92%</span></div>
          <input type="range" min="50" max="100" value="92" step="1" class="pcur-q">
        </div>
        <div class="pcur-btns">
          <button type="button" class="btn sm primary" data-act="copy">${icon("copy")}<span>Save as copy</span></button>
          <button type="button" class="btn sm secondary" data-act="overwrite">${icon("swap")}<span>Overwrite original</span></button>
          <button type="button" class="btn sm ghost" data-act="download">${icon("down")}<span>Download</span></button>
        </div>
        <p class="pcur-hint">“Save as copy” adds a new photo to the library. “Overwrite” replaces the original in place.</p>`;
      m.appendChild(dlg);
      rootEl.appendChild(m);
      const sel = m.querySelector(".pcur-sel");
      const q = m.querySelector(".pcur-q");
      const qv = m.querySelector(".pcur-val");
      const copyBtn = dlg.querySelector('[data-act="copy"]');
      const overwriteBtn = dlg.querySelector('[data-act="overwrite"]');
      q.addEventListener("input", () => { qv.textContent = q.value + "%"; });
      const updateFormatUi = () => {
        const isPdf = sel.value === "pdf";
        copyBtn.disabled = isPdf;
        overwriteBtn.disabled = isPdf;
        copyBtn.title = isPdf ? "PDF can only be downloaded" : "";
        overwriteBtn.title = isPdf ? "PDF can only be downloaded" : "";
      };
      sel.addEventListener("change", updateFormatUi);
      updateFormatUi();

      const closeM = () => m.remove();
      m.addEventListener("close-save", closeM);
      m.addEventListener("click", (e) => { if (e.target === m) closeM(); });

      const doSave = async (mode) => {
        const type = sel.value;
        if (type === "pdf") {
          toast("PDF can only be downloaded, not saved to the photo library.", "err");
          return;
        }
        try {
          const quality = Number(q.value) / 100;
          const blob = await exportBlob(type, quality);
          const fd = new FormData();
          fd.append("file", blob, "edit." + extFor(type));
          fd.append("mode", mode);
          fd.append("format", type);
          const res = await fetch(`/api/images/${id}/edit-save`, { method: "POST", body: fd });
          if (!res.ok) throw new Error(((await res.json().catch(() => ({}))).detail) || "HTTP " + res.status);
          const j = await res.json();
          closeM();
          toast(mode === "copy" ? "Saved as a new photo" : "Original updated", "ok");
          if (onSaved) onSaved(j);
        } catch (e) { toast("Save failed: " + e.message, "err"); }
      };
      copyBtn.onclick = () => doSave("copy");
      overwriteBtn.onclick = () => doSave("overwrite");
      dlg.querySelector('[data-act="download"]').onclick = () => { download(sel.value, Number(q.value) / 100); closeM(); };
    }

    // ---- compare (hold to view the unedited original) ----
    function setCompare(on) {
      if (S.compare === on) return;
      S.compare = on;
      if (on) {
        wrap.style.filter = "none";
        const x = ctx2d(dispC);
        x.clearRect(0, 0, dispC.width, dispC.height);
        x.drawImage(originalC, 0, 0, workC.width, workC.height);
        const ox = ctx2d(ovC);
        ox.clearRect(0, 0, ovC.width, ovC.height);
        ox.fillStyle = "rgba(0,0,0,.35)";
        ox.fillRect(0, 0, ovC.width, ovC.height);
        ox.fillStyle = "#fff";
        ox.font = `bold ${Math.max(16, Math.round(ovC.width / 40))}px sans-serif`;
        ox.textAlign = "center";
        ox.fillText("ORIGINAL", ovC.width / 2, ovC.height / 2);
      } else {
        syncDisp();
        renderOverlay();
      }
    }

    // ---- keyboard ----
    function onKey(ev) {
      if (closed) return;
      if (ev.key === "Escape") {
        const sb = rootEl.querySelector(".pcur-saveback");
        if (sb) {
          ev.preventDefault(); ev.stopPropagation();
          sb.dispatchEvent(new CustomEvent("close-save"));
          return;
        }
        ev.preventDefault(); ev.stopPropagation();
        close();
        return;
      }
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "z") {
        ev.preventDefault(); ev.stopPropagation();
        if (ev.shiftKey) redo(); else undo();
        return;
      }
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "y") { ev.preventDefault(); ev.stopPropagation(); redo(); return; }
      const tag = (ev.target && ev.target.tagName) || "";
      if (/INPUT|TEXTAREA|SELECT/.test(tag)) return;
      if (ev.key === " ") { S._space = true; stage.style.cursor = "grab"; ev.preventDefault(); return; }
      if (ev.key === "c" || ev.key === "C") { setCompare(true); return; }
      const n = parseInt(ev.key, 10);
      if (n >= 1 && n <= TOOLS.length) selectTool(TOOLS[n - 1][0]);
    }
    function onKeyUp(ev) {
      if (closed) return;
      if (ev.key === " ") { S._space = false; stage.style.cursor = ""; }
      if (ev.key === "c" || ev.key === "C") setCompare(false);
    }

    // ---- mount ----
    document.body.appendChild(rootEl);
    document.body.style.overflow = "hidden";
    stage.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove, true);
    window.addEventListener("pointerup", onPointerUp, true);
    stage.addEventListener("wheel", onWheel, { passive: false });
    stage.addEventListener("contextmenu", (e) => e.preventDefault());
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("resize", relayout);
    undoBtn.onclick = undo;
    redoBtn.onclick = redo;
    dlBtn.onclick = () => download("jpeg", 0.92);
    saveBtn.onclick = openSaveModal;
    closeBtn.onclick = close;
    compareBtn.addEventListener("pointerdown", (e) => { e.preventDefault(); setCompare(true); });
    const cmpUp = () => setCompare(false);
    compareBtn.addEventListener("pointerup", cmpUp);
    compareBtn.addEventListener("pointerleave", cmpUp);
    window.addEventListener("pointerup", cmpUp);

    // focus trap
    let releaseTrap = null;
    releaseTrap = (function () {
      const sel = 'button:not([disabled]), [href], input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])';
      const onTab = (ev) => {
        if (ev.key !== "Tab" || closed) return;
        const items = $$(sel, rootEl).filter((n) => !n.hidden && n.offsetParent !== null);
        if (!items.length) return;
        const first = items[0], last = items[items.length - 1];
        if (ev.shiftKey && document.activeElement === first) { ev.preventDefault(); last.focus(); }
        else if (!ev.shiftKey && document.activeElement === last) { ev.preventDefault(); first.focus(); }
      };
      window.addEventListener("keydown", onTab, true);
      return () => window.removeEventListener("keydown", onTab, true);
    })();
    const prevFocus = document.activeElement;

    let ro = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => relayout());
      ro.observe(stage);
    }

    relayout();
    syncDisp();
    selectTool("crop");
    historyBtns();
    await commit();
    state.active = true;
    closeBtn.focus();
    requestAnimationFrame(() => {
      relayout();
      syncDisp();
    });

    async function close() {
      if (closed) return;
      closed = true;
      state.active = false;
      if (ro) ro.disconnect();
      window.removeEventListener("pointermove", onPointerMove, true);
      window.removeEventListener("pointerup", onPointerUp, true);
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("keyup", onKeyUp, true);
      window.removeEventListener("pointerup", cmpUp);
      window.removeEventListener("resize", relayout);
      if (releaseTrap) releaseTrap();
      document.body.style.overflow = "";
      rootEl.remove();
      for (const b of S.history) closeBitmap(b);
      S.history = [];
      state.histLen = 0;
      if (S.sticker && S.sticker.bitmap && S.sticker.bitmap.close) closeBitmap(S.sticker.bitmap);
      if (prevFocus && prevFocus.focus && document.contains(prevFocus)) prevFocus.focus({ preventScroll: true });
      if (onClose) onClose();
    }

    return { close, state: S };
  }

  state.openPhotoEditor = openPhotoEditor;
  state.remaster = remaster;
  state.sharpen = sharpen;
  root.PCUR_EDITOR = state;
})(typeof window !== "undefined" ? window : globalThis);

export { };
