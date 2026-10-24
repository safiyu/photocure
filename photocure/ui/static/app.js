/* Photocure review dashboard (vanilla JS, no build step). */
(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (n) => Number(n) || 0;
  const fmtN = (n) => num(n).toLocaleString();
  const icon = (n, c) => `<svg class="ic ${c || ""}" aria-hidden="true"><use href="#i-${n}"/></svg>`;

  const PAGE = 200;
  const state = {
    tab: "library", cats: [], junk: new Set(), utility: new Set(), utilityKind: "", byId: {}, busy: false, taskRunning: false,
    inferred: false,
    triageSel: -1, triageGrid: null, undo: [], picks: new Set(), peoplePicks: new Set(), lastPick: null,
    overview: null, eventId: null,
    backend: null, thresholds: null, manualCount: 0,
    settings: null, draft: { device: "auto", device_index: 0 }, busyBtn: null,
    dismissedRun: null, lastRun: null,
  };

  const TITLES = {
    library: ["Pipeline", "Choose a folder and let Photocure sort what it finds. Nothing is moved until you apply a plan."],
    categories: ["Library", "Every photo, grouped by the category it was sorted into."],
    search: ["Search", "Describe a photo in your own words and Photocure finds it."],
    duplicates: ["Duplicates", "Identical and look-alike photos. Choose the one to keep in each group."],
    bursts: ["Bursts", "Shots taken seconds apart. The best one of each group is kept."],
    people: ["People", "Faces Photocure grouped together. Name a cluster and it carries over everywhere that person appears."],
    highlights: ["Highlights", "Your most appealing photos first."],
    events: ["Events", "Photos grouped by when and where they were taken."],
    junk: ["Junk review", "Duplicates, graphics and accidents. Move anything that does not belong."],
    utility: ["Utility", "Receipts, notes, screenshots, labels and signs, kept apart from your photos and from Junk."],
    uncertain: ["Finalize", "The analysis was not sure about these. Choose a category for each."],
    settings: ["Settings", "Choose what runs the analysis and how many workers it uses."],
    actions: ["Apply and undo", "Review the plan, then organize your files. Every change can be undone."],
  };

  // ------------------------------------------------------------------ utils
  function toast(msg, cls, action) {
    const box = $("#toasts");
    const el = document.createElement("div");
    el.className = "toast " + (cls || "");
    el.innerHTML = (cls === "err" ? icon("alert", "sm") : cls === "ok" ? icon("check", "sm") : "") +
      `<span>${esc(msg)}</span>`;
    let timer = null;
    const dismiss = () => {
      clearTimeout(timer);
      el.classList.add("out");
      afterAnim(el, moMs("--mo-fast", 160) + 100).then(() => el.remove());
    };
    if (action) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "toast-action";
      b.textContent = action.label;
      b.onclick = async () => {
        b.disabled = true;
        try { await action.fn(); } catch (e) { toast(e.message, "err"); }
        dismiss();
      };
      el.append(b);
    }
    box.appendChild(el);
    while (box.children.length > 4) box.firstChild.remove();
    timer = setTimeout(dismiss, cls === "err" ? 6000 : action ? 8000 : 3200);
  }

  // Motion helpers: every close waits on the animation but never depends on it alone.
  function reducedMotion() {
    return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  // Motion tokens from CSS (ms), so JS timings track the stylesheet.
  function moMs(name, fallback) {
    const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name));
    return v > 0 ? v : fallback;
  }

  function afterAnim(el, fallbackMs) {
    if (reducedMotion()) return Promise.resolve();
    return new Promise((resolve) => {
      let finished = false;
      let timer = null;
      const onEnd = (e) => { if (e.target === el) finish(); };
      function finish() {
        if (finished) return;
        finished = true;
        el.removeEventListener("animationend", onEnd);
        clearTimeout(timer);
        resolve();
      }
      el.addEventListener("animationend", onEnd);
      timer = setTimeout(finish, fallbackMs);
    });
  }

  function openOverlay(el) {
    if (!el) return;
    el._closeSeq = (el._closeSeq || 0) + 1;
    el.classList.remove("closing");
    el.hidden = false;
  }

  async function closeOverlay(el) {
    if (!el || el.hidden || el.classList.contains("closing")) return;
    const seq = el._closeSeq = (el._closeSeq || 0) + 1;
    el.classList.add("closing");
    await afterAnim(el, moMs("--mo-fast", 160) + 100);
    if (el._closeSeq !== seq) return;
    el.hidden = true;
    el.classList.remove("closing");
  }

  function positionNavIndicator() {
    const ind = $("#nav-indicator");
    if (!ind) return;
    const item = $("nav .tab.active");
    if (!item) { ind.style.opacity = "0"; return; }
    const y = item.offsetTop + (item.offsetHeight - ind.offsetHeight) / 2;
    if (!ind.dataset.ready) {
      ind.style.transition = "none";
      ind.style.setProperty("--ni-y", y + "px");
      void ind.offsetWidth;
      ind.style.transition = "";
      ind.dataset.ready = "1";
    } else {
      ind.style.setProperty("--ni-y", y + "px");
    }
    ind.style.opacity = "1";
  }

  // Shared focus trap helper (WCAG 2.1 AA focus containment).
  const activeTraps = [];

  function trapFocus(container, onEscape) {
    const selector = 'button:not([disabled]):not([hidden]), [href], input:not([disabled]):not([hidden]), select:not([disabled]):not([hidden]), textarea:not([disabled]):not([hidden]), [tabindex]:not([tabindex="-1"]):not([hidden])';
    const getFocusable = () => Array.from(container.querySelectorAll(selector)).filter((el) => {
      if (el.hidden || el.disabled || el.closest('[hidden]')) return false;
      if (el.style.display === "none" || el.style.visibility === "hidden") return false;
      const r = el.getBoundingClientRect();
      return r.width > 0 || r.height > 0 || el.getClientRects().length > 0;
    });

    const onKey = (e) => {
      if (activeTraps[activeTraps.length - 1] !== trapRecord) return;
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        if (onEscape) onEscape(e);
        return;
      }
      if (e.key !== "Tab") return;
      const items = getFocusable();
      if (!items.length) {
        e.preventDefault();
        return;
      }
      const idx = items.indexOf(document.activeElement);
      if (e.shiftKey) {
        if (idx <= 0) {
          items[items.length - 1].focus();
          e.preventDefault();
        }
      } else {
        if (idx === -1 || idx >= items.length - 1) {
          items[0].focus();
          e.preventDefault();
        }
      }
    };

    let isTrapping = false;
    const onDocFocusIn = (e) => {
      if (isTrapping) return;
      if (activeTraps[activeTraps.length - 1] !== trapRecord) return;
      if (!container.contains(e.target)) {
        const items = getFocusable();
        if (items.length) {
          isTrapping = true;
          try {
            items[0].focus();
          } finally {
            isTrapping = false;
          }
        }
      }
    };

    const trapRecord = { container, onEscape, getFocusable };
    activeTraps.push(trapRecord);

    container.addEventListener("keydown", onKey);
    document.addEventListener("focusin", onDocFocusIn);
    return () => {
      container.removeEventListener("keydown", onKey);
      document.removeEventListener("focusin", onDocFocusIn);
      const idx = activeTraps.indexOf(trapRecord);
      if (idx !== -1) activeTraps.splice(idx, 1);
    };
  }

  // Accessible confirmation and input dialog. All text goes in through textContent.
  function showConfirmDialog(o) {
    o = o || {};
    const modal = $("#confirm-modal"), ok = $("#confirm-modal-ok"), cancel = $("#confirm-modal-cancel");
    const backdrop = $(".confirm-backdrop", modal);
    const variant = ["danger", "warning", "primary"].includes(o.variant) ? o.variant : "primary";
    $("#confirm-modal-title").textContent = o.title || "Please confirm";
    const sub = $("#confirm-modal-subtitle");
    sub.textContent = o.subtitle || "";
    sub.hidden = !o.subtitle;
    $("#confirm-modal-message").textContent = o.message || "";
    const det = $("#confirm-modal-detail");
    det.replaceChildren();
    if (o.detail instanceof Node) det.appendChild(o.detail);
    else if (o.detail) det.textContent = String(o.detail);

    const inputWrap = $("#confirm-modal-input-wrap");
    const inputLabel = $("#confirm-modal-input-label");
    const inputEl = $("#confirm-modal-input");
    const inputErr = $("#confirm-modal-input-error");
    const hasInput = !!o.input;

    if (inputWrap) {
      inputWrap.hidden = !hasInput;
      if (hasInput) {
        inputLabel.textContent = o.input.label || "";
        inputLabel.hidden = !o.input.label;
        inputEl.value = o.input.value || "";
        inputEl.placeholder = o.input.placeholder || "";
        inputErr.hidden = true;
        inputErr.textContent = "";
      }
    }

    ok.textContent = o.confirmText || "OK";
    cancel.textContent = o.cancelText || "Cancel";
    ok.className = "btn " + { danger: "danger", warning: "commit", primary: "primary" }[variant];
    const previous = document.activeElement;
    openOverlay(modal);

    return new Promise((resolve) => {
      let releaseTrap = null;
      const finish = (value) => {
        if (releaseTrap) {
          releaseTrap();
          releaseTrap = null;
        }
        closeOverlay(modal);
        ok.onclick = cancel.onclick = backdrop.onclick = null;
        if (inputEl) inputEl.onkeydown = null;
        if (previous && previous.focus) previous.focus();
        resolve(value);
      };

      const trySubmit = () => {
        if (hasInput) {
          const val = inputEl.value;
          if (typeof o.validate === "function") {
            const err = o.validate(val);
            if (err) {
              inputErr.textContent = err;
              inputErr.hidden = false;
              inputEl.focus();
              return;
            }
          }
          finish(val);
        } else {
          finish(true);
        }
      };

      if (hasInput && inputEl) {
        inputEl.onkeydown = (e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            trySubmit();
          }
        };
      }

      releaseTrap = trapFocus(modal, () => finish(hasInput ? null : false));
      ok.onclick = trySubmit;
      cancel.onclick = () => finish(hasInput ? null : false);
      backdrop.onclick = () => finish(hasInput ? null : false);

      if (hasInput && inputEl) inputEl.focus();
      else (variant === "primary" ? ok : cancel).focus();
    });
  }

  // ---------------------------------------------------------------- auth
  async function exchangeCode(code) {
    const r = await fetch("/auth/exchange", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    return r.ok;
  }

  // A one-time code arrives in the URL fragment; clear it before any network call.
  async function exchangeFromHash() {
    const m = /^#code=([A-Za-z0-9_-]+)$/.exec(location.hash);
    if (!m) return;
    history.replaceState(null, "", location.pathname + location.search);
    await exchangeCode(m[1]);
  }

  let authPromise = null;
  function requireAuth() {
    if (authPromise) return authPromise;
    authPromise = new Promise((resolve) => {
      const modal = $("#auth-modal"), input = $("#auth-input"), err = $("#auth-error");
      input.value = "";
      err.hidden = true;
      modal.hidden = false;
      input.focus();
      $("#auth-form").onsubmit = async (e) => {
        e.preventDefault();
        if (await exchangeCode(input.value.trim())) {
          modal.hidden = true;
          authPromise = null;
          resolve();
        } else {
          err.textContent = "That code or token was not accepted.";
          err.hidden = false;
          input.select();
        }
      };
    });
    return authPromise;
  }

  async function api(path, opts) {
    opts = opts || {};
    let res = await fetch(path, opts);
    if (res.status === 401) {
      await requireAuth();
      res = await fetch(path, opts);
    }
    if (!res.ok) {
      let detail = res.statusText, body = null;
      try {
        body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : (body.error || JSON.stringify(body.detail || body));
      } catch (e) { /* keep status text */ }
      const err = new Error(detail || "HTTP " + res.status);
      err.status = res.status;
      err.body = body;
      throw err;
    }
    return res.json();
  }

  function post(path, body) {
    return api(path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body == null ? {} : body),
    });
  }

  const eff = (it) => it.manual_bucket || it.category;
  // Dynamic tier: micro (440px) when small, medium (800px) when slider is enlarged (>210px)
  // or on high-DPI screens, so thumbnails never get noisy/pixelated when enlarged.
  function gridThumbTier(density) {
    const d = density != null ? density : (state.density || 200);
    const dpr = typeof window !== "undefined" ? (window.devicePixelRatio || 1) : 1;
    return (d > 210 || d * dpr > 360) ? "medium" : "micro";
  }

  // Tiers: micro for the grid (when small), medium for enlarged cards/bursts, preview for lightbox.
  const thumbUrl = (id, tier) => {
    const it = state.byId[id];
    return "/thumbnail/" + num(id) + "?tier=" + (tier || "micro") + (it && it.fp ? "&v=" + it.fp : "");
  };
  const catKey = (c) => String(c || "unrouted").replace(/[^a-z_]/g, "") || "unknown";
  // Most useful first, so the number keys 1-9 cover the categories people assign most.
  const CAT_ORDER = ["people", "food", "scenery", "animal", "screenshot", "document", "graphic",
    "illustration", "slide", "utility", "architecture", "vehicle", "plant", "product", "event", "accidental",
    "uncertain", "duplicate", "near_duplicate", "burst_alternate", "unreadable", "custom", "unknown"];
  const TRIAGE_SKIP = new Set(["unknown", "duplicate", "near_duplicate", "uncertain",
    "burst_alternate", "unreadable", "custom"]);
  const sortCats = (cats) => cats.slice().sort((a, b) => {
    const ia = CAT_ORDER.indexOf(a), ib = CAT_ORDER.indexOf(b);
    return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib);
  });
  const catStyle = (c) => `--cat:var(--c-${catKey(c)}, var(--c-unknown))`;
  const catLabel = (c) => {
    if (c === "utility") return "Label or sign";
    if (c === "slide") return "Notes and slides";
    const t = String(c || "Not sorted yet").replace(/_/g, " ");
    return t.charAt(0).toUpperCase() + t.slice(1);
  };
  const fmtBytes = (n) => {
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0, v = num(n);
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return v.toFixed(i ? 1 : 0) + " " + u[i];
  };

  // "Scan finished: 600 files seen" becomes a bold headline plus a quieter detail line.
  function splitLog(msg) {
    const m = String(msg).match(/^(.{3,48}?):\s+(.+)$/);
    return m ? { title: m[1], detail: m[2] } : { title: String(msg), detail: "" };
  }

  function updateActivityCount(box) {
    const panel = box.closest(".activity");
    const badge = panel && panel.querySelector(".activity-count");
    if (badge) {
      const n = box.querySelectorAll(".logline[data-t]").length;
      badge.textContent = n ? String(n) : "";
    }
  }

  function logTo(boxSel, msg, cls) {
    const box = $(boxSel);
    if (!box) return;
    $$(":scope > .logline:not([data-t])", box).forEach((e) => e.remove());
    $$(":scope > .logline.running", box).forEach((e) => e.classList.remove("running"));
    const running = !cls && /(\.\.\.|\u2026)$/.test(msg);
    const { title, detail } = splitLog(msg);
    const row = document.createElement("div");
    row.className = "logline" + (cls ? " " + cls : "") + (running ? " running" : "");
    row.dataset.t = "1";
    const ico = document.createElement("span");
    ico.className = "log-ico";
    ico.setAttribute("aria-hidden", "true");
    ico.textContent = cls === "err" || cls === "warn" ? "!" : cls === "ok" ? "\u2713" : "";
    const body = document.createElement("div");
    body.className = "log-body";
    const head = document.createElement("span");
    head.className = "log-title";
    head.textContent = title;
    body.appendChild(head);
    if (detail) {
      const more = document.createElement("span");
      more.className = "log-detail";
      more.textContent = detail;
      body.appendChild(more);
    }
    const time = document.createElement("time");
    time.className = "log-time";
    time.textContent = new Date().toLocaleTimeString();
    row.append(ico, body, time);
    box.prepend(row);
    box.scrollTop = 0;
    while (box.children.length > 200) box.lastChild.remove();
    updateActivityCount(box);
  }
  const logLine = (m, c) => logTo("#action-log", m, c);
  const libLog = (m, c) => logTo("#library-log", m, c);

  // ------------------------------------------------------------------ theme
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("pcur-theme", t); } catch (e) { /* private mode */ }
    const sw = $("#theme-switch");
    if (sw) {
      sw.checked = t === "dark";
      sw.setAttribute("aria-checked", String(t === "dark"));
    }
  }

  // ------------------------------------------------------------------ cards
  function flagList(it) {
    const out = [];
    const add = (k, ic, g, text) => out.push({ k, ic, g, text });
    if (num(it.rating) === -1) add("bad", "\u2715", "Rating", "Rejected");
    if (it.is_favorite) add("fav", "\u2665", "Rating", "\u2665 Favourite");
    if (num(it.rating) > 0) add("", "\u2605", "Rating", "\u2605" + num(it.rating));
    if (it.has_blinks) add("warn", "\u25d0", "Quality", "Blink");
    if (it.is_blurry) add("warn", "\u2248", "Quality", "Blurry");
    if (it.is_too_dark) add("warn", "\u263e", "Quality", "Dark");
    if (it.is_overexposed) add("warn", "\u2600", "Quality", "Overexposed");
    if (it.locked) add("lock", "\u25a3", "Status", "Locked");
    if (it.is_best_shot) add("good", "\u25ce", "Status", "Best shot");
    if (it.is_duplicate) add("bad", "\u29c9", "Status", "Duplicate");
    if (it.near_dup_of != null) add("bad", "\u29c9", "Status", "Near duplicate");
    if (it.is_alternate) add("", "\u2194", "Status", "Alternate");
    return out;
  }

  // Small DOM builder: text always goes through textContent. "html" is only for trusted constants (icons).
  function el(tag, props, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k === "html") n.innerHTML = v;
      else n.setAttribute(k, v === true ? "" : String(v));
    }
    kids.flat().forEach((c) => { if (c != null && c !== false) n.append(c); });
    return n;
  }

  function shortReason(it) {
    const r = it.uncertain_reason;
    if (!r || it.manual_bucket) return "";
    if (it.top1 == null) return "Not analysed yet";
    const t = state.thresholds || {};
    const p1 = Math.round(num(it.top1_score) * 100);
    const kind = String(r.kind || "");
    const low = /low/.test(kind) ? true : /close|margin/.test(kind) ? false
      : (t.min_confidence != null ? num(it.top1_score) < t.min_confidence : false);
    if (low) return `Unsure, best guess ${catLabel(it.top1)} ${p1}%`;
    if (it.top2) return `Close call: ${catLabel(it.top1)} or ${catLabel(it.top2)}`;
    return r.text;
  }

  function cardNode(it) {
    const id = num(it.id);
    const rawCat = eff(it);
    const meta = [];
    if (it.width && it.height) meta.push(num(it.width) + "\u00d7" + num(it.height));
    if (it.taken_at) meta.push(String(it.taken_at).slice(0, 10));
    if (it.score != null) meta.push("match " + num(it.score).toFixed(2));
    const picked = state.picks.has(id);
    const isBest = !!(it.is_best_shot || it.burst_role === "best" || (it.tags && it.tags.includes("Best shot")));
    const cls = ["card", it.locked ? "locked" : "", picked ? "picked" : "", num(it.rating) === -1 ? "rejected" : "",
      it.delete_approved ? "marked" : "", isBest ? "best" : ""]
      .filter(Boolean).join(" ");
    const curTier = gridThumbTier();
    const v = it && it.fp ? "&v=" + it.fp : "";
    const img = el("img", {
      loading: "lazy",
      decoding: "async",
      src: thumbUrl(id, curTier),
      srcset: `${thumbUrl(id, "micro")} 440w, ${thumbUrl(id, "medium")} 800w`,
      sizes: `(max-width: 480px) 100vw, ${Math.round((state.density || 200) * 1.25)}px`,
      alt: it.filename || "",
    });
    img.onerror = () => { img.style.opacity = ".15"; };
    const thumb = el("div", { class: "thumb photo", title: "View photo" }, img);
    const showReason = it.uncertain_reason && !it.manual_bucket;
    // Flags collapse into a "+N" pill (hover or tap for the list); the full text stays in the DOM for screen readers.
    const allFlags = flagList(it);
    const flagPill = allFlags.length ? el("span", { class: "flags" },
      el("span", {
        class: "cat-tag-pill tag-overflow-pill",
        text: "+" + allFlags.length,
        "aria-hidden": "true",
        "data-overflow-tags": JSON.stringify(allFlags),
      }),
      el("span", { class: "sr-only", text: allFlags.map((f) => f.text).join(", ") })) : null;
    const cap = el("figcaption", { class: "caption info" },
      el("div", { class: "fn name", title: it.relative_path || it.filename, text: it.filename }),
      showReason ? el("div", { class: "reason", title: it.uncertain_reason.text, text: shortReason(it) }) : null,
      el("div", { class: "meta", text: meta.join(", ") }),
      el("div", { class: "catline", style: catStyle(rawCat) },
        el("button", { class: "catpick cat", type: "button", "data-catpick": id, title: "Change category",
          "aria-haspopup": "listbox", "aria-label": `Category ${catLabel(rawCat)}, change`, tabindex: -1 },
        el("i", { class: "dot" }), el("span", { class: "t", text: catLabel(rawCat) })),
        flagPill,
        it.manual_bucket ? [el("span", { class: "tag", text: "edited" }),
          el("button", { class: "x", "data-clear": id, title: "Back to the AI choice",
            "aria-label": "Back to the AI choice", tabindex: -1, html: icon("x", "sm") })] : null));

    const actions = el("div", { class: "actions" },
      el("button", {
        type: "button",
        class: "btn btn-del" + (it.delete_approved ? " is-del" : ""),
        "data-approve-delete": id,
        "aria-pressed": String(!!it.delete_approved),
        title: it.delete_approved ? "Marked for delete (click to unmark)" : "Mark for delete",
        "aria-label": it.delete_approved ? "Clear delete mark" : "Mark for delete",
        tabindex: -1,
      }, el("span", { class: "btn-ic", html: icon("trash", "sm") }), el("span", { class: "btn-lbl", text: it.delete_approved ? "Marked" : "Delete" })),
      el("button", {
        type: "button",
        class: "btn btn-lock" + (it.locked ? " is-locked" : ""),
        "data-lock": id,
        title: (it.locked ? "Unlock" : "Keep") + " (l)",
        "aria-label": it.locked ? "Unlock" : "Keep",
        tabindex: -1,
      }, el("span", { class: "btn-ic", html: icon("heart", "sm") }), el("span", { class: "btn-lbl", text: it.locked ? "Kept" : "Keep" })),
      el("button", {
        type: "button",
        class: "btn btn-pick" + (picked ? " is-picked" : ""),
        "data-pick": id,
        "aria-pressed": String(picked),
        title: picked ? "Deselect" : "Select",
        "aria-label": "Select " + (it.filename || ""),
        tabindex: -1,
      }, el("span", { class: "btn-ic", html: icon("check", "sm") }), el("span", { class: "btn-lbl", text: picked ? "Selected" : "Select" })),
      el("button", {
        type: "button",
        class: "btn btn-similar",
        "data-similar": id,
        title: "Find similar photos",
        "aria-label": "Find similar photos",
        tabindex: -1,
      }, el("span", { class: "btn-ic", html: icon("copy", "sm") }), el("span", { class: "btn-lbl", text: "Similar" })),
      it.face_count ? el("button", {
        type: "button",
        class: "btn btn-faces",
        "data-faces": id,
        title: "Inspect faces",
        "aria-label": "Inspect faces",
        tabindex: -1,
      }, el("span", { class: "btn-ic", html: icon("face", "sm") }), el("span", { class: "btn-lbl", text: "Faces" })) : null
    );

    return el("figure", { class: cls, "data-id": id, tabindex: -1, role: "gridcell" }, thumb, cap, actions);
  }

  const junkCardNode = cardNode;

  function emptyHTML(e) {
    return `<div class="empty">${icon(e.icon || "image")}<h3>${esc(e.title)}</h3><p>${esc(e.text || "")}</p>` +
      (e.action ? `<button class="btn" ${e.action.attr}>${esc(e.action.label)}</button>` : "") + "</div>";
  }

  // ------------------------------------------------------------ paged grids
  const grids = {};

  function skeletons(n) {
    return Array.from({ length: n }, () => '<div class="skel"></div>').join("");
  }

  function emptyNode(e) {
    const d = document.createElement("div");
    d.innerHTML = emptyHTML(e);  // emptyHTML escapes its text
    return d.firstElementChild || d;
  }

  // Cards fade in only while a freshly replaced grid fills, so single edits never replay it.
  function flashGrid(host) {
    host.classList.add("anim-in");
    clearTimeout(host._flash);
    host._flash = setTimeout(() => host.classList.remove("anim-in"), 1500);
  }

  // The Utility tab asks for the whole family, or for one kind (which is one category).
  function utilityQuery(kind) {
    return kind ? "category=" + encodeURIComponent(kind) : "group=utility";
  }

  function makeGrid(key, cfg) {
    if (!window.PCUR_GRID) throw new Error("virtual_grid.mjs did not load");
    const host = $(cfg.el);
    const g = { key, cfg, el: host, countEl: cfg.count ? $(cfg.count) : null, stale: false };
    const moreEl = cfg.more ? $(cfg.more) : null;
    if (moreEl) moreEl.hidden = true;  // pages load by themselves as you scroll
    g.vg = new window.PCUR_GRID.VirtualGrid({
      host,
      pageSize: PAGE,
      renderCard: cfg.renderCard || cardNode,
      captionH: cfg.captionH || 145,
      renderSkeleton: () => el("div", { class: "skel" }),
      async fetchPage(offset, limit) {
        const qs = cfg.params();
        const data = await api(`/api/images?${qs}${qs ? "&" : ""}limit=${limit}&offset=${offset}`);
        data.items.forEach((it) => { state.byId[it.id] = it; });
        return data;
      },
      onChange: () => g.sync(),
      onError(e) {
        g.vg.showEmpty(emptyNode({ icon: "alert", title: "Could not load photos", text: e.message }));
        toast(e.message, "err");
      },
    });
    g.sync = function () {
      const total = g.vg.total;
      if (g.countEl) g.countEl.textContent = total == null ? "" : `${fmtN(total)} photo${total === 1 ? "" : "s"}`;
      if (total === 0) {
        const e = typeof cfg.empty === "function" ? cfg.empty() : (cfg.empty || {});
        g.vg.showEmpty(emptyNode(Object.assign({ title: "No photos here" }, e)));
      }
    };
    Object.defineProperty(g, "total", { get: () => g.vg.total });
    g.reset = () => {
      flashGrid(host);
      return g.vg.reset();
    };
    g.update = () => { if (g.vg) g.vg.update(); };
    g.loadMore = async () => {};
    g.cards = () => $$(".card", host);
    g.applySel = () => g.vg.applyCursor();
    g.rerender = function (it) {
      const idx = g.vg.indexOfId(it.id);
      if (idx < 0) return;
      if (cfg.matches && !cfg.matches(it)) {
        const wasCursor = g.vg.cursorId === it.id;
        g.vg.removeItem(it.id);
        if (wasCursor) {  // keep the cursor where it was, on whatever slid into its place
          const next = g.vg.items[idx] || g.vg.items[idx - 1];
          g.vg.cursorId = next ? next.id : null;
          g.vg.applyCursor();
        }
      } else {
        g.vg.rerenderItem(it);
      }
    };
    grids[key] = g;
    return g;
  }

  function gridOf(node) {
    const host = node && node.closest && node.closest(".vgrid");
    return host ? allGrids().find((g) => g.el === host) || null : null;
  }

  const allGrids = () => Object.values(grids);
  function resetVisibleGrids() {
    allGrids().forEach((g) => { if (g.el.offsetParent !== null) g.reset(); else g.stale = true; });
  }
  function itemUpdated(it) {
    state.byId[it.id] = it;
    allGrids().forEach((g) => g.rerender(it));
    if (lb.open && lbIdAt(lb.i) === it.id) lbRender();
  }

  function setupGrids() {
    const val = (id) => { const e = $("#" + id); return e ? e.value : ""; };
    const chk = (id) => { const e = $("#" + id); return !!(e && e.checked); };
    makeGrid("cat", {
      el: "#grid", more: "#grid-more", count: "#rowcount",
      params() {
        const p = new URLSearchParams();
        if (val("q")) p.set("q", val("q"));
        if (val("catfilter")) p.set("category", val("catfilter"));
        if (chk("manualonly")) p.set("manual", "true");
        if (chk("neardup")) p.set("near_dup", "true");
        if (val("f-camera")) p.set("camera", val("f-camera"));
        const rt = val("f-rating");
        if (rt.startsWith("min:")) p.set("min_rating", rt.slice(4));
        else if (rt !== "") p.set("rating", rt);
        if (val("f-country")) p.set("country", val("f-country"));
        if (chk("f-fav")) p.set("favorite", "true");
        if (val("f-issue")) p.set("issue", val("f-issue"));
        scrubParams(p);
        p.set("sort", "taken");  // oldest first, so the date scrubber means something
        return p.toString();
      },
      matches(it) {
        if (val("catfilter") && eff(it) !== val("catfilter")) return false;
        if (chk("manualonly") && !it.manual_bucket) return false;
        if (chk("f-fav") && !it.is_favorite) return false;
        if (val("f-issue") && !hasIssue(it, val("f-issue"))) return false;
        const rt = val("f-rating");
        if (rt.startsWith("min:") && num(it.rating) < num(rt.slice(4))) return false;
        if (rt === "0" && num(it.rating) !== 0) return false;
        if (rt !== "" && !rt.startsWith("min:") && rt !== "0" && num(it.rating) !== num(rt)) return false;
        return true;
      },
      empty: () => (state.overview && state.overview.total === 0
        ? { icon: "import", title: "Your library is empty", text: "Import a folder of photos to get started.",
            action: { label: "Go to Import", attr: 'data-goto="library"' } }
        : { icon: "search", title: "No photos match", text: "Try a different search or clear the filters.",
            action: { label: "Clear filters", attr: "data-clearfilters" } }),
    });
    makeGrid("junk", {
      el: "#junkgrid", more: "#junkgrid-more", count: "#junkcount",
      renderCard: junkCardNode,
      captionH: 145,
      params() {
        const c = val("junkfilter");
        return c ? "category=" + encodeURIComponent(c) : "group=junk";
      },
      matches(it) {
        const c = val("junkfilter");
        return c ? eff(it) === c : state.junk.has(eff(it));
      },
      empty: { icon: "trash", title: "No junk to review", text: "Duplicates, graphics and accidental shots show up here after you analyze your photos." },
    });
    makeGrid("utility", {
      el: "#utilitygrid", more: "#utilitygrid-more", count: "#utilitycount",
      params: () => utilityQuery(state.utilityKind),
      matches(it) {
        const k = state.utilityKind;
        return k ? eff(it) === k : state.utility.has(eff(it));
      },
      empty: { icon: "inbox", title: "Nothing here yet", text: "Receipts, notes, screenshots, labels, signs and packaging show up here after you analyze your photos. Already analyzed? Press Re-check my photos." },
    });
    makeGrid("uncertain", {
      el: "#uncertaingrid", more: "#uncertaingrid-more", count: "#uncertaincount",
      params: () => {
        const p = new URLSearchParams();
        p.set("category", "uncertain");
        const qu = val("q-uncertain");
        if (qu) p.set("q", qu);
        const status = val("filter-uncertain-status");
        if (status === "chosen") p.set("manual", "true");
        return p.toString();
      },
      matches: (it) => {
        if (eff(it) !== "uncertain") return false;
        const status = val("filter-uncertain-status");
        if (status === "chosen" && !it.manual_bucket) return false;
        if (status === "unreviewed" && it.manual_bucket) return false;
        const qu = (val("q-uncertain") || "").toLowerCase().trim();
        if (qu && !((it.filename || "").toLowerCase().includes(qu) || (it.relative_path || "").toLowerCase().includes(qu))) return false;
        return true;
      },
      empty: { icon: "check", title: "Nothing to finalize", text: "Every photo was sorted with confidence." },
    });
    makeGrid("highlights", {
      el: "#highlightgrid", more: "#highlightgrid-more", count: "#highlightcount",
      params() {
        const m = val("minaes");
        return "highlights=true" + (m !== "" ? "&min_aesthetic=" + encodeURIComponent(m) : "");
      },
      empty: { icon: "star", title: "No highlights yet", text: "Analyze your photos so they can be scored." },
    });
    makeGrid("event", {
      el: "#eventdetail", more: "#eventdetail-more",
      params: () => "event_id=" + num(state.eventId),
      empty: { icon: "cal", title: "This event has no photos" },
    });
  }

  // -------------------------------------------------------- facets and scrubber
  const DAY = 86400000;
  const scrub = { lo: 0, days: 0 };

  function fillSelect(sel, firstLabel, rows) {
    const cur = sel.value;
    sel.replaceChildren(el("option", { value: "", text: firstLabel }),
      ...rows.map((r) => el("option", { value: r.value, text: r.text })));
    sel.value = rows.some((r) => r.value === cur) ? cur : "";
  }

  const countryName = (code) => {
    try { return new Intl.DisplayNames(["en"], { type: "region" }).of(code) || code; } catch (e) { return code; }
  };

  function scrubDate(n) { return new Date(scrub.lo + n * DAY).toISOString().slice(0, 10); }

  function scrubLabels() {
    $("#scrub-from-label").textContent = scrubDate(num($("#scrub-from").value));
    $("#scrub-to-label").textContent = scrubDate(num($("#scrub-to").value));
  }

  let datePreset = "all";

  function hasActiveFilters() {
    const val = (id) => { const e = $("#" + id); return e ? e.value : ""; };
    const chk = (id) => { const e = $("#" + id); return !!(e && e.checked); };
    if (val("q")) return true;
    if (val("catfilter")) return true;
    if (chk("manualonly")) return true;
    if (chk("neardup")) return true;
    if (val("f-camera")) return true;
    if (val("f-rating")) return true;
    if (val("f-country")) return true;
    if (chk("f-fav")) return true;
    if (val("f-issue")) return true;
    const box = $("#scrubber");
    if (box && !box.hidden && scrub.days > 0) {
      const a = num($("#scrub-from").value), b = num($("#scrub-to").value);
      if (a > 0 || b < scrub.days) return true;
    }
    if (datePreset !== "all" && datePreset !== "custom") return true;
    return false;
  }

  function updateResetFiltersButton() {
    const btn = $("#reset-filters");
    if (!btn) return;
    btn.hidden = !hasActiveFilters();
  }

  function animateThumbs(targetFrom, targetTo, onDone) {
    const fromEl = $("#scrub-from"), toEl = $("#scrub-to");
    if (!fromEl || !toEl) { if (onDone) onDone(); return; }
    const prefersReduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) {
      fromEl.value = targetFrom;
      toEl.value = targetTo;
      scrubLabels();
      if (onDone) onDone();
      return;
    }
    const startFrom = num(fromEl.value), startTo = num(toEl.value);
    const startTime = performance.now();
    const duration = 200;
    function step(t) {
      const p = Math.min(1, (t - startTime) / duration);
      const ease = 1 - (1 - p) * (1 - p);
      fromEl.value = Math.round(startFrom + (targetFrom - startFrom) * ease);
      toEl.value = Math.round(startTo + (targetTo - startTo) * ease);
      scrubLabels();
      if (p < 1) {
        requestAnimationFrame(step);
      } else {
        fromEl.value = targetFrom;
        toEl.value = targetTo;
        scrubLabels();
        if (onDone) onDone();
      }
    }
    requestAnimationFrame(step);
  }

  function setDatePreset(p, reQuery = true) {
    datePreset = p;
    $$("#scrub-presets .preset-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.preset === p);
    });
    if (p === "custom") {
      updateResetFiltersButton();
      return;
    }
    if (scrub.days <= 0) {
      updateResetFiltersButton();
      return;
    }
    const now = new Date();
    let targetFromVal = 0;
    let targetToVal = scrub.days;

    if (p === "all") {
      targetFromVal = 0;
      targetToVal = scrub.days;
    } else if (p === "year") {
      const yearStart = new Date(now.getFullYear(), 0, 1).getTime();
      const targetFromTime = Math.max(scrub.lo, Math.min(scrub.lo + scrub.days * DAY, yearStart));
      targetFromVal = Math.round((targetFromTime - scrub.lo) / DAY);
      targetToVal = scrub.days;
    } else if (p === "90d") {
      const t90 = now.getTime() - 90 * DAY;
      const targetFromTime = Math.max(scrub.lo, Math.min(scrub.lo + scrub.days * DAY, t90));
      targetFromVal = Math.round((targetFromTime - scrub.lo) / DAY);
      targetToVal = scrub.days;
    } else if (p === "30d") {
      const t30 = now.getTime() - 30 * DAY;
      const targetFromTime = Math.max(scrub.lo, Math.min(scrub.lo + scrub.days * DAY, t30));
      targetFromVal = Math.round((targetFromTime - scrub.lo) / DAY);
      targetToVal = scrub.days;
    }

    animateThumbs(targetFromVal, targetToVal, () => {
      updateResetFiltersButton();
      if (reQuery) grids.cat.reset();
    });
  }

  function resetAllFilters() {
    $("#q").value = "";
    $("#catfilter").value = "";
    $("#manualonly").checked = false;
    $("#neardup").checked = false;
    ["f-camera", "f-rating", "f-country", "f-issue"].forEach((id) => { const el = $("#" + id); if (el) el.value = ""; });
    $("#f-fav").checked = false;
    resetScrubber();
    setDatePreset("all", false);
    updateResetFiltersButton();
    if (state.overview) renderComp(state.overview);
    grids.cat.reset();
  }

  function scrubParams(p) {
    const box = $("#scrubber");
    if (!box || box.hidden) return;
    const a = num($("#scrub-from").value), b = num($("#scrub-to").value);
    if (a > 0) p.set("date_from", scrubDate(a));
    if (b < scrub.days) p.set("date_to", scrubDate(b));
  }

  function resetScrubber() {
    $("#scrub-from").value = 0;
    $("#scrub-to").value = scrub.days;
    if (!$("#scrubber").hidden) scrubLabels();
  }

  // Bounds come from the library itself, never from hard-coded years.
  const ISSUE_ALL = "blurry,dark,overexposed,blink,duplicate";
  const ISSUE_ROWS = [["blurry", "Blurry"], ["dark", "Too dark"], ["overexposed", "Overexposed"], ["blink", "Blink"], ["duplicate", "Duplicate"]];

  // Mirrors the server's issue filter so an edited card leaves the grid when it no longer matches.
  function hasIssue(it, names) {
    const hit = {
      blurry: !!it.is_blurry, dark: !!it.is_too_dark, overexposed: !!it.is_overexposed,
      blink: !!it.has_blinks, duplicate: !!it.is_duplicate || it.near_dup_of != null,
    };
    return names.split(",").some((n) => hit[n]);
  }

  function applyFacets(f) {
    if (f.issues) {
      fillSelect($("#f-issue"), "Any quality", [{ value: ISSUE_ALL, text: `Possible junk, any problem (${fmtN(f.issues.any)})` }]
        .concat(ISSUE_ROWS.map(([k, label]) => ({ value: k, text: `${label} (${fmtN(f.issues[k])})` }))));
    }
    fillSelect($("#f-camera"), "All cameras", f.cameras.map((c) => ({ value: c.name, text: `${c.name} (${fmtN(c.count)})` })));
    fillSelect($("#f-country"), "All countries",
      f.countries.map((c) => ({ value: c.code, text: `${countryName(c.code)} (${fmtN(c.count)})` })));
    $("#f-country").hidden = f.countries.length === 0;
    $("#f-camera").hidden = f.cameras.length === 0;
    const box = $("#scrubber");
    if (!f.taken_min || !f.taken_max) { box.hidden = true; updateResetFiltersButton(); return; }
    const lo = Date.parse(f.taken_min + "T00:00:00Z"), hi = Date.parse(f.taken_max + "T00:00:00Z");
    const days = Math.max(0, Math.round((hi - lo) / DAY));
    const changed = lo !== scrub.lo || days !== scrub.days;
    scrub.lo = lo;
    scrub.days = days;
    box.hidden = days === 0;
    $("#scrub-from").max = $("#scrub-to").max = String(days);
    if (changed) {
      resetScrubber();
      setDatePreset("all", false);
    }
    if (!box.hidden) scrubLabels();
    updateResetFiltersButton();
  }

  async function loadFacets() {
    try { applyFacets(await api("/api/facets")); } catch (e) { /* the filters just stay as they were */ }
  }

  function bindFilters() {
    let t = 0;
    const again = () => {
      clearTimeout(t);
      updateResetFiltersButton();
      t = setTimeout(() => grids.cat.reset(), 200);
    };
    ["f-camera", "f-rating", "f-country", "f-fav", "f-issue"].forEach((id) => $("#" + id).addEventListener("change", again));
    const from = $("#scrub-from"), to = $("#scrub-to");
    from.addEventListener("input", () => {
      if (num(from.value) > num(to.value)) to.value = from.value;
      scrubLabels();
      const a = num(from.value), b = num(to.value);
      datePreset = (a === 0 && b === scrub.days) ? "all" : "custom";
      $$("#scrub-presets .preset-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.preset === datePreset);
      });
      again();
    });
    to.addEventListener("input", () => {
      if (num(to.value) < num(from.value)) from.value = to.value;
      scrubLabels();
      const a = num(from.value), b = num(to.value);
      datePreset = (a === 0 && b === scrub.days) ? "all" : "custom";
      $$("#scrub-presets .preset-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.preset === datePreset);
      });
      again();
    });
    $$("#scrub-presets .preset-btn").forEach((btn) => {
      btn.onclick = () => {
        const p = btn.dataset.preset;
        if (p === "custom") {
          datePreset = "custom";
          $$("#scrub-presets .preset-btn").forEach((b) => b.classList.toggle("active", b.dataset.preset === "custom"));
          updateResetFiltersButton();
        } else {
          setDatePreset(p, true);
        }
      };
    });
    const rf = $("#reset-filters");
    if (rf) rf.onclick = resetAllFilters;
  }

  // ------------------------------------------------------------- selection
  function applyPicks() {
    $$(".card").forEach((c) => {
      const on = state.picks.has(num(c.dataset.id));
      c.classList.toggle("picked", on);
      const b = $(".pick, .btn-pick", c);
      if (b) {
        b.setAttribute("aria-pressed", String(on));
        b.classList.toggle("is-picked", on);
        const lbl = $(".btn-lbl", b);
        if (lbl) lbl.textContent = on ? "Selected" : "Select";
      }
    });
    syncPeopleSel();
    const pn = state.tab === "people" ? state.peoplePicks.size : 0;
    const n = pn || state.picks.size;
    document.body.classList.toggle("has-picks", n > 0);
    const bi = pn ? null : burstPickInfo();
    $("#bulkbar").classList.toggle("burst-ctx", !!bi);
    $("#bulkbar").classList.toggle("people-ctx", pn > 0);
    if (n === 0) closeOverlay($("#bulkbar")); else openOverlay($("#bulkbar"));
    if (pn) {
      $("#bulk-count").textContent = `${fmtN(pn)} ${pn === 1 ? "person" : "people"}`;
      $("#bulk-sub").textContent = pn >= 2 ? "will become one person" : "select 2 or more to merge";
      $("#bulk-merge").disabled = pn < 2;
      $("#bulk-name").disabled = pn !== 1;
      $("#bulk-del-people").lastChild.textContent = pn === 1 ? "Delete person" : "Delete people";
    } else if (bi) {
      $("#bulk-count").textContent = `${fmtN(n)} frame${n === 1 ? "" : "s"} to keep`;
      $("#bulk-sub").textContent = (bi.rest.length ? `${fmtN(bi.rest.length)} will be marked for delete` : "nothing else to delete") +
        (bi.rest.length && bi.bestKept ? ", best shot kept" : "");
      $("#bulk-best").disabled = !(bi.picked.length === 1 && !bi.picked[0].is_best_shot);
      $("#bulk-rest").disabled = !bi.rest.length;
      $("#bulk-rest-n").textContent = fmtN(bi.rest.length);
    } else {
      $("#bulk-count").textContent = `${fmtN(n)} selected`;
      $("#bulk-sub").textContent = pickSummary();
    }
  }

  function syncPeopleSel() {
    $$("#peoplegrid .people-card").forEach((c) => {
      const on = state.peoplePicks.has(num(c.dataset.personId));
      c.classList.toggle("picked", on);
      const b = $(".people-card-select", c);
      if (b) b.setAttribute("aria-pressed", String(on));
    });
  }

  // Name wins when exactly one is named; otherwise the biggest cluster (ties: lowest id).
  function pickMergeTarget(list) {
    const named = list.filter((p) => p.name);
    const pool = named.length === 1 ? named : list;
    return pool.slice().sort((a, b) => (b.face_count - a.face_count) || (a.id - b.id))[0];
  }
  window.pickMergeTarget = pickMergeTarget;

  function selectedPeople() {
    return $$("#peoplegrid .people-card").filter((c) => state.peoplePicks.has(num(c.dataset.personId))).map((c) => ({
      id: num(c.dataset.personId),
      name: $(".people-name-pill-wrapper", c).dataset.currentName || "",
      face_count: num(c.dataset.faceCount),
      card: c,
    }));
  }

  async function bulkMergePeople() {
    const sel = selectedPeople();
    if (sel.length < 2) return;
    const target = pickMergeTarget(sel);
    const tname = target.name || `Person #${target.id}`;
    const ok = await showConfirmDialog({
      title: `Merge ${fmtN(sel.length)} people?`,
      message: `They will become one person named "${tname}". Photos stay in your library.`,
      confirmText: "Merge", variant: "primary",
    });
    if (!ok) return;
    let added = 0;
    try {
      for (const src of sel.filter((p) => p !== target)) {
        await post(`/api/people/${src.id}/merge-into/${target.id}`, {});
        added += src.face_count;
        state.peoplePicks.delete(src.id);
        await rippleDeleteCard(src.card, src.id);
      }
      toast(`Merged into ${tname} ✓`, "ok");
    } catch (e) {
      toast(`Merge failed: ${e.message}`, "err");
    }
    if (added && target.card.isConnected) {
      const n = target.face_count + added;
      target.card.dataset.faceCount = n;
      const badge = $(".people-card-count-badge", target.card);
      if (badge) badge.textContent = fmtN(n);
    }
    state.peoplePicks.clear();
    applyPicks();
  }

  async function bulkDeletePeople() {
    const sel = selectedPeople();
    if (!sel.length) return;
    const who = sel.length === 1 ? `"${sel[0].name || `Person #${sel[0].id}`}"` : `${fmtN(sel.length)} people`;
    const ok = await showConfirmDialog({
      title: sel.length === 1 ? "Delete person cluster?" : "Delete person clusters?",
      message: `Delete cluster for ${who}? Photos will remain in your library, but face assignments will be cleared.`,
      confirmText: "Delete", variant: "danger",
    });
    if (!ok) return;
    try {
      for (const p of sel) {
        await api(`/api/people/${p.id}`, { method: "DELETE" });
        state.peoplePicks.delete(p.id);
        await rippleDeleteCard(p.card, p.id);
      }
      toast(sel.length === 1 ? "Person deleted" : "People deleted", "ok");
    } catch (e) {
      toast(`Delete failed: ${e.message}`, "err");
    }
    state.peoplePicks.clear();
    applyPicks();
  }

  // In a burst, the selection means "frames to keep"; the best shot is never deleted by "delete the rest".
  function burstPickInfo() {
    const d = state.burst;
    if (state.tab !== "bursts" || !d || !d.members || !state.picks.size) return null;
    const picked = d.members.filter((m) => state.picks.has(num(m.id)));
    if (!picked.length) return null;
    const rest = d.members.filter((m) => !state.picks.has(num(m.id)) && !m.is_best_shot && !m.delete_approved);
    const bestKept = d.members.some((m) => m.is_best_shot && !state.picks.has(num(m.id)));
    return { picked, rest, bestKept };
  }

  function pickSummary() {
    const cats = new Set();
    let unseen = false;
    state.picks.forEach((id) => {
      const it = state.byId[id];
      if (it) cats.add(eff(it)); else unseen = true;
    });
    if (cats.size === 1 && !unseen) return catLabel([...cats][0]);
    if (cats.size) return `${cats.size}${unseen ? "+" : ""} categories`;
    return "";
  }

  function clearPicks() { state.picks.clear(); state.peoplePicks.clear(); state.lastPick = null; applyPicks(); }

  async function togglePick(id, shift) {
    const card = $(`.card[data-id="${id}"]`);
    const g = gridOf(card);
    const box = g || (card ? card.parentElement : null);
    if (g) g.vg.setCursor(id);
    if (shift && state.lastPick && box && state.lastPick.box === box) {
      let ids = [];
      if (g) {
        const a = g.vg.indexOfId(state.lastPick.id), b = g.vg.indexOfId(id);
        if (a >= 0 && b >= 0) {
          const lo = Math.min(a, b), hi = Math.max(a, b);
          await g.vg.ensureLoaded(lo, hi + 1);  // the range may span pages that are not loaded yet
          ids = g.vg.items.slice(lo, hi + 1).filter(Boolean).map((x) => x.id);
        }
      } else {
        const all = $$(".card", box).map((c) => num(c.dataset.id));
        const a = all.indexOf(state.lastPick.id), b = all.indexOf(id);
        if (a >= 0 && b >= 0) ids = all.slice(Math.min(a, b), Math.max(a, b) + 1);
      }
      ids.forEach((i) => state.picks.add(i));
    } else if (state.picks.has(id)) {
      state.picks.delete(id);
    } else {
      state.picks.add(id);
    }
    state.lastPick = { id, box };
    applyPicks();
  }

  async function bulkReassign(bucket) {
    const ids = Array.from(state.picks);
    const moved = [];
    for (const id of ids) {
      try {
        const prev = await reassign(id, bucket);
        state.undo.push({ id, prev });
        moved.push({ id, prev });
      } catch (e) { toast(e.message, "err"); break; }
    }
    if (moved.length) {
      const undo = async () => {
        for (const m of moved) await reassign(m.id, m.prev);
        toast(`Moved ${fmtN(moved.length)} back`, "ok");
        resetVisibleGrids();
        loadStats();
      };
      toast(`Moved ${fmtN(moved.length)} photo${moved.length === 1 ? "" : "s"} to ${catLabel(bucket)}`, "ok", { label: "Undo", fn: undo });
    }
    clearPicks();
    loadStats();
  }

  const bulkJunk = () => bulkReassign("accidental");

  async function bulkMarkDelete() {
    const ids = Array.from(state.picks);
    try {
      await post("/api/deletes/approve", { ids, approved: true });
      const undo = async () => {
        await post("/api/deletes/approve", { ids, approved: false });
        resetVisibleGrids();
      };
      toast(`Marked ${fmtN(ids.length)} photo${ids.length === 1 ? "" : "s"} for delete. They will be left out of the result.`, "ok", { label: "Undo", fn: undo });
      clearPicks();
      resetVisibleGrids();
    } catch (e) { toast(e.message, "err"); }
  }

  async function bulkLock() {
    const ids = Array.from(state.picks);
    for (const id of ids) {
      try { await setLock(id, true); } catch (e) { toast(e.message, "err"); break; }
    }
    toast(`Kept ${fmtN(ids.length)} photo${ids.length === 1 ? "" : "s"}. Re-analysis will not change them.`, "ok");
    clearPicks();
  }

  // ------------------------------------------------------------ clean export
  const PII_WORDS = { license_plate: "license plates", payment_card: "payment cards",
    screen: "screens", house_number: "house numbers", id_document: "ID documents" };
  const ce = { scanned: false, unavailable: false };

  function openCleanExport() {
    const n = state.picks.size;
    if (!n) return;
    ce.scanned = false;
    ce.unavailable = false;
    $("#ce-title").textContent = `Clean export of ${fmtN(n)} photo${n === 1 ? "" : "s"}`;
    $("#ce-flagged").replaceChildren();
    $("#ce-status").textContent = "";
    $("#ce-policy").value = "blur";
    openOverlay($("#cleanexport"));
    $("#ce-dest").focus();
  }

  async function ceScan() {
    const btn = $("#ce-scan"), out = $("#ce-scan-result"), list = $("#ce-flagged");
    btn.disabled = true;
    out.textContent = "Checking...";
    list.replaceChildren();
    try {
      const { job_id } = await post("/api/export/clean/scan", { ids: Array.from(state.picks) });
      const r = await followJob(job_id, "Privacy scan");
      if (r.unavailable) {
        ce.unavailable = true;
        $("#ce-policy").value = "include";
        out.textContent = r.unavailable + " You can still export without this check.";
        return;
      }
      ce.scanned = true;
      const counts = {};
      r.flagged.forEach((f) => f.labels.forEach((l) => { counts[l] = (counts[l] || 0) + 1; }));
      const kinds = Object.entries(counts).map(([l, n]) => `${fmtN(n)} with ${PII_WORDS[l] || l}`).join(", ");
      out.textContent = r.flagged.length
        ? `${fmtN(r.flagged.length)} of ${fmtN(r.scanned)} photos may show sensitive details: ${kinds}.`
        : `Nothing sensitive found in ${fmtN(r.scanned)} photos. Small details can still be missed.`;
      r.flagged.slice(0, 20).forEach((f) => list.append(el("li", {
        text: `${f.filename}: ${f.labels.map((l) => PII_WORDS[l] || l).join(", ")}` })));
    } catch (e) {
      out.textContent = "The check failed: " + e.message;
    } finally { btn.disabled = false; }
  }

  async function ceExport() {
    const dest = $("#ce-dest").value.trim();
    const policy = $("#ce-policy").value;
    if (!dest) { toast("Choose a destination folder first", "err"); return; }
    if (policy !== "include" && !ce.scanned) {
      toast("Check for sensitive details first, or choose to export photos as they are", "err");
      return;
    }
    const go = $("#ce-go");
    go.disabled = true;
    $("#ce-status").textContent = "Exporting...";
    try {
      const { job_id } = await post("/api/export/clean", { ids: Array.from(state.picks), output_dir: dest,
        policy, html: $("#ce-html").checked, rename: $("#ce-rename").checked });
      const r = await followJob(job_id, "Clean export");
      const bits = [`${fmtN(r.exported)} exported`];
      if (r.blurred) bits.push(`${fmtN(r.blurred)} blurred`);
      if (r.skipped_flagged) bits.push(`${fmtN(r.skipped_flagged)} left out`);
      toast(`Clean export done: ${bits.join(", ")}`, "ok");
      libLog(`Clean export saved to ${r.folder} (${bits.join(", ")})`, r.failures.length ? "warn" : "ok");
      closeOverlay($("#cleanexport"));
    } catch (e) {
      toast(e.message, "err");
      $("#ce-status").textContent = "";
    } finally { go.disabled = false; }
  }

  // --------------------------------------------------- quick category picker
  let catPicker = null;

  function closeCatPicker(restoreFocus = true) {
    if (!catPicker) return;
    const { node, anchor, off } = catPicker;
    catPicker = null;
    off();
    node.remove();
    if (restoreFocus && document.contains(anchor)) anchor.focus({ preventScroll: true });
  }

  async function pickCategory(id, bucket, current) {
    closeCatPicker();
    if (bucket === current && !state.picks.has(id)) return;
    try {
      if (state.picks.has(id)) { await bulkReassign(bucket); return; }
      const prev = await reassign(id, bucket);
      state.undo.push({ id, prev });
      toast("Moved to " + catLabel(bucket), "ok");
      loadStats();
    } catch (e) { toast(e.message, "err"); }
  }

  function openCatPicker(anchor, id) {
    if (catPicker && catPicker.anchor === anchor) { closeCatPicker(); return; }
    closeCatPicker(false);
    const current = eff(state.byId[id] || {});
    const many = state.picks.has(id) && state.picks.size > 1;
    const items = state.cats.map((c) => el("button", {
      class: "catmenu-item" + (c === current ? " current" : ""), type: "button", role: "option",
      "aria-selected": String(c === current), "data-catitem": c, style: catStyle(c),
    }, el("i"), el("span", { text: catLabel(c) })));
    const node = el("div", { class: "catmenu", role: "listbox", "aria-label": "Move to category" },
      el("div", { class: "catmenu-head", text: many ? `Move ${fmtN(state.picks.size)} selected` : "Move to" }), items);
    document.body.append(node);

    const r = anchor.getBoundingClientRect();
    const w = node.offsetWidth, h = node.offsetHeight;
    const below = window.innerHeight - r.bottom - 8, above = r.top - 8;
    const top = below >= Math.min(h, 240) || below >= above ? r.bottom + 4 : Math.max(8, r.top - h - 4);
    node.style.top = Math.max(8, Math.min(top, window.innerHeight - h - 8)) + "px";
    node.style.left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8)) + "px";

    items.forEach((b) => { b.onclick = () => pickCategory(id, b.dataset.catitem, current); });
    node.addEventListener("keydown", (ev) => {
      ev.stopPropagation();  // keep rating and grid shortcuts out while the menu is open
      const i = items.indexOf(document.activeElement);
      if (ev.key === "Escape") { ev.preventDefault(); closeCatPicker(); }
      else if (ev.key === "Tab") closeCatPicker(false);
      else if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
        ev.preventDefault();
        const d = ev.key === "ArrowDown" ? 1 : -1;
        items[(i + d + items.length) % items.length].focus();
      }
    });
    const onDown = (ev) => { if (!node.contains(ev.target) && !anchor.contains(ev.target)) closeCatPicker(false); };
    const onAway = (ev) => { if (!(ev.target instanceof Node && node.contains(ev.target))) closeCatPicker(false); };
    document.addEventListener("mousedown", onDown, true);
    window.addEventListener("scroll", onAway, true);
    window.addEventListener("resize", onAway);
    const off = () => {
      document.removeEventListener("mousedown", onDown, true);
      window.removeEventListener("scroll", onAway, true);
      window.removeEventListener("resize", onAway);
    };
    catPicker = { node, anchor, off };
    const toFocus = items.find((b) => b.classList.contains("current")) || items[0];
    if (toFocus) toFocus.focus({ preventScroll: true });
  }

  // ----------------------------------------------------------- card actions
  async function reassign(id, bucket) {
    const prev = (state.byId[id] || {}).manual_bucket || null;
    await post(`/api/images/${num(id)}/reassign`, { bucket });
    const cur = state.byId[id] || { id };
    itemUpdated(Object.assign({}, cur, { manual_bucket: bucket }));
    return prev;
  }

  async function setLock(id, locked) {
    const cur = state.byId[id];
    const r = await post(`/api/images/${num(id)}/lock`, { locked });
    itemUpdated(Object.assign({}, cur || { id }, { locked: r.locked }));
    return r.locked;
  }

  async function toggleLock(id) {
    const cur = state.byId[id];
    const now = await setLock(id, !(cur && cur.locked));
    toast(now ? "Locked. Re-analysis will not change it." : "Unlocked", "ok");
  }

  async function showSimilar(id) {
    const panel = $("#similar-panel");
    const it = state.byId[id];
    $("#similar-title").textContent = "Photos similar to " + (it ? it.filename : "#" + num(id));
    const sg = $("#similar-grid");
    sg.replaceChildren(...Array.from({ length: 8 }, () => el("div", { class: "skel" })));
    openOverlay($("#similar-panel"));
    try {
      const data = await api(`/api/images/${num(id)}/similar?k=24`);
      data.items.forEach((x) => { state.byId[x.id] = x; });
      sg.replaceChildren(...(data.items.length
        ? data.items.map(cardNode)
        : [emptyNode({ icon: "copy", title: "No similar photos", text: "Analyze the library first, then try again." })]));
    } catch (e) {
      sg.replaceChildren(emptyNode({ icon: "alert", title: "Search failed", text: e.message }));
    }
  }

  // ------------------------------------------------------------- highlights
  const FEAT_OUT = { highlight_rating_weight: "#hl-o-rating", highlight_favorite_weight: "#hl-o-fav",
    highlight_preference_weight: "#hl-o-taste" };

  async function loadFeatures() {
    const data = await api("/api/features");
    state.features = data.features;
    $$("[data-feat]").forEach((inp) => {
      inp.value = data.features[inp.dataset.feat];
      $(FEAT_OUT[inp.dataset.feat]).textContent = num(inp.value).toFixed(2);
    });
  }

  async function refreshTaste() {
    const st = await api("/api/preferences");
    const t = $("#taste-text");
    const liked = `${fmtN(st.n_positive)} liked and ${fmtN(st.n_negative)} disliked`;
    if (st.trained) {
      t.textContent = `Learned from ${fmtN(st.model_positive)} liked and ${fmtN(st.model_negative)} disliked photos, ` +
        `about ${Math.round(num(st.cv_accuracy) * 100)}% right on photos it had not seen. ` +
        `${fmtN(st.scored)} photos have a taste score. Rate more photos, then learn again to refine it.`;
    } else if (st.can_train) {
      t.textContent = `Ready to learn from ${liked} photos.`;
    } else {
      t.textContent = `So far ${liked} photos. Photocure needs at least ${st.min_per_class} of each. ` +
        "Favourites and four or five stars count as liked, rejects and one or two stars as disliked.";
    }
    $("#btn-taste-learn").disabled = !st.can_train;
    $("#btn-taste-reset").disabled = !st.trained;
  }

  async function loadHighlightYears() {
    const data = await api("/api/highlights/years");
    const sel = $("#hl-year");
    const cur = sel.value;
    fillSelect(sel, "Year in review", data.years.map((y) => ({ value: String(y.year), text: `${y.year} (${fmtN(y.count)})` })));
    if (data.years.some((y) => String(y.year) === cur)) sel.value = cur;
    $("#btn-hl-export").disabled = !sel.value;
  }

  const SCALE_NAMES = { wide: "wide", closeup: "close-up", action: "action", ambient: "ambient", medium: "medium" };
  const SCALE_BADGES = { wide: "Wide", closeup: "Close-up", action: "Action", ambient: "Ambient", medium: "Medium" };
  const isEventView = () => ["events", "story"].includes($("#hl-view").value);

  // "1 wide, 2 close-up": how many photos of each shot scale a story holds, in story order.
  function coverageText(cov) {
    return Object.keys(SCALE_NAMES).filter((s) => cov && cov[s]).map((s) => `${cov[s]} ${SCALE_NAMES[s]}`).join(", ");
  }

  async function renderEventHighlights() {
    const host = $("#hl-events");
    const m = $("#minaes").value;
    const story = $("#hl-view").value === "story";
    host.replaceChildren(...Array.from({ length: 6 }, () => el("div", { class: "skel" })));
    try {
      const data = await api(`/api/highlights/events?per_event=${story ? 8 : 3}` + (story ? "&balanced=true" : "") +
        (m !== "" ? "&min_score=" + encodeURIComponent(m) : ""));
      $("#highlightcount").textContent = `${fmtN(data.events.length)} event${data.events.length === 1 ? "" : "s"}`;
      if (!data.events.length) {
        host.replaceChildren(emptyNode({ icon: "cal", title: "No events with scored photos",
          text: "Find events first, then analyze your photos so they can be scored." }));
        return;
      }
      const blocks = data.events.map((ev) => {
        ev.items.forEach((x) => { state.byId[x.id] = x; });
        const when = ev.start ? String(ev.start).slice(0, 10) : "";
        const cards = ev.items.map((x) => {
          const node = cardNode(x);
          const flags = $(".flags", node);
          if (x.scale && flags) flags.append(el("span", { class: "badge scale", "data-scale": x.scale, text: SCALE_BADGES[x.scale] || x.scale }));
          return node;
        });
        const cover = ev.coverage ? coverageText(ev.coverage) : "";
        return el("div", { class: "ev-block" },
          el("div", { class: "ev-head" }, el("h3", { text: ev.label }),
            el("span", { text: [when, `${story && data.balanced ? "" : "top "}${ev.count} of ${fmtN(ev.total)}`].filter(Boolean).join(", ") })),
          cover ? el("div", { class: "ev-cover" }, el("span", { class: "scale-chip", text: "Covers " + cover })) : null,
          el("div", { class: "grid" }, cards));
      });
      if (story && !data.balanced && data.note) blocks.unshift(el("p", { class: "help", role: "status", text: data.note }));
      host.replaceChildren(...blocks);
    } catch (e) {
      host.replaceChildren(emptyNode({ icon: "alert", title: "Could not load highlights", text: e.message }));
    }
  }

  function applyHighlightView() {
    const events = isEventView();
    $("#highlightgrid").hidden = events;
    $("#hl-events").hidden = !events;
    if (events) renderEventHighlights(); else grids.highlights.reset();
  }

  async function exportHighlights() {
    const year = $("#hl-year").value;
    if (!year) return;
    const a = el("a", { href: `/api/highlights/export?year=${encodeURIComponent(year)}&n=50`, download: `highlights-${year}.html` });
    document.body.append(a);
    a.click();
    a.remove();
  }

  async function learnTaste() {
    const btn = $("#btn-taste-learn");
    btn.disabled = true;
    try {
      const r = await post("/api/preferences/learn");
      if (r.status === "trained") toast(`Learned your taste, about ${Math.round(num(r.cv_accuracy) * 100)}% right on new photos`, "ok");
      else if (r.status === "weak") toast("Your ratings do not show a clear pattern yet, so nothing changed.", "warn");
      else toast(`Rate at least ${r.preferences.min_per_class} liked and ${r.preferences.min_per_class} disliked photos first.`, "warn");
      await refreshTaste();
      applyHighlightView();
    } catch (e) { toast(e.message, "err"); await refreshTaste().catch(() => {}); }
  }

  async function forgetTaste() {
    const ok = await showConfirmDialog({ title: "Forget your taste?", confirmText: "Forget",
      cancelText: "Keep", message: "Highlights go back to ranking by the AI's score and your stars. Your ratings are not touched." });
    if (!ok) return;
    try {
      await post("/api/preferences/reset");
      await refreshTaste();
      applyHighlightView();
    } catch (e) { toast(e.message, "err"); }
  }

  function bindHighlights() {
    $$("[data-feat]").forEach((inp) => {
      inp.addEventListener("input", () => { $(FEAT_OUT[inp.dataset.feat]).textContent = num(inp.value).toFixed(2); });
      inp.addEventListener("change", async () => {
        try {
          const data = await post("/api/features", { [inp.dataset.feat]: num(inp.value) });
          state.features = data.features;
          applyHighlightView();
        } catch (e) { toast(e.message, "err"); }
      });
    });
    $("#hl-view").addEventListener("change", applyHighlightView);
    $("#hl-year").addEventListener("change", () => { $("#btn-hl-export").disabled = !$("#hl-year").value; });
    $("#btn-hl-export").onclick = exportHighlights;
    $("#btn-taste-learn").onclick = learnTaste;
    $("#btn-taste-reset").onclick = forgetTaste;
  }

  // ------------------------------------------------------------------- watch
  const WATCH_STATE = { idle: "Watching", scanning: "Looking for new photos in", analyzing: "Analyzing new photos in", waiting: "Waiting for the current task, then continuing in", stopping: "Stopping", stopped: "Not watching" };
  let watchTimer = null;
  let watchSeen = null;

  function setText(node, text) { if (node && node.textContent !== text) node.textContent = text; }

  function renderWatch(st) {
    const box = $("#watch-toggle");
    if (box.checked !== !!st.running) box.checked = !!st.running;
    if (st.running && st.path && !$("#scan-path").value) $("#scan-path").value = st.path;
    let text = "";
    if (st.running) {
      text = `${WATCH_STATE[st.state] || "Watching"} ${st.path}` +
        (st.photos_added ? `. ${fmtN(st.photos_added)} new photo${st.photos_added === 1 ? "" : "s"} so far.` : ".");
      if (st.last_error) text += ` Last problem: ${st.last_error}`;
    } else if (st.last_error) text = st.last_error;
    setText($("#watch-status"), text);
    if (watchSeen !== null && st.photos_added !== watchSeen) { loadStats(); resetVisibleGrids(); }
    watchSeen = st.running ? st.photos_added : null;
  }

  async function refreshWatch() {
    clearTimeout(watchTimer);
    watchTimer = null;
    let st;
    try { st = await api("/api/watch"); } catch (e) { return; }
    renderWatch(st);
    if (st.running) watchTimer = setTimeout(refreshWatch, 5000);
  }

  function bindWatch() {
    $("#watch-toggle").addEventListener("change", async (e) => {
      const on = e.target.checked;
      try {
        if (on) {
          const path = $("#scan-path").value.trim();
          if (!path) { e.target.checked = false; toast("Choose a folder to watch first.", "warn"); return; }
          renderWatch(await post("/api/watch", { enabled: true, path }));
        } else {
          renderWatch(await post("/api/watch", { enabled: false }));
        }
        refreshWatch();
      } catch (err) { e.target.checked = !on; toast(err.message, "err"); }
    });
  }

  // ---------------------------------------------------------------- keywords
  async function refreshKeywords() {
    let st;
    try { st = await api("/api/keywords"); } catch (e) { return; }
    $("#set-keywords").checked = !!st.enabled;
    ["#set-keywords", "#btn-kw-write", "#btn-kw-remove"].forEach((sel) => { $(sel).disabled = !st.sidecars; });
    $("#kw-note").textContent = st.sidecars
      ? `Writing adds a .xmp file next to any of your ${fmtN(st.photos)} photos that does not have one. Photos whose sidecar another program changed are skipped.`
      : "Sidecar writing is turned off, so keywords cannot be written.";
  }

  async function keywordsRun(remove) {
    const n = (await api("/api/keywords")).photos;
    const ok = await showConfirmDialog({
      title: remove ? "Remove Photocure keywords?" : "Write keywords for every photo?",
      confirmText: remove ? "Remove" : "Write", cancelText: "Cancel",
      message: remove
        ? `Photocure's own keywords are removed from the sidecars of up to ${fmtN(n)} photos. Nothing else in them changes.`
        : `This creates or updates a .xmp sidecar next to up to ${fmtN(n)} photos. Only Photocure's own keywords are added, and your ratings are not touched.` });
    if (!ok) return;
    try {
      const r = await post("/api/keywords/write", { remove });
      logLine(remove ? "Removing keywords from sidecars" : "Writing keywords to sidecars", "ok");
      const res = await followJob(r.job_id, "Keywords");
      const left = num(res && res.edited_elsewhere);
      toast(`${fmtN(res && res.written)} sidecar${num(res && res.written) === 1 ? "" : "s"} updated` +
        (left ? `, ${fmtN(left)} left alone because another program edited them` : ""), "ok");
    } catch (e) { toast(e.message, "err"); }
    pollProgress();
  }

  function bindKeywords() {
    $("#set-keywords").addEventListener("change", async (e) => {
      try { await post("/api/features", { write_keywords: e.target.checked }); } catch (err) {
        e.target.checked = !e.target.checked;
        toast(err.message, "err");
      }
    });
    $("#btn-kw-write").onclick = () => keywordsRun(false);
    $("#btn-kw-remove").onclick = () => keywordsRun(true);
  }

  // -------------------------------------------------------------- duplicates
  const DUP_PAGE = 20;
  const dup = { groups: [], total: 0, pick: {}, busy: false };

  function setDupCount(summary) {
    dupCountAt = Date.now();
    dup.summary = summary;
    const badge = $('[data-count="duplicates"]');
    const text = summary && summary.groups ? fmtN(summary.groups) : "";
    if (badge && badge.textContent !== text) badge.textContent = text;  // no write when nothing changed
  }

  let dupCountAt = 0;

  async function refreshDupCount() {
    if (Date.now() - dupCountAt < 4000) return;  // loadStats runs after every edit; groups take a moment to find
    dupCountAt = Date.now();
    try { setDupCount((await api("/api/duplicates?limit=1")).summary); } catch (e) { /* badge stays as it was */ }
  }

  function dupMemberNode(g, m, chosenId) {
    const it = m.item || { id: m.id, filename: m.filename };
    state.byId[m.id] = it;
    const chosen = m.id === chosenId;
    const stars = m.rating === -1 ? "Rejected" : (m.rating > 0 ? "\u2605".repeat(m.rating) : "");
    const facts = [`${num(m.width)}\u00d7${num(m.height)}`, fmtBytes(m.file_size),
      m.taken_at ? String(m.taken_at).slice(0, 10) : "", stars, m.favorite ? "Favourite" : "",
      g.kind === "near" && m.similarity != null && m.id !== g.keeper_id ? `${Math.round(num(m.similarity) * 100)}% alike` : ""]
      .filter(Boolean).join(", ");
    const img = el("img", { loading: "lazy", decoding: "async", src: thumbUrl(m.id, "medium"), alt: m.filename || "" });
    img.onerror = () => { img.style.opacity = ".15"; };
    const badge = el("span", { class: "dup-badge " + (chosen ? "keep" : "out") },
      chosen ? el("span", { "aria-hidden": "true", text: "\u2713" }) : null,
      el("span", { text: chosen ? (g.reviewed ? "Kept" : "Keep") : "Duplicate" }));
    return el("figure", { class: "dup-member" + (chosen ? " keep" : " out"), "data-mid": m.id },
      el("button", { class: "dup-thumb", type: "button", "data-dup-open": m.id, "aria-label": "Open " + (m.filename || "photo") }, img, badge),
      el("figcaption", {},
        el("div", { class: "fn", title: m.path, text: m.filename }),
        el("div", { class: "facts", text: facts }),
        m.suggested ? el("span", { class: "tag sug", text: "Suggested" }) : null,
        g.reviewed || chosen ? null : el("button", { class: "btn sm ghost", type: "button",
          "data-dup-keep": m.id, text: "Keep this one instead" })));
  }

  function dupGroupNode(g) {
    const chosen = dup.pick[g.id] != null && g.members.some((m) => m.id === dup.pick[g.id]) ? dup.pick[g.id] : g.keeper_id;
    const others = g.members.filter((m) => m.id !== chosen);
    const keeper = g.members.find((m) => m.id === chosen);
    const frees = others.reduce((a, m) => a + num(m.file_size), 0);
    const title = g.kind === "exact" ? `${g.members.length} identical files` : `${g.members.length} look-alike photos`;
    const foot = g.reviewed
      ? el("div", { class: "dup-foot" }, el("span", { class: "done", text: "Decided. The others will be treated as duplicates when you apply a plan." }))
      : el("div", { class: "dup-foot" },
        el("p", { class: "dup-plan", text: `Keep ${keeper ? keeper.filename : "this photo"}, mark ${others.length} other${others.length === 1 ? "" : "s"} as a duplicate${others.length === 1 ? "" : "s"}.` }),
        el("div", { class: "dup-actions" },
          g.kind === "near" ? el("button", { class: "btn ghost", type: "button", "data-dup-dismiss": g.id, text: "Not duplicates" }) : null,
          el("button", { class: "btn commit", type: "button", "data-dup-apply": g.id, text: "Confirm" })));
    return el("article", { class: "dup-group", "data-gid": g.id },
      el("header", { class: "dup-head" }, el("h3", { text: title }),
        el("span", { class: "saves", text: `Frees ${fmtBytes(frees)}` })),
      el("div", { class: "dup-row" }, g.members.map((m) => dupMemberNode(g, m, chosen))),
      foot);
  }

  function dupSummaryText() {
    const sm = dup.summary;
    $("#dup-summary").textContent = sm && sm.groups
      ? `${fmtN(sm.groups)} group${sm.groups === 1 ? "" : "s"}, ${fmtN(sm.images)} photos, up to ${fmtBytes(sm.reclaimable_bytes)} to free`
      : "";
    $("#btn-dup-all").disabled = !(sm && sm.groups) || $("#dup-reviewed").checked;
  }

  async function loadDuplicates(append) {
    const host = $("#dup-list");
    if (!append) {
      dup.groups = [];
      host.replaceChildren(...Array.from({ length: 3 }, () => el("div", { class: "skel", style: "height:220px" })));
    }
    try {
      const kind = $("#dup-kind").value;
      const qs = new URLSearchParams({ limit: DUP_PAGE, offset: dup.groups.length });
      if (kind) qs.set("kind", kind);
      if ($("#dup-reviewed").checked) qs.set("reviewed", "true");
      const data = await api("/api/duplicates?" + qs);
      dup.total = data.total;
      dup.groups = dup.groups.concat(data.groups);
      setDupCount(data.summary);
      dupSummaryText();
      if (!append) host.replaceChildren();
      if (!dup.groups.length) {
        host.replaceChildren(emptyNode({ icon: "check", title: "No duplicates to review",
          text: "Identical and look-alike photos appear here after you analyze your library." }));
      } else {
        data.groups.forEach((g) => host.append(dupGroupNode(g)));
      }
      $("#btn-dup-more").hidden = dup.groups.length >= dup.total;
    } catch (e) {
      host.replaceChildren(emptyNode({ icon: "alert", title: "Could not load duplicates", text: e.message }));
    }
  }

  function dupFinish(res, msg) {
    setDupCount(res.summary);
    toast(msg, "ok");
    loadDuplicates(false);
  }

  async function dupApply(gid) {
    const g = dup.groups.find((x) => x.id === gid);
    if (!g || dup.busy) return;
    const chosen = dup.pick[gid] != null ? dup.pick[gid] : g.keeper_id;
    const discard = g.members.filter((m) => m.id !== chosen).map((m) => m.id);
    dup.busy = true;
    try {
      const res = await post("/api/duplicates/resolve", { keep_id: chosen, discard_ids: discard, kind: g.kind });
      dupFinish(res, `Kept 1 photo and marked ${discard.length} as duplicates. Nothing moves until you apply a plan.`);
    } catch (e) { toast(e.message, "err"); } finally { dup.busy = false; }
  }

  async function dupDismiss(gid) {
    const g = dup.groups.find((x) => x.id === gid);
    if (!g || dup.busy) return;
    dup.busy = true;
    try {
      const res = await post("/api/duplicates/dismiss", { ids: g.members.map((m) => m.id) });
      dupFinish(res, "Marked as different photos.");
    } catch (e) { toast(e.message, "err"); } finally { dup.busy = false; }
  }

  async function dupResolveAll() {
    const n = dup.summary ? dup.summary.groups : 0;
    const ok = await showConfirmDialog({ title: "Accept the suggested photo in every group?",
      confirmText: "Accept all", cancelText: "Cancel",
      message: `${fmtN(n)} group${n === 1 ? "" : "s"} will be resolved. In each, the suggested photo stays and the rest are marked as duplicates. ` +
        "Nothing is moved or deleted until you apply a plan, and you can change any photo's category afterwards." });
    if (!ok) return;
    try {
      const res = await post("/api/duplicates/resolve-all", {});
      dupFinish(res, `Resolved ${fmtN(res.result.groups)} groups, ${fmtN(res.result.discarded)} photos marked as duplicates.`);
    } catch (e) { toast(e.message, "err"); }
  }

  let activityBound = false;
  function bindActivity() {
    if (activityBound) return;
    activityBound = true;
    document.addEventListener("click", (ev) => {
      const filter = ev.target.closest && ev.target.closest("[data-act-filter]");
      if (filter) {
        const panel = filter.closest(".activity");
        panel.dataset.filter = filter.dataset.actFilter;
        $$("[data-act-filter]", panel).forEach((b) => b.classList.toggle("active", b === filter));
        return;
      }
      const clear = ev.target.closest && ev.target.closest("[data-act-clear]");
      if (clear) {
        const box = $(".action-log", clear.closest(".activity"));
        const empty = document.createElement("div");
        empty.className = "logline logline-empty";
        empty.textContent = "Nothing has run yet.";
        box.replaceChildren(empty);
        updateActivityCount(box);
      }
    });
  }

  function bindDuplicates() {
    $("#dup-kind").addEventListener("change", () => loadDuplicates(false));
    $("#dup-reviewed").addEventListener("change", () => loadDuplicates(false));
    $("#btn-dup-more").onclick = () => loadDuplicates(true);
    $("#btn-dup-all").onclick = dupResolveAll;
    $("#dup-list").addEventListener("click", (ev) => {
      const t = ev.target.closest("[data-dup-keep],[data-dup-open],[data-dup-apply],[data-dup-dismiss]");
      if (!t) return;
      const node = t.closest(".dup-group");
      const gid = node && node.dataset.gid;
      const g = dup.groups.find((x) => x.id === gid);
      if (!g) return;
      if (t.dataset.dupKeep) {
        dup.pick[gid] = num(t.dataset.dupKeep);
        node.replaceWith(dupGroupNode(g));
      } else if (t.dataset.dupOpen) {
        openLightboxList(g.members.map((m) => m.id), num(t.dataset.dupOpen), t);
      } else if (t.dataset.dupApply) dupApply(gid);
      else if (t.dataset.dupDismiss) dupDismiss(gid);
    });
  }

  // ------------------------------------------------------------------ search
  let searchSeq = 0;

  async function runSearch(weak = false) {
    const q = $("#searchq").value.trim();
    const host = $("#searchgrid");
    const count = $("#searchcount");
    if (!q) {
      count.textContent = "";
      host.replaceChildren(emptyNode({ icon: "search", title: "Describe the photo",
        text: "Try dog on the beach, birthday cake, or a red car at night." }));
      return;
    }
    const seq = ++searchSeq;
    count.textContent = "Searching...";
    host.replaceChildren(...Array.from({ length: 12 }, () => el("div", { class: "skel" })));
    try {
      const dups = $("#search-dups").checked ? "&duplicates=true" : "";
      const data = await api(`/api/search?q=${encodeURIComponent(q)}&k=120${dups}${weak === true ? "&strict=false" : ""}`);
      if (seq !== searchSeq) return;
      data.items.forEach((x) => { state.byId[x.id] = x; });
      const n = data.items.length;
      const parts = data.parts && data.parts.length ? ` with ${data.parts.join(" and ")}` : "";
      const note = data.more > 0 ? ", best matches only" : (weak === true ? ", including weaker matches" : "");
      count.textContent = `${fmtN(n)} photo${n === 1 ? "" : "s"}${parts}${note}`;
      const nodes = n ? data.items.map(cardNode)
        : [emptyNode({ icon: "search", title: "Nothing matched",
          text: "Try other words, or analyze your photos first so they can be searched." })];
      if (data.more > 0) {
        const more = el("button", { class: "btn ghost search-more", type: "button",
          text: `Show ${fmtN(data.more)} weaker match${data.more === 1 ? "" : "es"}` });
        more.onclick = () => runSearch(true);
        nodes.push(more);
      }
      host.replaceChildren(...nodes);
    } catch (e) {
      if (seq !== searchSeq) return;
      count.textContent = "";
      host.replaceChildren(emptyNode({ icon: "alert",
        title: e.status === 503 ? "Search needs the AI model" : "Search failed", text: e.message }));
    }
  }

  function setCategoryFilter(cat) {
    const cur = ($("#catfilter") || {}).value || "";
    const next = (cur === cat && cat) ? "" : (cat || "");
    $("#catfilter").value = next;
    updateResetFiltersButton();
    showTab("categories", true);
    grids.cat.reset();
    if (state.overview) renderComp(state.overview);
  }

  let delegatesBound = false;
  function bindDelegates() {
    if (delegatesBound) return;
    delegatesBound = true;
    document.addEventListener("click", async (ev) => {
      const t = ev.target.closest && ev.target.closest(
        "[data-pick],[data-catpick],[data-clear],[data-lock],[data-approve-delete],[data-similar],[data-faces],[data-more],[data-goto],[data-clearfilters],[data-cat],[data-sortnow],[data-copy],[data-copytext],[data-reanalyze],.thumb");
      if (!t) return;
      try {
        if (t.dataset.pick) await togglePick(num(t.dataset.pick), ev.shiftKey);
        else if (t.dataset.catpick) openCatPicker(t, num(t.dataset.catpick));
        else if (t.dataset.clear) {
          const id = num(t.dataset.clear);
          const prev = await reassign(id, null);
          state.undo.push({ id, prev });
          toast("Back to the AI choice", "ok");
        } else if (t.dataset.lock) await toggleLock(num(t.dataset.lock));
        else if (t.dataset.approveDelete) await toggleApproveDelete(num(t.dataset.approveDelete));
        else if (t.dataset.similar) {
          closeLightbox();
          await showSimilar(num(t.dataset.similar));
        } else if (t.dataset.faces) {
          await openBiometricProjection(t.dataset.burst ? num(t.dataset.burst) : null, num(t.dataset.faces));
        } else if (t.dataset.more) {
          const g = grids[t.dataset.more];
          if (g) await g.loadMore();
        } else if (t.dataset.goto) { ev.preventDefault(); gotoTarget(t); }
        else if (t.hasAttribute("data-clearfilters")) {
          resetAllFilters();
        } else if (t.hasAttribute("data-copytext")) {
          ev.preventDefault();
          await copyText(t.dataset.copytext, t);
        } else if (t.hasAttribute("data-copy")) await copyHint(t);
        else if (t.hasAttribute("data-reanalyze")) $("#btn-infer").click();
        else if (t.hasAttribute("data-sortnow")) await sortNow(false);
        else if (t.dataset.cat !== undefined) setCategoryFilter(t.dataset.cat);
        else if (t.classList.contains("thumb")) {
          const card = t.closest(".card");
          if (card) openLightbox(card);
        }
      } catch (e) { toast(e.message, "err"); }
    });
    // +N flags tooltip: a grouped card anchored under the pill (hover on mouse, tap on touch)
    (function () {
      const tip = document.getElementById("card-tags-tooltip");
      const body = document.getElementById("card-tags-tooltip-list");
      if (!tip || !body) return;
      const GROUPS = ["Rating", "Quality", "Status"];
      let curPill = null;

      function build(tags) {
        body.textContent = "";
        GROUPS.forEach((g) => {
          const items = tags.filter((t) => t.g === g);
          if (!items.length) return;
          body.append(el("div", { class: "tt-group" },
            el("em", { text: g }),
            el("div", { class: "tt-chips" }, items.map((t) =>
              el("span", { class: "tt-chip" + (t.k ? " " + t.k : "") }, el("u", { text: t.ic }), t.text)))));
        });
      }

      function show(pill) {
        let tags = [];
        try { tags = JSON.parse(pill.dataset.overflowTags || "[]"); } catch (_) {}
        if (!tags.length) return;
        curPill = pill;
        build(tags);
        const GAP = 9, M = 8;
        const r = pill.getBoundingClientRect();
        const w = tip.offsetWidth, h = tip.offsetHeight;
        const left = Math.max(M, Math.min(r.left + r.width / 2 - w / 2, window.innerWidth - w - M));
        const above = r.bottom + GAP + h > window.innerHeight - M && r.top - GAP - h > M;
        const ax = (r.left + r.width / 2 - left) + "px";
        tip.style.left = left + "px";
        tip.style.top = (above ? r.top - GAP - h : r.bottom + GAP) + "px";
        tip.style.setProperty("--ax", ax);
        tip.classList.toggle("above", above);
        tip.classList.add("open");
        tip.setAttribute("aria-hidden", "false");
      }

      function hide() {
        tip.classList.remove("open");
        tip.setAttribute("aria-hidden", "true");
        curPill = null;
      }

      document.addEventListener("mouseover", (ev) => {
        const pill = ev.target.closest && ev.target.closest(".tag-overflow-pill");
        if (pill && pill !== curPill) show(pill);
      });
      document.addEventListener("mouseout", (ev) => {
        if (!curPill) return;
        const to = ev.relatedTarget;
        if (to && curPill.contains(to)) return;
        if (ev.target.closest && ev.target.closest(".tag-overflow-pill")) hide();
      });
      document.addEventListener("click", (ev) => {
        const pill = ev.target.closest && ev.target.closest(".tag-overflow-pill");
        if (!pill) { if (curPill) hide(); return; }
        if (curPill === pill && ev.pointerType !== "mouse") hide(); else show(pill);
      });
      document.addEventListener("scroll", () => { if (curPill) hide(); }, { capture: true, passive: true });
    })();

    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        const goto = ev.target.closest && ev.target.closest("[data-goto]");
        if (goto && goto.dataset.goto) {
          ev.preventDefault();
          gotoTarget(goto);
          return;
        }
      }
      if ((ev.key === "Enter" || ev.key === " ") && ev.target.classList && ev.target.classList.contains("card")) {
        ev.preventDefault();
        openLightbox(ev.target);
      }
    });
  }

  // ------------------------------------------------------------- lightbox
  const lb = { open: false, ids: [], i: 0, token: 0, grid: null };
  const lbCount = () => (lb.grid ? lb.grid.vg.total || 0 : lb.ids.length);
  const lbIdAt = (i) => (lb.grid ? lb.grid.vg.idAt(i) : lb.ids[i]);

  function openLightbox(card) {
    const g = gridOf(card);
    const id = num(card.dataset.id);
    lb.grid = g;
    if (g) {
      lb.ids = [];
      lb.i = Math.max(0, g.vg.indexOfId(id));
      g.vg.setCursor(id);
    } else {  // a plain list of cards (burst view, similar photos)
      // If card is inside a card-wrap (burst grid), collect all burst frames via the grid parent
      const wrap = card.closest(".card-wrap");
      const listRoot = wrap ? (wrap.closest(".burstgrid") || wrap.parentElement) : card.parentElement;
      lb.ids = $$(".card", listRoot).map((c) => num(c.dataset.id));
      lb.i = Math.max(0, lb.ids.indexOf(id));
    }
    lbShow(card);
  }

  function openLightboxList(ids, id, opener) {
    lb.grid = null;
    lb.ids = ids.slice();
    lb.i = Math.max(0, lb.ids.indexOf(id));
    lbShow(opener);
  }

  function openLightboxFor(g, id) {
    const idx = g.vg.indexOfId(id);
    if (idx < 0) return;
    lb.grid = g;
    lb.ids = [];
    lb.i = idx;
    g.vg.setCursor(id);
    const card = g.vg.host.querySelector(`.card[data-id="${id}"]`);
    lbShow(card || document.activeElement);
  }

  function lbShow(opener) {
    lb.open = true;
    lb.opener = opener || document.activeElement;
    openOverlay($("#lightbox"));
    document.body.style.overflow = "hidden";
    lbRender();
    if (lb.releaseTrap) lb.releaseTrap();
    lb.releaseTrap = trapFocus($("#lightbox"), closeLightbox);
    $("#lb-close").focus();
  }

  async function closeLightbox() {
    if (!lb.open) return;
    lb.open = false;
    if (lb.releaseTrap) {
      lb.releaseTrap();
      lb.releaseTrap = null;
    }
    closeOverlay($("#lightbox"));
    if ($("#person-photos-overlay") && !$("#person-photos-overlay").hidden) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    if (lb.grid) {  // leave the grid on the photo that was last viewed
      const id = lbIdAt(lb.i);
      if (id != null) lb.grid.vg.setCursor(id);
      await lb.grid.vg.scrollToIndex(lb.i);
      const card = id != null
        ? lb.grid.vg.host.querySelector(`.card[data-id="${id}"]`)
        : lb.grid.vg.cards.get(lb.i);
      if (card && document.contains(card)) {
        card.focus({ preventScroll: true });
      } else if (lb.opener && document.contains(lb.opener)) {
        lb.opener.focus({ preventScroll: true });
      } else {
        const main = $("#main");
        if (main && main.focus) main.focus();
      }
    } else if (lb.opener && document.contains(lb.opener)) {
      lb.opener.focus({ preventScroll: true });
    } else {
      const main = $("#main");
      if (main && main.focus) main.focus();
    }
  }

  async function lbStep(d) {
    const n = lb.i + d;
    if (n < 0 || n >= lbCount()) return;
    if (lb.grid) {  // the next photo may sit on a page that has not been fetched yet
      await lb.grid.vg.ensureLoaded(n, n + 1);
      if (lb.grid.vg.idAt(n) == null) return;
    }
    lb.i = n;
    lbRender();
  }

  function lbRender() {
    const id = lbIdAt(lb.i);
    const it = state.byId[id] || { id, filename: "Photo #" + id, _basic: true };
    const img = $("#lb-img");
    const my = ++lb.token;
    img.alt = it.filename || "";
    img.classList.remove("swap-in");
    void img.offsetWidth;
    img.classList.add("swap-in");
    img.src = thumbUrl(id, gridThumbTier());  // already cached from the grid: shows instantly
    $("#lb-loading").hidden = false;
    const big = new Image();
    big.onload = () => { if (my === lb.token) { img.src = big.src; $("#lb-loading").hidden = true; } };
    big.onerror = () => { if (my === lb.token) $("#lb-loading").hidden = true; };
    big.src = thumbUrl(id, "preview");

    $("#lb-title").textContent = it.filename || "";
    $("#lb-path").textContent = it.relative_path || it.path || "";
    const rows = [];
    const add = (k, v) => { if (v != null && v !== "") rows.push(`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`); };
    if (it.width && it.height) add("Size", `${num(it.width)}×${num(it.height)}`);
    if (it.taken_at) add("Taken", String(it.taken_at).slice(0, 19).replace("T", " "));
    if (it.top1) add("Looks like", `${it.top1} (${Math.round(num(it.top1_score) * 100)}%)`);
    if (it.quality_score != null) add("Quality", num(it.quality_score).toFixed(2));
    if (it.aesthetic != null) add("Appeal", num(it.aesthetic).toFixed(2));
    if (it.preference != null) add("Your taste", num(it.preference).toFixed(2));
    if (it.face_count) add("Faces", it.face_count);
    if (it.rating != null) add("Rating", "★".repeat(Math.max(0, Math.min(5, num(it.rating)))) || num(it.rating));
    const notes = flagList(it).map((f) => f.text);
    if (notes.length) add("Notes", notes.join(", "));
    if (it.city || it.country_code) add("Place", [it.city, it.region, it.country_code].filter(Boolean).join(", "));
    $("#lb-meta").innerHTML = rows.join("");

    renderWhy(it);
    const full = !it._basic;
    const sel = $("#lb-cat");
    sel.innerHTML = '<option value="" disabled>Not sorted yet</option>' + state.cats.map((c) =>
      `<option value="${esc(c)}">${esc(catLabel(c))}</option>`).join("");
    sel.value = eff(it) || "";
    sel.disabled = !full;
    $("#lb-lock").disabled = !full;
    $("#lb-clear").disabled = !full || !it.manual_bucket;
    $("#lb-lock span").textContent = it.locked ? "Unlock" : "Lock";
    const ap = $("#lb-approve");
    ap.disabled = !full;
    ap.setAttribute("aria-pressed", String(!!it.delete_approved));
    ap.textContent = it.delete_approved ? "Marked for delete (undo)" : "Mark for delete";
    $("#lb-prev").disabled = lb.i === 0;
    $("#lb-next").disabled = lb.i >= lbCount() - 1;
    $("#lb-count").textContent = `${fmtN(lb.i + 1)} of ${fmtN(lbCount())}`;
    const nb = lbIdAt(lb.i + 1);
    if (nb != null) new Image().src = thumbUrl(nb, "preview");
    updateLbRating(it);
  }

  function renderWhy(it) {
    const box = $("#lb-why");
    const r = it.manual_bucket ? null : it.uncertain_reason;
    box.hidden = !r;
    if (!r) return;
    $("#lb-why-text").textContent = r.text || "";
    let guesses = it.guesses && it.guesses.length ? it.guesses : [];
    if (!guesses.length && it.top1) {
      guesses = [{ name: it.top1, score: num(it.top1_score) }];
      if (it.top2) guesses.push({ name: it.top2, score: num(it.top2_score) });
    }
    const maxS = Math.max(0.01, ...guesses.map((g) => num(g.score)));
    $("#lb-guesses").innerHTML = guesses.map((g) =>
      `<div class="bar" style="${catStyle(g.name)}"><span class="name">${esc(catLabel(g.name))}</span>
        <span class="track"><span class="fill" style="width:${Math.max(3, (num(g.score) / maxS) * 100)}%"></span></span>
        <span class="n">${Math.round(num(g.score) * 100)}%</span></div>`).join("");
  }

  function bindLightbox() {
    $("#lb-close").onclick = closeLightbox;
    $("#lb-prev").onclick = () => lbStep(-1);
    $("#lb-next").onclick = () => lbStep(1);

    const stage = $(".lb-stage");
    let touchStartX = 0, touchStartY = 0, touchValid = false;
    let swipedRecently = false;
    const SWIPE_THRESHOLD_PX = 50;  // (tune) starting default for swipe detection

    if (stage) {
      stage.addEventListener("pointerdown", (e) => {
        if (e.pointerType !== "touch") return;
        if (e.target.closest("button, .lb-rate")) return;
        touchStartX = e.clientX;
        touchStartY = e.clientY;
        touchValid = true;
      });

      stage.addEventListener("pointerup", (e) => {
        if (!touchValid || e.pointerType !== "touch") return;
        touchValid = false;
        const dx = e.clientX - touchStartX;
        const dy = e.clientY - touchStartY;
        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);
        if (absDx >= SWIPE_THRESHOLD_PX && absDx >= 2 * absDy) {
          swipedRecently = true;
          setTimeout(() => { swipedRecently = false; }, 120);
          if (dx < 0) lbStep(1);
          else lbStep(-1);
        }
      });

      stage.addEventListener("pointercancel", (e) => {
        if (e.pointerType === "touch") touchValid = false;
      });
    }

    $("#lightbox").addEventListener("click", (e) => {
      if (swipedRecently) return;
      if (e.target.id === "lightbox" || e.target.classList.contains("lb-stage")) closeLightbox();
    });
    $("#lb-cat").onchange = async (e) => {
      const id = lbIdAt(lb.i);
      try {
        const prev = await reassign(id, e.target.value);
        state.undo.push({ id, prev });
        toast("Moved to " + catLabel(e.target.value), "ok");
        loadStats();
      } catch (err) { toast(err.message, "err"); }
    };
    $("#lb-lock").onclick = () => toggleLock(lbIdAt(lb.i)).catch((e) => toast(e.message, "err"));
    $("#lb-approve").onclick = () => toggleApproveDelete(lbIdAt(lb.i)).catch((e) => toast(e.message, "err"));
    $$("#lb-rate .star").forEach((b) => {
      b.onclick = () => {
        const id = lbIdAt(lb.i);
        const n = num(b.dataset.star);
        const same = num((state.byId[id] || {}).rating) === n;  // the current star again clears it
        rateItem(id, same ? 0 : n).catch((e) => toast(e.message, "err"));
      };
    });
    $("#lb-fav").onclick = () => toggleFavorite(lbIdAt(lb.i)).catch((e) => toast(e.message, "err"));
    $("#lb-reject").onclick = () => toggleReject(lbIdAt(lb.i)).catch((e) => toast(e.message, "err"));
    $("#lb-similar").onclick = () => { const id = lbIdAt(lb.i); closeLightbox(); showSimilar(id); };
    $("#lb-clear").onclick = async () => {
      const id = lbIdAt(lb.i);
      try { const prev = await reassign(id, null); state.undo.push({ id, prev }); toast("Back to the AI choice", "ok"); }
      catch (e) { toast(e.message, "err"); }
    };
  }

  // ------------------------------------------------------------ folder picker
  const pk = { onPick: null, mode: "photos", data: null };

  async function pickerLoad(path) {
    const list = $("#picker-list");
    list.innerHTML = '<li class="picker-empty">Loading</li>';
    let data;
    try {
      data = await api("/api/browse" + (path ? "?path=" + encodeURIComponent(path) : ""));
    } catch (e) {
      if (path) {
        const up = path.replace(/[\\/][^\\/]*$/, "");
        if (/not found/i.test(e.message) && up && up !== path) return pickerLoad(up);
        toast(e.message, "err");
        return pickerLoad(null);
      }
      list.innerHTML = `<li class="picker-empty">${esc(e.message)}</li>`;
      return;
    }
    pk.data = data;
    $("#picker-path").value = data.path;
    $("#picker-up").disabled = !data.parent;
    $("#picker-roots").innerHTML = data.roots.map((r) =>
      `<li><button class="pk-item" data-go="${esc(r.path)}">${icon("folder")}<span>${esc(r.name)}</span></button></li>`).join("");
    list.innerHTML = data.error
      ? `<li class="picker-empty">${esc(data.error)}</li>`
      : data.entries.length
        ? data.entries.map((e) =>
          `<li><button class="pk-item" data-go="${esc(e.path)}">${icon("folder")}<span>${esc(e.name)}</span>${icon("right", "go")}</button></li>`).join("")
        : '<li class="picker-empty">No folders in here.</li>';
    list.scrollTop = 0;
    const info = $("#picker-info"), ok = $("#picker-ok");
    info.className = "grow";
    ok.disabled = false;
    if (pk.mode === "output") {
      if (data.writable) { info.textContent = "Organized photos will be written into this folder."; info.classList.add("good"); }
      else { info.textContent = "You cannot write to this folder."; info.classList.add("bad"); ok.disabled = true; }
    } else {
      info.textContent = data.photos
        ? `${fmtN(data.photos)} photo${data.photos === 1 ? "" : "s"} here. Subfolders are scanned too.`
        : "No photos directly in this folder. Subfolders are scanned too.";
    }
  }

  function openPicker(opts) {
    pk.opener = document.activeElement;
    pk.onPick = opts.onPick;
    pk.mode = opts.mode || "photos";
    $("#picker-title").textContent = opts.title || "Choose a folder";
    $("#picker-ok").textContent = opts.okLabel || "Use this folder";
    openOverlay($("#picker"));
    document.body.style.overflow = "hidden";
    if (pk.releaseTrap) pk.releaseTrap();
    pk.releaseTrap = trapFocus($("#picker"), closePicker);
    pickerLoad(opts.initial || null).then(() => {
      if (!$("#picker").hidden && (!document.activeElement || $("#picker").contains(document.activeElement))) {
        $("#picker-ok").focus();
      }
    });
  }

  function closePicker() {
    if (pk.releaseTrap) {
      pk.releaseTrap();
      pk.releaseTrap = null;
    }
    closeOverlay($("#picker"));
    document.body.style.overflow = "";
    if (pk.opener && document.contains(pk.opener)) {
      pk.opener.focus({ preventScroll: true });
    }
  }

  let pickerBound = false;
  function bindPicker() {
    if (pickerBound) return;
    pickerBound = true;
    $("#picker-close").onclick = closePicker;
    $("#picker-cancel").onclick = closePicker;
    $("#picker").addEventListener("click", (e) => { if (e.target.id === "picker") closePicker(); });
    $("#picker").addEventListener("click", (e) => {
      const b = e.target.closest("[data-go]");
      if (b) pickerLoad(b.dataset.go);
    });
    $("#picker-up").onclick = () => { if (pk.data && pk.data.parent) pickerLoad(pk.data.parent); };
    $("#picker-path").addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); pickerLoad($("#picker-path").value.trim()); }
    });
    $("#picker-new").onclick = async () => {
      if (!pk.data) return;
      const name = await showConfirmDialog({
        title: "New folder",
        confirmText: "Create",
        cancelText: "Cancel",
        input: {
          label: "Folder name",
          placeholder: "e.g. Vacation 2026",
          value: "",
        },
        validate(val) {
          const v = (val || "").trim();
          if (!v) return "Folder name cannot be empty.";
          if (v === "." || v === "..") return "Folder name not allowed.";
          if (/[/\\:*?"<>|\0]/.test(v)) return "Folder name cannot contain slashes or special characters (/ \\ : * ? \" < > |).";
          if (v.length > 200) return "Folder name must be 200 characters or fewer.";
          return null;
        },
      });
      if (name == null) return;
      try {
        const r = await post("/api/browse/mkdir", { parent: pk.data.path, name: name.trim() });
        await pickerLoad(r.path);
      } catch (e) { toast(e.message, "err"); }
    };
    $("#picker-ok").onclick = () => {
      if (!pk.data) return;
      const p = pk.data.path;
      closePicker();
      if (pk.onPick) pk.onPick(p);
    };
    $("#btn-browse-scan").onclick = () => openPicker({
      title: "Choose your photo folder", mode: "photos", initial: $("#scan-path").value.trim(),
      okLabel: "Use this folder",
      onPick: (p) => {
        const i = $("#scan-path");
        i.value = p;
        i.classList.remove("input-error");
        const bar = $(".folder-pick-bar");
        if (bar) bar.classList.remove("input-error");
        if (window.__syncScanPathDisplay) window.__syncScanPathDisplay();
      },
    });
    $("#btn-browse-out").onclick = () => openPicker({
      title: "Choose the destination folder", mode: "output", initial: $("#orgroot").value.trim(),
      okLabel: "Use this folder",
      onPick: (p) => { $("#orgroot").value = p; },
    });
  }

  // ------------------------------------------------------------ triage keys
  const triageAll = () => state.cats.filter((c) => !TRIAGE_SKIP.has(c));
  const triageCats = () => triageAll().slice(0, 9);

  function renderLegend() {
    const el = $("#triage-legend");
    if (!el) return;
    const rest = triageAll().slice(9);
    el.innerHTML = triageCats().map((c, i) =>
      `<span><kbd>${i + 1}</kbd> ${esc(catLabel(c))}</span>`).join("") +
      (rest.length ? `<span class="muted">Use the viewer or Move to for ${esc(rest.map(catLabel).join(", "))}</span>` : "");
  }

  function activeGrid() {
    const key = { categories: "cat", highlights: "highlights", events: "event", junk: "junk", utility: "utility", uncertain: "uncertain" }[state.tab];
    return key ? grids[key] || null : null;
  }

  const isTriage = (g) => g === grids.uncertain || g === grids.junk;
  const MODALS = ["#picker", "#help", "#facepanel", "#person-photos-overlay", "#similar-panel"];
  const SELF_MANAGED = ["#confirm-modal", "#auth-modal", "#restart-overlay"];  // they handle their own keys
  const open = (sel) => { const e = $(sel); return !!(e && !e.hidden); };
  const anyModal = () => MODALS.some(open) || !!catPicker;

  let helpOpener = null;
  let helpReleaseTrap = null;

  function openHelp() {
    helpOpener = document.activeElement;
    openOverlay($("#help"));
    if (helpReleaseTrap) helpReleaseTrap();
    helpReleaseTrap = trapFocus($("#help"), closeHelp);
    const btn = $("#help-close");
    if (btn) btn.focus();
  }

  function closeHelp() {
    if (helpReleaseTrap) {
      helpReleaseTrap();
      helpReleaseTrap = null;
    }
    closeOverlay($("#help"));
    if (helpOpener && document.contains(helpOpener)) {
      helpOpener.focus({ preventScroll: true });
    }
  }

  // Rating, reject, favourite and lock for one photo. In the triage tabs 1-9 mean categories instead.
  async function shortcut(k, id, triage) {
    if (id == null) return false;
    if (/^[1-5]$/.test(k) && !triage) { await rateItem(id, Number(k)); return true; }
    if (k === "0") { await rateItem(id, 0); return true; }
    if (k === "x") { await toggleReject(id); return true; }
    if (k === "f") { await toggleFavorite(id); return true; }
    if (k === "l") { await toggleLock(id); return true; }
    return false;
  }

  async function onKey(ev) {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (SELF_MANAGED.some(open)) return;
    const tag = (ev.target.tagName || "").toLowerCase();
    const typing = tag === "input" || tag === "select" || tag === "textarea";

    if (ev.key === "Escape") {
      if (open("#picker")) closePicker();
      else if (open("#help")) closeHelp();
      else if (open("#facepanel")) { if (!closeFaceRename()) closeBiometricProjection(); }
      else if (lb.open) closeLightbox();
      else if (open("#person-photos-overlay")) closePersonPhotos();
      else if (!$("#similar-panel").hidden) closeOverlay($("#similar-panel"));
      else if (state.picks.size || state.peoplePicks.size) clearPicks();
      else document.body.classList.remove("nav-open");
      return;
    }
    if (typing || anyModal()) return;
    if (ev.key === "?") { openHelp(); ev.preventDefault(); return; }
    const k = ev.key.toLowerCase();

    try {
      if (lb.open) {
        if (k === "arrowleft") await lbStep(-1);
        else if (k === "arrowright") await lbStep(1);
        else if (!(await shortcut(k, lbIdAt(lb.i), false))) return;
        ev.preventDefault();
        return;
      }

      const g = activeGrid();
      if (!g) return;
      const triage = isTriage(g);
      if (triage) state.triageGrid = g;
      const cols = Math.max(1, g.vg.cols);
      const rowsInView = Math.max(1, Math.floor(window.innerHeight / (g.vg.rowH || 200)));
      const pageStep = cols * rowsInView;
      const curIdx = g.vg.indexOfId(g.vg.cursorId);
      let step;
      if (k === "arrowright") step = 1;
      else if (k === "arrowleft") step = -1;
      else if (k === "arrowdown") step = cols;
      else if (k === "arrowup") step = -cols;
      else if (k === "pageup") step = -pageStep;
      else if (k === "pagedown") step = pageStep;
      else if (k === "home") step = -Math.max(0, curIdx);
      else if (k === "end") step = (g.vg.total || 0) - 1 - curIdx;
      else if (triage && (k === "j" || k === "k")) step = k === "j" ? 1 : -1;
      if (step !== undefined) {  // unrendered photos are scrolled into view first
        await g.vg.moveCursor(step);
        ev.preventDefault();
        return;
      }
      const id = g.vg.cursorId;
      if (k === " " || k === "enter") {
        if (ev.target.closest && ev.target.closest(".card, button, a, [role='button']")) return;  // they handle it
        if (id == null) return;
        openLightboxFor(g, id);
      } else if (triage && /^[1-9]$/.test(k)) {
        const cat = triageCats()[Number(k) - 1];
        if (id == null || !cat) return;
        const prev = await reassign(id, cat);
        state.undo.push({ id, prev });
        toast(`Moved to ${catLabel(cat)}`, "ok");
        loadStats();
      } else if (triage && k === "u") {
        const last = state.undo.pop();
        if (!last) { toast("Nothing to undo", "err"); return; }
        await reassign(last.id, last.prev);
        toast("Change undone", "ok");
        await g.reset();
      } else if (!(await shortcut(k, id, triage))) {
        return;
      }
      ev.preventDefault();
    } catch (e) { toast(e.message, "err"); }
  }

  // ------------------------------------------------------ ratings and favourites
  async function ratingRequest(id, kind, body, describe) {
    const url = `/api/images/${num(id)}/${kind}`;
    try {
      const r = await post(url, body);
      itemUpdated(r.item);
      return r.item;
    } catch (e) {
      if (!(e.status === 409 && e.body && e.body.needs_confirmation)) throw e;
      // Another program changed this photo's sidecar. Ask before overwriting it.
      const yes = await showConfirmDialog({
        title: "Overwrite the sidecar?",
        message: `This photo's sidecar was modified outside Photocure. Overwrite it with ${describe}?`,
        confirmText: "Overwrite", variant: "warning" });
      if (!yes) return null;
      const r = await post(url, Object.assign({}, body, { overwrite: true }));
      itemUpdated(r.item);
      return r.item;
    }
  }

  async function rateItem(id, n) {
    const it = await ratingRequest(id, "rating", { rating: n },
      n ? `a rating of ${n} star${n === 1 ? "" : "s"}` : "no rating");
    if (it) toast(n ? `Rated ${n} star${n === 1 ? "" : "s"}` : "Rating cleared", "ok");
  }

  async function toggleReject(id) {
    const un = num((state.byId[id] || {}).rating) === -1;
    const it = await ratingRequest(id, "reject", { reject: !un }, un ? "no rating" : "a reject");
    if (it) toast(un ? "No longer rejected" : "Rejected. Nothing was deleted.", "ok");
  }

  async function toggleApproveDelete(id) {
    const r = await post(`/api/images/${num(id)}/approve-delete`, {});
    itemUpdated(r.item);
    toast(r.item.delete_approved ? "Marked for delete. It will be left out of the result."
      : "Delete mark cleared. This photo stays in the result", "ok");
  }

  // Runs after sorting when "Approve all proposed deletions" was ticked before analysing.
  async function approveProposedDeletes(onlyNew) {
    if (!$("#approve-all-deletes").checked) return;
    try {
      const r = await post("/api/deletes/approve", onlyNew ? { only_new: true } : {});
      libLog(`Approved ${fmtN(r.changed)} proposed deletions`, "ok");
      resetVisibleGrids();
    } catch (e) { libLog(`Could not approve deletions: ${e.message}`, "warn"); }
  }

  async function toggleFavorite(id) {
    const r = await post(`/api/images/${num(id)}/favorite`, {});
    itemUpdated(r.item);
    toast(r.item.is_favorite ? "Added to favourites" : "Removed from favourites", "ok");
  }

  function updateLbRating(it) {
    const rating = num(it.rating);
    $$("#lb-rate .star").forEach((b) => b.classList.toggle("on", num(b.dataset.star) <= rating));
    $("#lb-fav").setAttribute("aria-pressed", String(!!it.is_favorite));
    $("#lb-reject").setAttribute("aria-pressed", String(rating === -1));
    $$("#lb-rate button").forEach((b) => { b.disabled = !!it._basic; });
  }

  // ------------------------------------------------------------------ bursts
  async function renderBursts(keepId) {
    const data = await api("/api/bursts");
    const list = $("#burstlist");
    const scroll = list.scrollTop;
    if (!data.items.length) {
      list.innerHTML = "";
      $("#burstdetail").innerHTML = emptyHTML({
        icon: "layers", title: "No burst groups yet",
        text: "Bursts are shots taken seconds apart. Run the Group bursts step to find them.",
        action: { label: "Go to Import", attr: 'data-goto="library"' },
      });
      return;
    }
    list.innerHTML = data.items.map((b) =>
      `<button class="burstrow burst-item" type="button" data-burst="${num(b.id)}">
        <span class="body"><div class="name">${esc(b.name)}</div>
             <div class="sub">${fmtN(b.image_count)} photos, best: ${esc(b.best_shot || "none")}</div></span>
        <svg class="chev" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>
      </button>`).join("");
    $$(".burstrow", list).forEach((row) => {
      const go = () => showBurst(num(row.dataset.burst));
      row.onclick = go;
      row.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } };
    });
    const keep = keepId != null && data.items.some((b) => b.id === keepId) ? keepId : data.items[0].id;
    list.scrollTop = scroll;
    await showBurst(keep);
  }

  let activePpoPersonId = null;

  // Cache so tab-switching back to People skips DOM rebuild & image re-load.
  // Mutating actions (merge, rename, delete, recluster) pass force=true.
  const _pc = { sort: null, fp: null };
  const pendingPeopleDeletions = new Set();

  function updatePeopleSaveBar() {
    const bar = $("#people-savebar");
    const countEl = $("#people-save-count");
    const tbBtn = $("#people-toolbar-save-btn");
    const tbCountEl = $("#people-toolbar-save-count");
    const count = pendingPeopleDeletions.size;

    if (count > 0) {
      const text = `${fmtN(count)} ${count === 1 ? "person" : "people"} marked for deletion`;
      if (countEl) countEl.textContent = text;
      if (bar) openOverlay(bar);
      if (tbBtn) tbBtn.hidden = false;
      if (tbCountEl) tbCountEl.textContent = fmtN(count);
    } else {
      if (bar) closeOverlay(bar);
      if (tbBtn) tbBtn.hidden = true;
    }

    const host = $("#peoplegrid");
    if (host) {
      const remaining = host.querySelectorAll(".people-card").length;
      const sumEl = $("#people-summary");
      if (sumEl) {
        sumEl.textContent = count > 0
          ? `${fmtN(remaining)} ${remaining === 1 ? "person" : "people"} (${fmtN(count)} pending delete)`
          : `${fmtN(remaining)} ${remaining === 1 ? "person" : "people"}`;
      }
    }
  }

  function wirePeopleSaveBar() {
    const bar = $("#people-savebar");
    if (bar && !bar._wired) {
      bar._wired = true;

      const discardBtn = $("#people-discard-btn");
      if (discardBtn) {
        discardBtn.addEventListener("click", async () => {
          if (!pendingPeopleDeletions.size) return;
          pendingPeopleDeletions.clear();
          updatePeopleSaveBar();
          _pc.fp = null;
          await renderPeople(true);
          toast("Deletions discarded", "ok");
        });
      }

      const saveBtn = $("#people-save-btn");
      if (saveBtn) {
        saveBtn.addEventListener("click", () => commitPeopleDeletions());
      }
    }

    const tbBtn = $("#people-toolbar-save-btn");
    if (tbBtn && !tbBtn._wired) {
      tbBtn._wired = true;
      tbBtn.addEventListener("click", () => commitPeopleDeletions());
    }
  }

  async function commitPeopleDeletions() {
    const ids = Array.from(pendingPeopleDeletions);
    if (!ids.length) return;
    const saveBtn = $("#people-save-btn");
    const tbBtn = $("#people-toolbar-save-btn");
    if (saveBtn) saveBtn.disabled = true;
    if (tbBtn) tbBtn.disabled = true;

    try {
      const res = await post("/api/people/bulk-delete", { person_ids: ids });
      const delCount = res.deleted_count || ids.length;
      toast(`Deleted ${fmtN(delCount)} ${delCount === 1 ? "person" : "people"} ✓`, "ok");
      pendingPeopleDeletions.clear();
      updatePeopleSaveBar();
      _pc.fp = null;
      await renderPeople(true);
      loadStats();
    } catch (e) {
      toast(`Failed to save deletions: ${e.message}`, "err");
    } finally {
      if (saveBtn) saveBtn.disabled = false;
      if (tbBtn) tbBtn.disabled = false;
    }
  }

  const allKnownPeopleNames = new Set();
  function getKnownPeopleNames() {
    const dl = $("#people-names");
    if (dl) Array.from(dl.options || []).forEach((o) => { if (o.value) allKnownPeopleNames.add(o.value); });
    if (Array.isArray(activeOverlayPeopleNames)) {
      activeOverlayPeopleNames.forEach((n) => { if (n) allKnownPeopleNames.add(n); });
    }
    return Array.from(allKnownPeopleNames).filter(Boolean).sort((a, b) => a.localeCompare(b));
  }

  function setupNameAutocomplete(input, getNames, onCommit) {
    let menu = null;
    let selectedIndex = 0;
    let currentMatches = [];

    const closeMenu = () => {
      if (menu) {
        menu.remove();
        menu = null;
        currentMatches = [];
        selectedIndex = -1;
      }
    };

    const updateHighlight = () => {
      if (!menu) return;
      const items = menu.querySelectorAll(".name-autocomplete-item");
      items.forEach((item, idx) => {
        const isSel = idx === selectedIndex;
        item.classList.toggle("active", isSel);
        item.setAttribute("aria-selected", isSel ? "true" : "false");
        if (isSel) item.scrollIntoView({ block: "nearest" });
      });
    };

    const commitIndex = (idx) => {
      const match = currentMatches[idx];
      if (match) {
        input.value = match;
        closeMenu();
        if (onCommit) onCommit(match);
      }
    };

    const renderMenu = (query) => {
      const q = (query || "").trim().toLowerCase();
      if (!q) {
        closeMenu();
        return;
      }

      const allNames = (getNames() || []).filter(Boolean);
      const seen = new Set();
      const uniqueNames = [];
      allNames.forEach((n) => {
        const lower = n.toLowerCase();
        if (!seen.has(lower)) {
          seen.add(lower);
          uniqueNames.push(n);
        }
      });

      const prefix = uniqueNames.filter((n) => n.toLowerCase().startsWith(q));
      const sub = uniqueNames.filter((n) => !n.toLowerCase().startsWith(q) && n.toLowerCase().includes(q));
      currentMatches = [...prefix, ...sub];

      if (!currentMatches.length) {
        closeMenu();
        return;
      }

      selectedIndex = 0;

      if (!menu) {
        menu = document.createElement("div");
        menu.className = "name-autocomplete-menu";
        menu.setAttribute("role", "listbox");
        const parent = input.closest(".people-name-pill-wrapper") || input.closest(".face-rename-pop") || input.parentElement;
        parent.style.position = parent.style.position || "relative";
        parent.appendChild(menu);
      }

      menu.innerHTML = currentMatches.map((name, idx) => {
        const matchIdx = name.toLowerCase().indexOf(q);
        let labelHtml = esc(name);
        if (matchIdx !== -1) {
          const before = esc(name.slice(0, matchIdx));
          const hit = esc(name.slice(matchIdx, matchIdx + q.length));
          const after = esc(name.slice(matchIdx + q.length));
          labelHtml = `${before}<b>${hit}</b>${after}`;
        }
        return `
          <div class="name-autocomplete-item${idx === 0 ? " active" : ""}" role="option" data-idx="${idx}" aria-selected="${idx === 0}">
            <svg class="ic" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
              <circle cx="12" cy="7" r="4"></circle>
            </svg>
            <span class="name-autocomplete-label">${labelHtml}</span>
            ${idx === 0 ? `<span class="name-autocomplete-hint">Tab or ↵</span>` : ""}
          </div>
        `;
      }).join("");

      menu.querySelectorAll(".name-autocomplete-item").forEach((el) => {
        el.addEventListener("mousedown", (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          const idx = parseInt(el.dataset.idx, 10);
          commitIndex(idx);
        });
      });
    };

    input.addEventListener("input", () => {
      renderMenu(input.value);
    });

    input.addEventListener("keydown", (e) => {
      if (menu && currentMatches.length > 0) {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          e.stopPropagation();
          selectedIndex = (selectedIndex + 1) % currentMatches.length;
          updateHighlight();
          return;
        }
        if (e.key === "ArrowUp") {
          e.preventDefault();
          e.stopPropagation();
          selectedIndex = (selectedIndex - 1 + currentMatches.length) % currentMatches.length;
          updateHighlight();
          return;
        }
        if (e.key === "Tab") {
          if (selectedIndex >= 0 && selectedIndex < currentMatches.length) {
            e.preventDefault();
            e.stopPropagation();
            const picked = currentMatches[selectedIndex];
            input.value = picked;
            closeMenu();
            input.focus();
            input.setSelectionRange(picked.length, picked.length);
            return;
          }
        }
        if (e.key === "Enter") {
          if (selectedIndex >= 0 && selectedIndex < currentMatches.length) {
            e.preventDefault();
            e.stopPropagation();
            commitIndex(selectedIndex);
            return;
          }
        }
        if (e.key === "Escape") {
          e.preventDefault();
          e.stopPropagation();
          closeMenu();
          return;
        }
      } else {
        const q = input.value.trim().toLowerCase();
        if (q && (e.key === "Tab" || e.key === "Enter")) {
          const allNames = (getNames() || []).filter(Boolean);
          const match = allNames.find((n) => n.toLowerCase().startsWith(q) && n.toLowerCase() !== q);
          if (match) {
            if (e.key === "Tab") {
              e.preventDefault();
              e.stopPropagation();
              input.value = match;
              input.focus();
              input.setSelectionRange(match.length, match.length);
              return;
            }
            if (e.key === "Enter") {
              e.preventDefault();
              e.stopPropagation();
              input.value = match;
              if (onCommit) onCommit(match);
              return;
            }
          }
        }
      }
    });

    input.addEventListener("blur", () => {
      setTimeout(closeMenu, 150);
    });

    return { close: closeMenu };
  }

  function _peopleFP(items) {
    return items.map((p) =>
      `${p.id}:${p.face_count}:${p.name || ""}:${(!p.name && p.similar_to) ? p.similar_to.target_id : ""}`
    ).join("|");
  }

  // Smooth FLIP ripple delete: removes a single card and ripples subsequent cards into place
  async function rippleDeleteCard(card, pid) {
    const host = $("#peoplegrid");
    if (!host || !card) return;

    if (reducedMotion()) {
      card.remove();
      _finalizeDelete(host, pid);
      return;
    }

    // 1. Identify all sibling cards that follow this card in DOM order
    const allCards = Array.from(host.querySelectorAll(".people-card"));
    const delIdx = allCards.indexOf(card);
    const subsequent = delIdx !== -1 ? allCards.slice(delIdx + 1) : [];

    // 2. First: Record their current bounding rects before DOM changes
    const firstRects = subsequent.map((el) => el.getBoundingClientRect());

    // 3. Animate the deleted card: shrink, fade out, and blur
    card.style.pointerEvents = "none";
    const fastMs = moMs("--mo-fast", 160), slowMs = moMs("--mo-slow", 380), staggerMs = moMs("--mo-stagger", 28);
    const soft = getComputedStyle(document.documentElement).getPropertyValue("--ease-soft").trim() || "cubic-bezier(.22, 1, .36, 1)";
    card.style.transition = `transform ${fastMs}ms ${soft}, opacity ${fastMs}ms ${soft}, filter ${fastMs}ms ${soft}`;
    card.style.transform = "scale(0.72)";
    card.style.opacity = "0";
    card.style.filter = "blur(4px)";

    await new Promise((r) => setTimeout(r, fastMs));

    // 4. Last: Remove the deleted card from the DOM
    card.remove();

    // 5. Invert: Set delta transform so subsequent cards visually stay where they were
    subsequent.forEach((el, i) => {
      const lastRect = el.getBoundingClientRect();
      const dx = firstRects[i].left - lastRect.left;
      const dy = firstRects[i].top - lastRect.top;
      if (dx !== 0 || dy !== 0) {
        el.style.transition = "none";
        el.style.transform = `translate(${dx}px, ${dy}px)`;
      }
    });

    // Force layout reflow
    void host.offsetHeight;

    // 6. Play: Animate from inverted position back to natural (0,0) with a ripple wave delay
    requestAnimationFrame(() => {
      subsequent.forEach((el, i) => {
        const rippleDelay = Math.min(i, 8) * staggerMs;
        el.style.transition = `transform ${slowMs}ms ${soft} ${rippleDelay}ms`;
        el.style.transform = "";
        const clean = (e) => {
          if (e.target === el) {
            el.style.transition = "";
            el.removeEventListener("transitionend", clean);
          }
        };
        el.addEventListener("transitionend", clean);
      });
    });

    _finalizeDelete(host, pid);
  }

  function setPeopleCount(v) {
    const e = $('[data-count="people"]');
    if (!e) return;
    const next = v ? fmtN(v) : "";
    const had = e.textContent;
    if (had !== next) {
      e.textContent = next;
      e.classList.remove("tick");
      void e.offsetWidth;
      e.classList.add("tick");
      afterAnim(e, moMs("--mo-slow", 380) * 2).then(() => e.classList.remove("tick"));
    }
  }

  function _finalizeDelete(host, pid) {
    const remaining = host.querySelectorAll(".people-card");
    const count = remaining.length;
    setPeopleCount(count);
    if (state.peoplePicks.delete(num(pid))) applyPicks();
    const pendingCount = pendingPeopleDeletions.size;
    const sumEl = $("#people-summary");
    if (sumEl) {
      sumEl.textContent = pendingCount > 0
        ? `${fmtN(count)} ${count === 1 ? "person" : "people"} (${fmtN(pendingCount)} pending delete)`
        : `${fmtN(count)} ${count === 1 ? "person" : "people"}`;
    }

    if (count === 0) {
      host.innerHTML = `<div class="people-empty"><svg class="ic" aria-hidden="true"><use href="#i-user"/></svg><p>No identified people yet.</p></div>`;
    }

    if (_pc.fp) {
      _pc.fp = _pc.fp.split("|").filter((seg) => !seg.startsWith(`${pid}:`)).join("|");
    }
  }

  async function renderPeople(force = false) {
    const host = $("#peoplegrid");
    const sort = state.peopleSort || "similarity";
    const data = await api(`/api/people?sort=${encodeURIComponent(sort)}`);
    (data.items || []).forEach((p) => { if (p.name) allKnownPeopleNames.add(p.name); });
    const allItems = (data.items || []).filter((p) => p.face_count > 0 && (p.sample_face_id != null || p.name));
    const items = allItems.filter((p) => !pendingPeopleDeletions.has(p.id));
    const fp = _peopleFP(items);
    const live = new Set(items.map((p) => num(p.id)));
    let pruned = false;
    state.peoplePicks.forEach((id) => { if (!live.has(id)) { state.peoplePicks.delete(id); pruned = true; } });
    if (pruned) applyPicks();

    wirePeopleSaveBar();
    updatePeopleSaveBar();

    // ── Cache hit: data unchanged, DOM already built → just scroll-restore ────
    if (!force && sort === _pc.sort && fp === _pc.fp && host && host.children.length > 0) {
      // Update counts/summary even without a full rebuild
      const sumEl = $("#people-summary");
      if (sumEl) sumEl.textContent = `${fmtN(items.length)} ${items.length === 1 ? "person" : "people"}`;
      setPeopleCount(items.length);
      return;
    }
    _pc.sort = sort;
    _pc.fp   = fp;

    // Summary count
    const sumEl = $("#people-summary");
    if (sumEl) sumEl.textContent = `${fmtN(items.length)} ${items.length === 1 ? "person" : "people"}`;
    setPeopleCount(items.length);

    // Sort buttons
    $$(".people-sort-group .chip-btn").forEach((btn) => {
      btn.setAttribute("aria-pressed", btn.dataset.sort === sort);
      if (!btn._wired) {
        btn._wired = true;
        btn.addEventListener("click", () => {
          state.peopleSort = btn.dataset.sort;
          _pc.sort = null; // sort changed → force rebuild
          renderPeople();
        });
      }
    });

    // Recluster button
    const reclusterBtn = $("#people-recluster-btn");
    if (reclusterBtn && !reclusterBtn._wired) {
      reclusterBtn._wired = true;
      reclusterBtn.addEventListener("click", async () => {
        const ok = await showConfirmDialog({
          title: "Re-cluster all faces?",
          message: "This will separate faces that were mistakenly grouped together by applying the strict facial identity threshold (0.64). Custom names will be preserved.",
          confirmText: "Re-cluster",
          variant: "primary",
        });
        if (!ok) return;
        reclusterBtn.disabled = true;
        toast("Re-clustering faces with strict threshold…", "ok");
        try {
          const res = await post("/api/people/recluster", {});
          toast(`Done! Formed ${fmtN(res.clusters_created)} person clusters across ${fmtN(res.faces_clustered)} faces.`, "ok");
          _pc.fp = null;
          await renderPeople(true);
          loadStats();
        } catch (e) {
          toast(`Re-clustering failed: ${e.message}`, "err");
        } finally {
          reclusterBtn.disabled = false;
        }
      });
    }

    const datalist = $("#people-names");
    if (datalist) {
      datalist.innerHTML = items.filter((p) => p.name)
        .map((p) => `<option value="${esc(p.name)}"></option>`).join("");
    }
    if (!items.length) {
      host.innerHTML = emptyHTML({
        icon: "users", title: "No faces grouped yet",
        text: "Photocure groups faces it believes are the same person as photos are analyzed. Run Analyze to find them.",
      });
      return;
    }

    // ── Build cards ──────────────────────────────────────────────────────────
    host.innerHTML = items.map((p, i) => `
      <div class="people-card" draggable="true" data-person-id="${num(p.id)}" data-face-count="${num(p.face_count)}" style="--card-index:${i}">
        <!-- Portrait Image -->
        <div class="people-card-portrait">
          ${p.sample_face_id
            ? `<img class="people-face-img" src="/api/faces/${num(p.sample_face_id)}.jpg" alt="" loading="lazy"
                 onload="this.classList.add('loaded')"
                 onerror="this.classList.add('loaded');this.style.display='none';this.nextElementSibling.style.display='flex'">
               <div class="people-face-fallback" style="display:none"><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5'><circle cx='12' cy='8' r='4'/><path d='M4 20c0-4 3.6-7 8-7s8 3 8 7'/></svg></div>`
            : `<div class="people-face-fallback"><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5'><circle cx='12' cy='8' r='4'/><path d='M4 20c0-4 3.6-7 8-7s8 3 8 7'/></svg></div>`
          }
        </div>

        <!-- Dark Bottom Gradient Vignette -->
        <div class="people-card-vignette"></div>

        <button class="people-card-select" type="button" aria-pressed="${state.peoplePicks.has(num(p.id))}" aria-label="Select ${esc(p.name || 'Person #' + p.id)}">
          <svg class="ic" aria-hidden="true"><use href="#i-check"/></svg>
        </button>

        <!-- Top-Left Hover Dismiss Button -->
        <button class="people-card-dismiss" type="button" title="Not a person / Delete cluster" aria-label="Delete person">
          <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <!-- Top-Right Face Count Pill -->
        <div class="people-card-count-badge" data-count="${num(p.face_count)}">
          ${fmtN(p.face_count)}
        </div>

        <!-- Drag & Drop Overlay -->
        <div class="people-drop-overlay">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M8 12h8M12 8l4 4-4 4"/>
            <circle cx="12" cy="12" r="10" stroke-dasharray="4 2"/>
          </svg>
          <span class="people-drop-label">Merge here</span>
        </div>

        <!-- Bottom Controls Container -->
        <div class="people-card-bottom">
          ${!p.name && p.similar_to ? `
            <button class="people-merge-pill" type="button"
                    data-source-id="${num(p.id)}"
                    data-target-id="${num(p.similar_to.target_id)}"
                    data-target-name="${esc(p.similar_to.target_name || 'Person #' + p.similar_to.target_id)}"
                    title="Visually similar (${Math.round(p.similar_to.score * 100)}% match). Click to merge!">
              ${p.similar_to.target_sample_face_id ? `
                <img class="people-merge-thumb" src="/api/faces/${num(p.similar_to.target_sample_face_id)}.jpg" alt="" loading="lazy">
              ` : `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="18" cy="18" r="3"></circle>
                  <circle cx="6" cy="6" r="3"></circle>
                  <path d="M6 21V9a9 9 0 0 0 9 9"></path>
                </svg>
              `}
              <span>Merge with ${esc(p.similar_to.target_name || 'Person #' + p.similar_to.target_id)}?</span>
            </button>` : ""
          }

          <!-- Interactive Name Pill -->
          <div class="people-name-pill-wrapper" data-person-id="${num(p.id)}" data-current-name="${esc(p.name || '')}">
            <button class="people-name-pill" type="button">
              <span class="people-name-label">
                ${p.name ? `<span class="people-name-dot"></span>${esc(p.name)}` : `+ Name this person`}
              </span>
              <span class="people-name-action">Edit</span>
            </button>
          </div>
        </div>
      </div>`).join("");

    // ── Card Interactions, Inline Editing, Merging, Drag & Drop ──────────────
    let dragSourceId = null;
    let dragSourceCount = 0;

    $$(".people-card", host).forEach((card) => {
      const pid = card.dataset.personId;

      // ── Inline Naming ──────────────────────────────────────────────────────
      const pillWrapper = card.querySelector(".people-name-pill-wrapper");
      if (pillWrapper) {
        pillWrapper.addEventListener("click", (ev) => {
          ev.stopPropagation();
          if (pillWrapper.querySelector(".people-name-form")) return;

          const currentName = pillWrapper.dataset.currentName || "";
          pillWrapper.innerHTML = `
            <form class="people-name-form" onsubmit="return false;">
              <input type="text" class="people-name-input" autocomplete="off" placeholder="Type a name..." value="${esc(currentName)}">
              <button type="submit" class="people-name-submit" title="Save name">
                <svg fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              </button>
            </form>
          `;
          const input = pillWrapper.querySelector(".people-name-input");
          const form = pillWrapper.querySelector(".people-name-form");
          input.focus();
          input.select();

          let saved = false;
          let ac = null;
          const save = async (nameOverride) => {
            if (saved) return;
            saved = true;
            if (ac) ac.close();
            const newName = (nameOverride != null ? nameOverride : input.value).trim();
            if (newName === currentName) {
              restore();
              return;
            }
            try {
              await post(`/api/people/${pid}/rename`, { name: newName });
              toast("Renamed ✓", "ok");
              allKnownPeopleNames.add(newName);
              _pc.fp = null; await renderPeople(true);
            } catch (e) {
              toast(e.message, "err");
              saved = false;
              restore();
            }
          };

          const restore = () => {
            if (ac) ac.close();
            pillWrapper.dataset.currentName = currentName;
            pillWrapper.innerHTML = `
              <button class="people-name-pill" type="button">
                <span class="people-name-label">
                  ${currentName ? `<span class="people-name-dot"></span>${esc(currentName)}` : `+ Name this person`}
                </span>
                <span class="people-name-action">Edit</span>
              </button>
            `;
          };

          ac = setupNameAutocomplete(input, getKnownPeopleNames, (name) => save(name));

          form.addEventListener("submit", (e) => {
            e.preventDefault();
            save();
          });

          input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              save();
            } else if (e.key === "Escape") {
              e.preventDefault();
              saved = true;
              restore();
            }
          });

          // Leaving the field cancels; only Enter or the check button commits.
          form.addEventListener("focusout", (e) => {
            if (saved || (e.relatedTarget && (form.contains(e.relatedTarget) || pillWrapper.contains(e.relatedTarget)))) return;
            saved = true;
            restore();
          });
        });
      }

      // ── Suggested Merge Pill ───────────────────────────────────────────────
      const mergeBtn = card.querySelector(".people-merge-pill");
      if (mergeBtn) {
        mergeBtn.addEventListener("click", async (ev) => {
          ev.stopPropagation();
          const sourceId = mergeBtn.dataset.sourceId;
          const targetId = mergeBtn.dataset.targetId;
          const targetName = mergeBtn.dataset.targetName;
          try {
            await post(`/api/people/${sourceId}/merge-into/${targetId}`, {});
            toast(`Merged with ${targetName} ✓`, "ok");
            const targetCard = host.querySelector(`.people-card[data-person-id="${targetId}"]`);
            if (targetCard) {
              const srcCount = parseInt(card.dataset.faceCount, 10) || 0;
              const tgtCount = parseInt(targetCard.dataset.faceCount, 10) || 0;
              const newCount = tgtCount + srcCount;
              targetCard.dataset.faceCount = newCount;
              const badge = targetCard.querySelector(".people-card-count-badge");
              if (badge) badge.textContent = fmtN(newCount);
            }
            await rippleDeleteCard(card, sourceId);
          } catch (e) {
            toast(`Merge failed: ${e.message}`, "err");
          }
        });

        // ── Merge-thumb zoom-preview tooltip ────────────────────────────────
        const mergeThumb = mergeBtn.querySelector(".people-merge-thumb");
        if (mergeThumb) {
          const preview     = document.getElementById("people-merge-preview");
          const previewImg  = document.getElementById("people-merge-preview-img");
          const previewLbl  = document.getElementById("people-merge-preview-label");
          const GAP = 10; // px gap between tooltip and cursor

          mergeThumb.addEventListener("mouseenter", () => {
            previewImg.src = mergeThumb.src;
            previewLbl.textContent = mergeBtn.dataset.targetName || "Preview";
            preview.classList.add("visible");
          });

          mergeThumb.addEventListener("mousemove", (ev) => {
            // Position tooltip above the cursor, centered
            const TW = 120, TH = 140; // approx tooltip dimensions
            let x = ev.clientX - TW / 2;
            let y = ev.clientY - TH - GAP;
            // Clamp to viewport
            x = Math.max(8, Math.min(x, window.innerWidth  - TW - 8));
            y = Math.max(8, Math.min(y, window.innerHeight - TH - 8));
            preview.style.left = x + "px";
            preview.style.top  = y + "px";
          });

          mergeThumb.addEventListener("mouseleave", () => {
            preview.classList.remove("visible");
          });
        }
      }

      // ── Multi-select ───────────────────────────────────────────────────────
      const selBtn = card.querySelector(".people-card-select");
      if (selBtn) {
        selBtn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          const id = num(pid);
          if (state.peoplePicks.has(id)) state.peoplePicks.delete(id); else state.peoplePicks.add(id);
          applyPicks();
        });
      }

      // ── Dismiss / Delete Cluster (Staged UI Deletion) ───────────────────────
      const dismissBtn = card.querySelector(".people-card-dismiss");
      if (dismissBtn) {
        dismissBtn.addEventListener("click", async (ev) => {
          ev.stopPropagation();
          const personIdNum = parseInt(pid, 10);
          if (isNaN(personIdNum)) return;

          pendingPeopleDeletions.add(personIdNum);
          await rippleDeleteCard(card, pid);
          updatePeopleSaveBar();
        });
      }

      // ── Drag & Drop Merging ────────────────────────────────────────────────
      card.addEventListener("dragstart", (ev) => {
        dragSourceId = pid;
        dragSourceCount = parseInt(card.dataset.faceCount, 10) || 0;
        card.classList.add("people-card--dragging");
        ev.dataTransfer.effectAllowed = "move";
        ev.dataTransfer.setData("text/plain", dragSourceId);

        const portrait = card.querySelector(".people-card-portrait");
        if (portrait) {
          ev.dataTransfer.setDragImage(portrait, portrait.offsetWidth / 2, portrait.offsetHeight / 2);
        }
      });

      card.addEventListener("dragend", () => {
        card.classList.remove("people-card--dragging");
        $$(".people-card--drop-target", host).forEach((c) => {
          c.classList.remove("people-card--drop-target");
          const badge = c.querySelector(".people-card-count-badge");
          if (badge) {
            const orig = parseInt(c.dataset.faceCount, 10) || 0;
            badge.textContent = fmtN(orig);
          }
        });
        dragSourceId = null;
        dragSourceCount = 0;
      });

      card.addEventListener("dragenter", (ev) => {
        ev.preventDefault();
        if (!dragSourceId || pid === dragSourceId) return;
        card.classList.add("people-card--drop-target");

        const badge = card.querySelector(".people-card-count-badge");
        if (badge) {
          const targetCount = parseInt(card.dataset.faceCount, 10) || 0;
          const combined = targetCount + dragSourceCount;
          badge.textContent = fmtN(combined);
        }
      });

      card.addEventListener("dragover", (ev) => {
        ev.preventDefault();
        if (!dragSourceId || pid === dragSourceId) return;
        ev.dataTransfer.dropEffect = "move";
      });

      card.addEventListener("dragleave", (ev) => {
        if (card.contains(ev.relatedTarget)) return;
        card.classList.remove("people-card--drop-target");
        const badge = card.querySelector(".people-card-count-badge");
        if (badge) {
          const orig = parseInt(card.dataset.faceCount, 10) || 0;
          badge.textContent = fmtN(orig);
        }
      });

      card.addEventListener("drop", async (ev) => {
        ev.preventDefault();
        if (!dragSourceId || pid === dragSourceId) return;

        card.classList.remove("people-card--drop-target");
        card.classList.add("people-card--merging");

        try {
          await post(`/api/people/${dragSourceId}/merge-into/${pid}`, {});
          toast("Clusters merged ✓", "ok");
          card.classList.remove("people-card--merging");

          const srcCount = dragSourceCount || 0;
          const tgtCount = parseInt(card.dataset.faceCount, 10) || 0;
          const newCount = tgtCount + srcCount;
          card.dataset.faceCount = newCount;
          const badge = card.querySelector(".people-card-count-badge");
          if (badge) badge.textContent = fmtN(newCount);

          const srcCard = host.querySelector(`.people-card[data-person-id="${dragSourceId}"]`);
          if (srcCard) {
            await rippleDeleteCard(srcCard, dragSourceId);
          } else {
            _pc.fp = null;
            await renderPeople(true);
          }
        } catch (e) {
          card.classList.remove("people-card--merging");
          toast(`Merge failed: ${e.message}`, "err");
        }
      });

      // ── Card Click → Open Person Photo Overlay ─────────────────────────────
      card.addEventListener("click", (ev) => {
        if (ev.target.closest(".people-name-pill-wrapper") ||
            ev.target.closest(".people-merge-pill") ||
            ev.target.closest(".people-card-dismiss") ||
            ev.target.closest(".people-card-select")) {
          return;
        }
        if (dragSourceId) return; // ignore stray clicks right after a drag
        showPersonPhotos(pid);
      });
    });
  }


  // ============================================================================
  // PERSON PHOTOS OVERLAY
  // ============================================================================

  const ppoEl       = () => $("#person-photos-overlay");
  const ppoGrid     = () => $("#ppo-grid");
  const ppoName     = () => $("#ppo-name");
  const ppoCount    = () => $("#ppo-count");
  const ppoAvatar   = () => $("#ppo-avatar");

  async function showPersonPhotos(personId) {
    const overlay = ppoEl();
    if (!overlay) return;
    activePpoPersonId = personId;

    // Show overlay immediately with a loading state
    ppoName().textContent = "Loading…";
    ppoCount().textContent = "";
    ppoAvatar().innerHTML = "";
    ppoGrid().innerHTML = `<div class="ppo-loading"><span class="spinner"></span></div>`;
    openOverlay(overlay);
    document.body.style.overflow = "hidden";

    try {
      const data = await api(`/api/people/${personId}/photos`);
      const { person, photos } = data;

      // Header
      const displayName = person.name || "Unnamed";
      ppoName().textContent = displayName;
      ppoCount().textContent = `${fmtN(person.face_count)} ${person.face_count === 1 ? "photo" : "photos"}`;

      // Avatar — use first photo's face crop if available
      const firstWithFace = photos.find((p) => p.face_url);
      if (firstWithFace) {
        ppoAvatar().innerHTML =
          `<img src="${firstWithFace.face_url}" alt="" loading="lazy"
                onerror="this.style.display='none'" class="ppo-avatar-img">`;
      } else {
        ppoAvatar().innerHTML =
          `<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5'>
             <circle cx='12' cy='8' r='4'/><path d='M4 20c0-4 3.6-7 8-7s8 3 8 7'/>
           </svg>`;
      }

      if (!photos.length) {
        ppoGrid().innerHTML = `<p class="ppo-empty">No analysed photos found for this person yet.</p>`;
        return;
      }

      // Photo grid — display photos
      ppoGrid().innerHTML = photos.map((p, i) => `
        <div class="ppo-thumb" data-image-id="${num(p.id)}"
             style="--i:${i}" tabindex="0" role="button"
             aria-label="Open ${esc(p.filename)}">
          <img class="ppo-thumb-img" src="${p.thumb_url}" alt="${esc(p.filename)}" loading="lazy">
          ${p.face_url ? `<img class="ppo-thumb-face" src="${p.face_url}" alt="Detected face" loading="lazy">` : ""}
          <div class="ppo-thumb-info">
            <span class="ppo-thumb-filename">${esc(p.filename)}</span>
            ${p.taken_at ? `<span class="ppo-thumb-date">${new Date(p.taken_at).toLocaleDateString()}</span>` : ""}
          </div>
        </div>`).join("");

      // Wire photo click → open lightbox with full list of photos so user can browse
      const allIds = photos.map((p) => num(p.id));
      $$(".ppo-thumb", ppoGrid()).forEach((thumb) => {
        const imgId = num(thumb.dataset.imageId);
        const openPhoto = () => {
          openLightboxList(allIds, imgId, thumb);
        };
        thumb.addEventListener("click", openPhoto);
        thumb.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            openPhoto();
          }
        });
      });

    } catch (e) {
      ppoGrid().innerHTML = `<p class="ppo-empty">Failed to load photos: ${esc(e.message)}</p>`;
    }
  }

  function hidePpoFaceZoom() {
    const z = $("#ppo-face-zoom");
    if (z) z.hidden = true;
  }

  function showPpoFaceZoom(avatar) {
    let z = $("#ppo-face-zoom");
    if (!z) {
      z = el("div", { id: "ppo-face-zoom", class: "ppo-face-zoom", hidden: true, "aria-hidden": "true" }, el("img", { alt: "" }));
      document.body.append(z);
    }
    const img = z.firstElementChild;
    if (img.getAttribute("src") !== avatar.src) img.src = avatar.src;
    z.hidden = false;
    const r = avatar.getBoundingClientRect();
    const w = z.offsetWidth || 240;
    const h = z.offsetHeight || 240;
    const gap = 10;
    const left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
    let top = r.bottom + gap;
    if (top + h > window.innerHeight - 8) top = Math.max(8, r.top - h - gap);
    z.style.left = left + "px";
    z.style.top = top + "px";
  }

  function closePersonPhotos() {
    const overlay = ppoEl();
    if (!overlay) return;
    hidePpoFaceZoom();
    activePpoPersonId = null;
    closeOverlay(overlay);
    document.body.style.overflow = "";
  }

  // Wire up close button + backdrop click + Escape + delete
  function initPpo() {
    const overlay = ppoEl();
    if (!overlay) return;
    $("#ppo-close")?.addEventListener("click", closePersonPhotos);
    const grid = ppoGrid();
    if (grid) {
      grid.addEventListener("mouseover", (ev) => {
        const a = ev.target.closest && ev.target.closest(".ppo-thumb-face");
        if (a) showPpoFaceZoom(a);
      });
      grid.addEventListener("mouseout", (ev) => {
        if (ev.target.closest && ev.target.closest(".ppo-thumb-face")) hidePpoFaceZoom();
      });
      grid.addEventListener("scroll", hidePpoFaceZoom, { passive: true });
    }
    overlay.querySelector(".ppo-backdrop")?.addEventListener("click", closePersonPhotos);
    overlay.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && !lb.open) { ev.stopPropagation(); closePersonPhotos(); }
    });
    $("#ppo-delete")?.addEventListener("click", async () => {
      if (!activePpoPersonId) return;
      const name = ppoName().textContent.trim() || `Person #${activePpoPersonId}`;
      const ok = await showConfirmDialog({
        title: "Delete person?",
        message: `Delete cluster for "${name}"? Photos will remain in your library, but face assignments to this person will be cleared.`,
        confirmText: "Delete",
        variant: "danger",
      });
      if (!ok) return;
      try {
        await api(`/api/people/${activePpoPersonId}`, { method: "DELETE" });
        const targetPid = activePpoPersonId;
        closePersonPhotos();
        toast("Person deleted", "ok");
        const host = $("#peoplegrid");
        const card = host?.querySelector(`.people-card[data-person-id="${targetPid}"]`);
        if (card) {
          await rippleDeleteCard(card, targetPid);
        } else {
          _pc.fp = null;
          await renderPeople(true);
        }
      } catch (e) {
        toast(`Delete failed: ${e.message}`, "err");
      }
    });
  }
  initPpo();


  // ============================================================================
  // DIRECT 1-TO-1 FACE-TO-ZOOM BIOMETRIC PROJECTION (NeuralVision)
  // ============================================================================
  // Brand palette: blue, pink, white, amber (Photocure design system)
  const FACE_PALETTE = ['#2b9af6', '#fa3b91', '#ffffff', '#f6a315', '#36bdfb', '#ff5aa5', '#fcce40', '#94a3b8'];

  let activeOverlayBurstId = null;
  let activeOverlayImageId = null;
  let activeOverlayBurstData = null;
  let activeOverlayMembers = [];
  let activeOverlayFaces = [];
  // Stand-in for curMember when there is no burst (activeOverlayMembers is empty): built from
  // the per-image faces response so a standalone photo renders the same chrome as a burst frame.
  let activeOverlayStandaloneMember = null;
  let activeOverlayPeopleNames = [];
  let overlayLayout = 'radial';      // 'radial' | 'split' | 'cinema' | 'matrix'
  let faceZoomMultiplier = 1.08;     // 0.86 (1.5x) | 1.08 (2.5x) | 1.28 (3.5x)
  let hoveredFaceId = null;
  let selectedFaceId = null;
  let currentBurstId = null;
  let overlayReq = 0;                // bumped on every open/close so a late response cannot revive a dismissed overlay

  let facePanelRelease = null;
  let facePanelOpener = null;

  function enterFacePanel(overlay) {
    if (!facePanelRelease) {
      facePanelOpener = document.activeElement;
      facePanelRelease = trapFocus(overlay, closeBiometricProjection);
    }
    if (!overlay.contains(document.activeElement)) {
      const first = overlay.querySelector("button, input, [tabindex]");
      if (first) first.focus({ preventScroll: true });
    }
  }

  function closeBiometricProjection() {
    overlayReq++;
    const overlay = $("#facepanel");
    if (facePanelRelease) { facePanelRelease(); facePanelRelease = null; }
    if (overlay) {
      overlay.hidden = true;
      overlay.replaceChildren();
    }
    if (facePanelOpener && document.contains(facePanelOpener)) facePanelOpener.focus({ preventScroll: true });
    facePanelOpener = null;
    closeFaceRename();
    activeOverlayBurstId = null;
    activeOverlayImageId = null;
    activeOverlayBurstData = null;
    activeOverlayMembers = [];
    activeOverlayFaces = [];
    activeOverlayStandaloneMember = null;
    activeOverlayPeopleNames = [];
    hoveredFaceId = null;
    selectedFaceId = null;
  }

  function setOverlayLayout(mode) {
    overlayLayout = mode;
    if (activeOverlayImageId) renderBiometricOverlay();
  }

  function setFaceZoom(multiplier) {
    faceZoomMultiplier = multiplier;
    if (activeOverlayImageId) renderBiometricOverlay();
  }

  function setSpeed(speed) {
    document.documentElement.style.setProperty('--speed', speed || 0.5);
    if (activeOverlayImageId) renderBiometricOverlay();
  }

  function setHoveredFace(faceId) {
    if (hoveredFaceId === faceId) return;
    hoveredFaceId = faceId;
    updateBiometricSpotlight();
  }

  function selectFaceInOverlay(faceId, anchor) {
    selectedFaceId = faceId;
    hoveredFaceId = faceId;
    updateBiometricSpotlight();
    openFaceRename(faceId, anchor);
  }

  function closeFaceRename() {
    const pop = $("#face-rename-pop");
    if (!pop) return false;
    pop.remove();
    return true;
  }

  function openFaceRename(faceId, anchor) {
    const overlay = $("#facepanel");
    const face = activeOverlayFaces.find((f) => f.id === faceId);
    if (!overlay || !face || !anchor) return;
    closeFaceRename();
    const current = face.personId != null ? face.name : "";
    const input = el("input", {
      type: "text", id: "face-rename-input", class: "face-rename-input",
      placeholder: "Who is this?", maxlength: 80, autocomplete: "off", "aria-label": "Person name",
    });
    input.value = current;
    const ok = el("button", {
      type: "submit", class: "face-rename-ok", title: "Save name", "aria-label": "Save name",
      html: icon("check", "sm"),
    });
    const names = el("datalist", { id: "face-rename-names" },
      activeOverlayPeopleNames.map((n) => el("option", { value: n })));
    const pop = el("form", { id: "face-rename-pop", class: "face-rename-pop" }, input, ok, names);
    const commitRename = async (chosenName) => {
      const name = (chosenName != null ? chosenName : input.value).trim();
      if (!name || name === current) { closeFaceRename(); return; }
      ok.disabled = true;
      await reassignFace(faceId, { new_name: name });
      allKnownPeopleNames.add(name);
      ok.disabled = false;
    };
    pop.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      await commitRename();
    });
    setupNameAutocomplete(
      input,
      () => activeOverlayPeopleNames.length ? activeOverlayPeopleNames : getKnownPeopleNames(),
      (name) => commitRename(name)
    );
    overlay.append(pop);
    const o = overlay.getBoundingClientRect();
    const a = anchor.getBoundingClientRect();
    const w = pop.offsetWidth || 220;
    const h = pop.offsetHeight || 40;
    const left = Math.min(Math.max(8, a.left + a.width / 2 - o.left - w / 2), o.width - w - 8);
    let top = a.bottom - o.top + 8;
    if (top + h > o.height - 8) top = Math.max(8, a.top - o.top - h - 8);
    pop.style.left = left + "px";
    pop.style.top = top + "px";
    input.focus();
    input.select();
  }

  if (!window._faceRenameBound) {
    window._faceRenameBound = true;
    document.addEventListener("mousedown", (ev) => {
      if (!ev.target.closest) return;
      if (!ev.target.closest("#face-rename-pop, .bio-face-reticle, .enlarged-face-node")) closeFaceRename();
    });
  }

  async function openBiometricProjection(burstId, imageId, initialLayout) {
    const req = ++overlayReq;
    activeOverlayBurstId = burstId;
    activeOverlayImageId = imageId;
    if (initialLayout) {
      overlayLayout = initialLayout;
    }
    hoveredFaceId = null;
    selectedFaceId = null;
    document.documentElement.style.setProperty('--speed', 0.5);

    const overlay = $("#facepanel");
    if (!overlay) return;
    const wasOpen = !overlay.hidden;
    // Opening fresh: stay hidden until the data is ready, so the scrim, HUD and faces animate in
    // together once. Already open (frame switch): keep what is on screen instead of blanking it.
    if (!wasOpen) {
      overlay.className = "bio-overlay bg-cyber-grid";
      overlay.replaceChildren();
    }
    const reveal = () => { overlay.hidden = false; enterFacePanel(overlay); };

    try {
      const [burstData, faceData, peopleData] = await Promise.all([
        burstId ? api("/api/burst/" + num(burstId)) : Promise.resolve(null),
        api(`/api/images/${num(imageId)}/faces`),
        api("/api/people").catch(() => ({ items: [] })),
      ]);
      if (req !== overlayReq) return;

      activeOverlayBurstData = burstData;
      activeOverlayMembers = (burstData && burstData.members) || [];
      activeOverlayPeopleNames = (peopleData.items || []).filter((p) => p.name).map((p) => p.name);
      activeOverlayStandaloneMember = activeOverlayMembers.length ? null : {
        id: imageId, filename: faceData.filename, width: faceData.width, height: faceData.height,
        orientation: faceData.orientation, taken_at: faceData.taken_at,
        is_best_shot: false, delete_approved: false,
      };
      const curMember = activeOverlayMembers.find((m) => m.id === imageId)
        || activeOverlayMembers[0] || activeOverlayStandaloneMember;
      if (!curMember) throw new Error("Frame not found");

      const items = faceData.items || [];
      if (!items.length) {
        reveal();
        overlay.innerHTML = `
          <div class="bio-backdrop" onclick="closeBiometricProjection()"></div>
          <div style="position:relative;z-index:30;padding:40px;text-align:center;color:#94a3b8;font-family:var(--font-mono, monospace);">
            No faces were found in this photo.
            <br><br>
            <button type="button" class="bio-close-btn" style="margin:0 auto;" onclick="closeBiometricProjection()">Close</button>
          </div>
        `;
        activeOverlayFaces = [];
        return;
      }
      const wording = { open: "Eyes open", possible: "Eyes may be closing", closed: "Eyes closed", unknown: "Eyes not measured" };

      activeOverlayFaces = items.map((f, idx) => {
        let x = 50, y = 50, w = 15, h = 20;
        if (f.box && f.box.length === 4) {
          w = Math.max(8, (f.box[2] - f.box[0]) * 100);
          h = Math.max(8, (f.box[3] - f.box[1]) * 100);
          x = ((f.box[0] + f.box[2]) / 2) * 100;
          y = ((f.box[1] + f.box[3]) / 2) * 100;
        } else {
          const count = Math.max(1, items.length);
          x = count > 1 ? 20 + (idx / (count - 1)) * 60 : 50;
          y = 45;
        }
        const color = f.is_blink ? "var(--err, #dc2626)" : FACE_PALETTE[idx % FACE_PALETTE.length];
        const stateKey = f.eye_status || "unknown";
        const eyeScoreText = f.eye_closed_score != null ? ` (${num(f.eye_closed_score).toFixed(2)})` : "";
        const confPercent = f.confidence ? Math.round(f.confidence * 100) + "%" : "99%";
        const statusLabel = f.is_blink ? "BLINK" : (stateKey === "open" ? "OPEN" : (stateKey === "possible" ? "POSSIBLE" : "DETECTED"));
        const statusBg = f.is_blink ? "var(--err, #dc2626)" : (stateKey === "open" ? "var(--ok, #059669)" : (stateKey === "possible" ? "var(--commit, #f6a315)" : "var(--mut, #475569)"));

        return {
          id: f.id,
          name: f.person_name || `Person ${idx + 1}`,
          personId: f.person_id != null ? f.person_id : null,
          role: f.is_blink ? "Closed Eyes" : (stateKey === "open" ? "Clear / Open" : "Subject"),
          conf: confPercent,
          sharpness: f.sharpness,
          eyeStatusText: (wording[stateKey] || wording.unknown) + eyeScoreText,
          eyeStatus: stateKey,
          isBlink: f.is_blink,
          statusLabel,
          statusBg,
          color,
          url: f.url || thumbUrl(imageId, "preview"),
          x, y, w, h
        };
      });

      if (activeOverlayFaces.length) {
        selectedFaceId = activeOverlayFaces[0].id;
      }

      reveal();
      renderBiometricOverlay();
      enterFacePanel(overlay);
    } catch (e) {
      if (req !== overlayReq) return;
      reveal();
      overlay.innerHTML = `
        <div class="bio-backdrop" onclick="closeBiometricProjection()"></div>
        <div style="position:relative;z-index:30;padding:40px;text-align:center;color:#ef4444;font-family:var(--font-mono, monospace);">
          Failed to load faces: ${esc(e.message || e)}
          <br><br>
          <button type="button" class="bio-close-btn" style="margin:0 auto;" onclick="closeBiometricProjection()">Close</button>
        </div>
      `;
    }
  }

  function renderFaceMatrixView() {
    const matrix = (activeOverlayBurstData && activeOverlayBurstData.face_matrix) || { subjects: [], rows: [] };
    const subjects = matrix.subjects || [];
    const rows = matrix.rows || [];
    const wording = { open: "Eyes open", possible: "Eyes may be closing", closed: "Eyes closed", unknown: "Eyes not measured" };

    if (!rows.length || !subjects.length) {
      return `
        <div class="bio-stage-matrix">
          <div class="bio-matrix-empty-state">
            <div style="font-size:18px; font-weight:700; color:#fff; margin-bottom:8px;">No faces detected in this burst</div>
            <p style="color:#94a3b8; font-size:13px; max-width:440px; margin:0 auto 16px auto;">The matrix comparison aligns tracked faces across all burst frames. No face data is available for this burst.</p>
            <button type="button" class="bio-action-btn best" style="display:inline-block;" onclick="setOverlayLayout('radial')">Return to Radial View</button>
          </div>
        </div>
      `;
    }

    const avatarSize = Math.max(68, Math.min(130, Math.round(84 * faceZoomMultiplier)));

    return `
      <div class="bio-stage-matrix">
        <div class="bio-matrix-scroll-pane">
          <table class="bio-matrix-table">
            <thead>
              <tr class="bio-matrix-head-row">
                <th class="bio-matrix-th bio-matrix-th-frame">
                  <span>Frame & Actions</span>
                </th>
                ${subjects.map((subj, sIdx) => {
                  const color = FACE_PALETTE[sIdx % FACE_PALETTE.length];
                  let posText = 'Subject ' + (sIdx + 1);
                  const K = subjects.length;
                  if (K === 2) {
                    posText = sIdx === 0 ? 'Left' : 'Right';
                  } else if (K === 3) {
                    posText = sIdx === 0 ? 'Left' : (sIdx === 1 ? 'Center' : 'Right');
                  } else if (K >= 4) {
                    posText = sIdx === 0 ? 'Left' : (sIdx === K - 1 ? 'Right' : (sIdx === 1 ? 'Center-Left' : (sIdx === K - 2 ? 'Center-Right' : 'Center')));
                  }
                  return `
                    <th class="bio-matrix-th bio-matrix-th-person" style="border-top: 3px solid ${color};">
                      <div class="bio-matrix-person-hd">
                        <span class="face-dot" style="background:${color};"></span>
                        <span class="bio-matrix-person-name">${esc(subj.name)}</span>
                      </div>
                      <div class="bio-matrix-person-pos">${posText}</div>
                    </th>
                  `;
                }).join('')}
              </tr>
            </thead>
            <tbody>
              ${rows.map((row, rIdx) => {
                const isBest = row.is_best_shot;
                const isDel = row.delete_approved;
                return `
                  <tr class="bio-matrix-tr${isBest ? ' is-best' : ''}${isDel ? ' is-deleted' : ''}">
                    <td class="bio-matrix-td bio-matrix-td-frame">
                      <div class="bio-matrix-frame-card">
                        <div class="bio-matrix-thumb-box" onclick="openBiometricProjection(${activeOverlayBurstId}, ${row.id}, 'radial')" title="View frame in single-photo inspection">
                          <img src="${thumbUrl(row.id, "preview")}" alt="${esc(row.filename)}" loading="lazy">
                          ${isBest ? '<span class="bio-matrix-badge-best">★ Best</span>' : ''}
                          ${row.has_blinks ? '<span class="bio-matrix-badge-blink">Blink</span>' : ''}
                          ${isDel ? '<span class="bio-matrix-badge-del">Marked</span>' : ''}
                        </div>
                        <div class="bio-matrix-frame-info">
                          <div class="bio-matrix-frame-title" title="${esc(row.filename)}">
                            <span class="bio-matrix-frame-idx">Frame 0${rIdx + 1}</span>
                            <span class="bio-matrix-frame-fn">${esc(row.filename)}</span>
                          </div>
                          <div class="bio-matrix-frame-actions">
                            ${isBest
                              ? `<button type="button" class="bio-action-btn best-active mini" disabled title="This is the best shot">★ Best shot</button>`
                              : `<button type="button" class="bio-action-btn best mini" onclick="makeOverlayBestShot(${row.id})" title="Set this frame as best shot">★ Set best</button>`}
                            <button 
                              type="button" 
                              class="bio-action-btn delete mini${isDel ? ' on' : ''}" 
                              onclick="toggleOverlayMarkDelete(${row.id}, ${isBest ? 'true' : 'false'}, ${isDel ? 'true' : 'false'})" 
                              title="${isBest ? 'Marking best shot for delete' : 'Mark this frame for delete'}"
                            >
                              ${isDel ? 'Marked (undo)' : 'Mark delete'}
                            </button>
                          </div>
                        </div>
                      </div>
                    </td>
                    ${subjects.map((subj, sIdx) => {
                      const f = (row.faces && row.faces[sIdx]) || null;
                      const color = FACE_PALETTE[sIdx % FACE_PALETTE.length];
                      if (!f) {
                        return `
                          <td class="bio-matrix-td bio-matrix-td-person">
                            <div class="bio-matrix-empty-cell" title="${esc(subj.name)} not detected in this frame">
                              <div class="bio-matrix-empty-circle" style="width:${avatarSize}px; height:${avatarSize}px;">
                                <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-3.3 3.6-6 8-6s8 2.7 8 6"/></svg>
                              </div>
                              <span class="bio-matrix-empty-text">Not detected</span>
                            </div>
                          </td>
                        `;
                      }
                      const stateKey = f.eye_status || "unknown";
                      const isBlink = f.is_blink || stateKey === "closed";
                      const statusLabel = isBlink ? "BLINK" : (stateKey === "open" ? "OPEN" : (stateKey === "possible" ? "POSSIBLE" : "DETECTED"));
                      const statusBg = isBlink ? "var(--err, #dc2626)" : (stateKey === "open" ? "var(--ok, #059669)" : (stateKey === "possible" ? "var(--commit, #f6a315)" : "var(--mut, #475569)"));
                      const eyeScoreText = f.eye_closed_score != null ? ` (${num(f.eye_closed_score).toFixed(2)})` : "";
                      const eyeText = (wording[stateKey] || wording.unknown) + eyeScoreText;

                      return `
                        <td class="bio-matrix-td bio-matrix-td-person">
                          <figure class="facechip bio-matrix-chip${isBlink ? ' blink' : ''}" style="width:${avatarSize + 32}px;">
                            <div class="bio-matrix-avatar-wrap" style="width:${avatarSize}px; height:${avatarSize}px;">
                              <div class="bio-matrix-face-avatar" style="border-color:${isBlink ? 'var(--err, #dc2626)' : color}; box-shadow: 0 4px 16px rgba(0,0,0,0.6), 0 0 12px ${isBlink ? 'rgba(220,38,38,0.45)' : color + '40'};">
                                <img src="${f.url}" alt="${esc(subj.name)}" loading="lazy">
                              </div>
                              <div class="bio-matrix-chip-badge" style="background:${statusBg}; color:#fff;">
                                ${statusLabel}
                              </div>
                            </div>
                            <figcaption class="bio-matrix-caption">
                              <span class="bio-matrix-eye-text">${esc(eyeText)}</span>
                              ${f.sharpness ? `<span class="bio-matrix-sharpness">Sharpness ${num(f.sharpness).toFixed(1)}</span>` : ''}
                            </figcaption>
                          </figure>
                        </td>
                      `;
                    }).join('')}
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  function renderBiometricOverlay() {
    const overlay = $("#facepanel");
    if (!overlay || !activeOverlayImageId) return;

    const curMember = activeOverlayMembers.find((m) => m.id === activeOverlayImageId)
      || activeOverlayMembers[0] || activeOverlayStandaloneMember;
    if (!curMember) return;

    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const centerX = vw / 2;
    const centerY = vh / 2 + 6;

    const baseNodeSize = vw < 640 ? 88 : 124;
    const nodeDiameter = Math.round(baseNodeSize * faceZoomMultiplier);
    // Prevent overlap when there are many faces by calculating safe diameter and spacing
    const count = activeOverlayFaces.length;
    let safeNodeDiameter = nodeDiameter;
    // For many faces (e.g. 4+ per flank), reduce node size so they never touch or overlap
    const maxFlankCount = Math.ceil(count / 2);
    if (overlayLayout === 'radial') {
      const availH = vh - 180;
      if (maxFlankCount * (safeNodeDiameter + 24) > availH) {
        safeNodeDiameter = Math.max(68, Math.floor((availH / maxFlankCount) - 20));
      }
    } else if (overlayLayout === 'cinema') {
      const availW = vw - 80;
      if (count * (safeNodeDiameter + 24) > availW) {
        safeNodeDiameter = Math.max(68, Math.floor((availW / count) - 16));
      }
    }

    // Use oriented dimensions so photos taken in portrait orientation don't get cropped
    let rawW = curMember.width || 4;
    let rawH = curMember.height || 3;
    const orient = curMember.orientation || 1;
    if (orient >= 5 && orient <= 8) {
      const tmp = rawW;
      rawW = rawH;
      rawH = tmp;
    }
    const aspect = rawH / rawW;

    let sourceBox = { x: centerX - 210, y: centerY - 145, w: 420, h: 290 };
    const faceNodesPositions = [];

    if (overlayLayout === 'radial') {
      if (vw >= 1024) {
        let srcW = Math.min(460, vw * 0.35);
        let srcH = Math.round(srcW * aspect);
        const maxH = vh * 0.58;
        if (srcH > maxH) {
          srcH = Math.round(maxH);
          srcW = Math.round(srcH / aspect);
        }
        sourceBox = { x: centerX - srcW / 2, y: centerY - srcH / 2 - 15, w: srcW, h: srcH };

        const half = Math.ceil(count / 2);
        const leftFaces = activeOverlayFaces.slice(0, half);
        const rightFaces = activeOverlayFaces.slice(half);
        const flankOffsetX = srcW / 2 + safeNodeDiameter / 2 + 80;

        // Position left faces in clean non-overlapping vertical column
        const leftH = (leftFaces.length - 1) * (safeNodeDiameter + 28);
        const leftStartY = centerY - 15 - leftH / 2;
        leftFaces.forEach((f, i) => {
          faceNodesPositions.push({
            face: f,
            cx: centerX - flankOffsetX,
            cy: leftStartY + i * (safeNodeDiameter + 28)
          });
        });

        // Position right faces in clean non-overlapping vertical column
        const rightH = (rightFaces.length - 1) * (safeNodeDiameter + 28);
        const rightStartY = centerY - 15 - rightH / 2;
        rightFaces.forEach((f, i) => {
          faceNodesPositions.push({
            face: f,
            cx: centerX + flankOffsetX,
            cy: rightStartY + i * (safeNodeDiameter + 28)
          });
        });
      } else {
        let srcW = Math.min(340, vw * 0.84);
        let srcH = Math.round(srcW * aspect);
        const maxH = vh * 0.38;
        if (srcH > maxH) {
          srcH = Math.round(maxH);
          srcW = Math.round(srcH / aspect);
        }
        sourceBox = { x: centerX - srcW / 2, y: 95, w: srcW, h: srcH };
        const spanW = Math.min(vw * 0.9, count * (safeNodeDiameter + 16));
        const startX = centerX - spanW / 2 + safeNodeDiameter / 2;
        const stepX = count > 1 ? (spanW - safeNodeDiameter) / (count - 1) : 0;
        activeOverlayFaces.forEach((f, idx) => {
          faceNodesPositions.push({ face: f, cx: startX + idx * stepX, cy: sourceBox.y + sourceBox.h + 90 });
        });
      }
    } else if (overlayLayout === 'split') {
      if (vw >= 1024) {
        let srcW = Math.min(480, vw * 0.38);
        let srcH = Math.round(srcW * aspect);
        const maxH = vh * 0.62;
        if (srcH > maxH) {
          srcH = Math.round(maxH);
          srcW = Math.round(srcH / aspect);
        }
        sourceBox = { x: Math.max(56, centerX - 540), y: centerY - srcH / 2 - 12, w: srcW, h: srcH };

        const rightAreaStart = sourceBox.x + sourceBox.w + 100;
        const rightAreaWidth = Math.min(620, vw - rightAreaStart - 56);
        const cols = count <= 3 ? 1 : count <= 8 ? 2 : 3;
        const rows = Math.ceil(count / cols);
        const colSpacing = cols > 1 ? Math.min(220, rightAreaWidth / (cols - 0.2)) : 0;
        const rowSpacing = safeNodeDiameter + 32;

        const gridOriginX = rightAreaStart + safeNodeDiameter / 2 + 20;
        const gridOriginY = centerY - ((rows - 1) * rowSpacing) / 2 - 15;

        activeOverlayFaces.forEach((f, idx) => {
          const colIdx = idx % cols;
          const rowIdx = Math.floor(idx / cols);
          faceNodesPositions.push({
            face: f,
            cx: gridOriginX + colIdx * colSpacing,
            cy: gridOriginY + rowIdx * rowSpacing
          });
        });
      } else {
        let srcW = Math.min(340, vw * 0.84);
        let srcH = Math.round(srcW * aspect);
        const maxH = vh * 0.38;
        if (srcH > maxH) {
          srcH = Math.round(maxH);
          srcW = Math.round(srcH / aspect);
        }
        sourceBox = { x: centerX - srcW / 2, y: 95, w: srcW, h: srcH };
        const spanW = Math.min(vw * 0.9, count * (safeNodeDiameter + 16));
        const startX = centerX - spanW / 2 + safeNodeDiameter / 2;
        const stepX = count > 1 ? (spanW - safeNodeDiameter) / (count - 1) : 0;
        activeOverlayFaces.forEach((f, idx) => {
          faceNodesPositions.push({ face: f, cx: startX + idx * stepX, cy: sourceBox.y + sourceBox.h + 90 });
        });
      }
    } else if (overlayLayout === 'cinema') {
      let srcW = Math.min(460, vw * 0.78);
      let srcH = Math.round(srcW * aspect);
      const maxH = vh * 0.44;
      if (srcH > maxH) {
        srcH = Math.round(maxH);
        srcW = Math.round(srcH / aspect);
      }
      sourceBox = { x: centerX - srcW / 2, y: Math.max(84, centerY - srcH - 56), w: srcW, h: srcH };

      const dockSpanWidth = Math.min(vw * 0.88, count * (safeNodeDiameter + 24));
      const startX = centerX - dockSpanWidth / 2 + safeNodeDiameter / 2;
      const stepX = count > 1 ? (dockSpanWidth - safeNodeDiameter) / (count - 1) : 0;

      activeOverlayFaces.forEach((f, idx) => {
        faceNodesPositions.push({
          face: f,
          cx: count === 1 ? centerX : startX + idx * stepX,
          cy: sourceBox.y + sourceBox.h + safeNodeDiameter / 2 + 50
        });
      });
    }

    overlay.dataset.sourceBox = JSON.stringify(sourceBox);
    overlay.dataset.nodeDiameter = String(safeNodeDiameter);
    window._currentFaceNodes = faceNodesPositions;

    const matrixRows = (activeOverlayBurstData && activeOverlayBurstData.face_matrix && activeOverlayBurstData.face_matrix.rows) || [];
    const matrixSubjects = (activeOverlayBurstData && activeOverlayBurstData.face_matrix && activeOverlayBurstData.face_matrix.subjects) || [];

    // Entry animations only play on the first render; later swaps (layout, zoom, frame) stay still.
    overlay.classList.toggle("bio-rerender", !!overlay.querySelector(".bio-hud-top"));
    overlay.innerHTML = `
      <div class="bio-backdrop" onclick="closeBiometricProjection()"></div>
      <svg id="overlay-svg-canvas" class="bio-svg-canvas"></svg>

      <!-- TOP HUD -->
      <header class="bio-hud-top">
        <div class="bio-hud-left">
          ${overlayLayout === 'matrix' ? `
            <div>
              <h2 class="bio-photo-title">All Faces Matrix</h2>
              <p class="bio-photo-sub">${fmtN(matrixRows.length || activeOverlayMembers.length)} frames • ${fmtN(matrixSubjects.length)} tracked subjects</p>
            </div>
          ` : `
            <div>
              <h2 class="bio-photo-title">${esc(curMember.filename)}</h2>
              <p class="bio-photo-sub">${num(curMember.width)}×${num(curMember.height)} • ${count} detected faces</p>
            </div>
          `}
        </div>

        <!-- Layout Selector -->
        <div class="bio-layout-bar">
          <button type="button" class="bio-layout-btn${overlayLayout === 'radial' ? ' on' : ''}" onclick="setOverlayLayout('radial')">
            <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="9"/></svg>
            <span>Radial</span>
          </button>
          <button type="button" class="bio-layout-btn${overlayLayout === 'split' ? ' on' : ''}" onclick="setOverlayLayout('split')">
            <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7"/></svg>
            <span>Split</span>
          </button>
          <button type="button" class="bio-layout-btn${overlayLayout === 'cinema' ? ' on' : ''}" onclick="setOverlayLayout('cinema')">
            <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="11" rx="2"/><rect x="3" y="16" width="5" height="5" rx="1"/><rect x="10" y="16" width="5" height="5" rx="1"/><rect x="17" y="16" width="5" height="5" rx="1"/></svg>
            <span>Cinema</span>
          </button>
          <button type="button" class="bio-layout-btn${overlayLayout === 'matrix' ? ' on' : ''}" onclick="setOverlayLayout('matrix')">
            <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z"/></svg>
            <span>Matrix</span>
          </button>
        </div>

        <!-- Frame Actions (Best Shot & Mark Delete) & Zoom & Close -->
        <div class="bio-hud-right">
          ${overlayLayout === 'matrix' ? `
          <div class="bio-actions-bar">
            <span style="font-size:12px; color:var(--blue-200, #bfdbfe); opacity:0.85; font-weight:500;">Same subject per column</span>
          </div>
          ` : activeOverlayBurstId ? `
          <!-- Frame Actions (burst only: a standalone photo has no best-shot/delete concept here) -->
          <div class="bio-actions-bar">
            ${curMember.is_best_shot
              ? `<button type="button" class="bio-action-btn best-active" disabled title="This is the best shot">★ Best shot</button>`
              : `<button type="button" class="bio-action-btn best" onclick="makeOverlayBestShot(${curMember.id})" title="Set this frame as best shot">★ Set best</button>`}
            <button
              type="button"
              class="bio-action-btn delete${curMember.delete_approved ? ' on' : ''}"
              onclick="toggleOverlayMarkDelete(${curMember.id}, ${curMember.is_best_shot ? 'true' : 'false'}, ${curMember.delete_approved ? 'true' : 'false'})"
              title="${curMember.is_best_shot ? 'Marking best shot for delete' : 'Mark this frame for delete'}"
            >
              ${curMember.delete_approved ? 'Marked (undo)' : 'Mark delete'}
            </button>
          </div>
          ` : ''}

          <div class="bio-zoom-bar">
            <span class="bio-ctrl-label">Zoom:</span>
            <button type="button" class="bio-zoom-btn${faceZoomMultiplier === 0.86 ? ' on' : ''}" onclick="setFaceZoom(0.86)">1.5x</button>
            <button type="button" class="bio-zoom-btn${faceZoomMultiplier === 1.08 ? ' on' : ''}" onclick="setFaceZoom(1.08)">2.5x</button>
            <button type="button" class="bio-zoom-btn${faceZoomMultiplier === 1.28 ? ' on' : ''}" onclick="setFaceZoom(1.28)">3.5x</button>
          </div>

          <button type="button" class="bio-close-btn" onclick="closeBiometricProjection()" title="Close (ESC)">
            <span>ESC</span>
            <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 18L18 6M6 6l12 12"/></svg>
          </button>
        </div>
      </header>

      <!-- CENTER STAGE -->
      <div id="overlay-stage" class="bio-stage${overlayLayout === 'matrix' ? ' matrix-mode' : ''}">
        ${overlayLayout === 'matrix' ? renderFaceMatrixView() : `
        <!-- SOURCE PHOTO BOX -->
        <div 
          style="left:${sourceBox.x}px; top:${sourceBox.y}px; width:${sourceBox.w}px; height:${sourceBox.h}px;"
          class="bio-source-box"
        >
          <div class="bio-scan" aria-hidden="true"></div>
          <img 
            src="${thumbUrl(curMember.id, "preview")}" 
            alt="${esc(curMember.filename)}"
            class="bio-source-img${hoveredFaceId ? ' dimmed' : ''}"
          />
          <div class="bio-source-badge-bar">
            <span>SOURCE PHOTO • ${esc(curMember.filename)}</span>
            <span style="color:var(--primary, #2b9af6);">${count} FACES</span>
          </div>

          <!-- Face reticles on photo -->
          ${activeOverlayFaces.map((f, idx) => {
            const isHovered = hoveredFaceId === f.id;
            return `
              <div
                class="bio-face-reticle"
                data-reticle-id="${f.id}"
                onmouseenter="setHoveredFace(${f.id})"
                onmouseleave="setHoveredFace(null)"
                onclick="selectFaceInOverlay(${f.id}, this)"
                style="left:${f.x - f.w/2}%; top:${f.y - f.h/2}%; width:${f.w}%; height:${f.h}%; border-color:${f.color}; box-shadow:0 0 ${isHovered ? '24px' : '10px'} ${f.color}; animation-delay:calc(${(f.y / 100).toFixed(3)} * 0.35s / var(--speed, 0.5) - 60ms);"
              >
                <span class="tick tick-tl"></span>
                <span class="tick tick-tr"></span>
                <span class="tick tick-bl"></span>
                <span class="tick tick-br"></span>
                <span class="bio-face-reticle-tag" style="background:${f.color};">
                  ${esc(f.name)}
                </span>
              </div>
            `;
          }).join('')}
        </div>

        <!-- ENLARGED FACE NODES (Clean and uncrossed) -->
        ${faceNodesPositions.map((node, idx) => {
          const f = node.face;
          const isHovered = hoveredFaceId === f.id;
          const diam = isHovered ? Math.round(safeNodeDiameter * 1.08) : safeNodeDiameter;
          return `
            <figure
              class="enlarged-face-node facechip${f.isBlink ? ' blink' : ''}"
              data-node-id="${f.id}"
              onmouseenter="setHoveredFace(${f.id})"
              onmouseleave="setHoveredFace(null)"
              onclick="selectFaceInOverlay(${f.id}, this)"
              style="left:${node.cx}px; top:${node.cy}px; width:${diam}px; height:${diam}px; transform:translate(-50%, -50%);"
            >
              <div class="bio-node-enter" style="animation-delay:calc(0.35s / var(--speed, 0.5) + ${60 + idx * 40}ms);">
                <!-- Outer Rotating Ring -->
                <svg class="hud-ring-spin" viewBox="0 0 120 120">
                  <circle cx="60" cy="60" r="56" fill="none" stroke="${f.color}" stroke-width="${isHovered ? '2.4' : '1.4'}" stroke-dasharray="20 10 6 10" stroke-opacity="${isHovered ? '1' : '0.65'}"/>
                </svg>
                <!-- Inner Counter-Rotating Ring -->
                <svg class="hud-ring-reverse" viewBox="0 0 120 120">
                  <circle cx="60" cy="60" r="56" fill="none" stroke="${f.color}" stroke-width="1" stroke-dasharray="4 14" stroke-opacity="0.7"/>
                </svg>
                <!-- Face Cutout Circle -->
                <div class="face-crop-circle" style="border-color:${f.color}; box-shadow: 0 16px 45px rgba(0,0,0,0.85), 0 0 ${isHovered ? '32px' : '16px'} ${f.color}88;">
                  <img src="${f.url}" alt="${esc(f.name)}" loading="lazy">
                </div>
                <!-- Badge -->
                <div class="face-badge-pill" style="background:${f.statusBg}; color:#fff;">
                  ${f.statusLabel}
                </div>
                <!-- Figcaption Nameplate -->
                <figcaption class="face-nameplate" style="border-color:${isHovered ? f.color : 'rgba(51,65,85,0.8)'};">
                  <div class="face-nameplate-title">
                    <span class="face-dot" style="background:${f.color}"></span>
                    <span>${esc(f.name)}</span>
                  </div>
                  <div class="face-nameplate-sub">
                    <span>${esc(f.eyeStatusText)}</span>
                    ${f.sharpness ? `<span>• Sharpness ${num(f.sharpness).toFixed(1)}</span>` : ''}
                  </div>
                </figcaption>
              </div>
            </figure>
          `;
        }).join('')}
        `}
      </div>

      <!-- BOTTOM HUD -->
      <footer class="bio-hud-bottom"${overlayLayout === 'matrix' ? ' style="display:none;"' : ''}>
        ${activeOverlayBurstId ? `
        <div class="bio-frame-switcher-wrap">
          <span class="bio-frame-label">Frames:</span>
          <div id="overlay-card-switcher" class="bio-frame-switcher">
            ${activeOverlayMembers.map((m, idx) => `
              <button
                type="button"
                class="bio-frame-btn${m.id === activeOverlayImageId ? ' on' : ''}"
                onclick="openBiometricProjection(${activeOverlayBurstId}, ${m.id})"
                title="${esc(m.filename)}"
              >
                ${m.is_best_shot ? '★ ' : ''}Frame 0${idx + 1}
              </button>
            `).join('')}
          </div>
        </div>
        ` : ''}

        <div id="selected-face-drawer" class="bio-face-inspector"></div>
      </footer>
    `;

    updateBiometricSpotlight();
  }

  function updateBiometricSpotlight() {
    const overlay = $("#facepanel");
    if (!overlay || overlayLayout === 'matrix' || !window._currentFaceNodes) return;

    // Clear SVG tether lines per user request ("remove the line from the photo to the circular zoom face.. no need of it")
    const svgEl = $("#overlay-svg-canvas");
    if (svgEl) {
      svgEl.innerHTML = '';
    }

    const activeFace = activeOverlayFaces.find((f) => f.id === selectedFaceId) || activeOverlayFaces[0];
    const drawerEl = $("#selected-face-drawer");
    if (drawerEl && activeFace && drawerEl.dataset.faceId !== String(activeFace.id)) {
      drawerEl.dataset.faceId = String(activeFace.id);
      drawerEl.innerHTML = `
        <div style="display:flex; align-items:center; gap:12px;">
          <div
            style="width:36px; height:36px; border-radius:50%; overflow:hidden; border:2px solid ${activeFace.color}; flex:none;"
          >
            <img src="${activeFace.url}" style="width:100%; height:100%; object-fit:cover;" alt="">
          </div>
          <div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span style="font-size:13px; font-weight:700; color:#fff; font-family:var(--font-body);">${esc(activeFace.name)}</span>
              <span style="font-size:10px; font-family:var(--font-mono); font-weight:700; padding:1px 6px; border-radius:4px; background:${activeFace.statusBg}; color:#fff;">
                ${esc(activeFace.statusLabel)}
              </span>
            </div>
            <div style="font-size:11px; font-family:var(--font-body); color:var(--blue-100, #cfeafe); opacity:0.85; margin-top:1px;">
              ${esc(activeFace.eyeStatusText)}${activeFace.sharpness ? ` • Sharpness ${num(activeFace.sharpness).toFixed(1)}` : ''}
            </div>
          </div>
        </div>
      `;
    }

    $$(".enlarged-face-node", overlay).forEach((node) => {
      const id = Number(node.dataset.nodeId);
      const isHovered = hoveredFaceId === id;
      const isDimmed = hoveredFaceId && !isHovered;
      node.style.opacity = isDimmed ? '0.35' : '1';
      node.style.filter = isDimmed ? 'brightness(0.75)' : 'none';
    });

    $$(".bio-face-reticle", overlay).forEach((ret) => {
      const id = Number(ret.dataset.reticleId);
      const isHovered = hoveredFaceId === id;
      const isDimmed = hoveredFaceId && !isHovered;
      ret.style.opacity = isDimmed ? '0.35' : '1';
    });
  }

  // Window listeners for projection
  if (!window._bioResizeBound) {
    window._bioResizeBound = true;
    window.addEventListener("resize", () => {
      const overlay = $("#facepanel");
      if (overlay && !overlay.hidden && activeOverlayImageId) {
        renderBiometricOverlay();
      }
    });
  }

  async function makeOverlayBestShot(imageId) {
    if (!activeOverlayBurstId) return;
    await makeBestShot(activeOverlayBurstId, imageId);
    await openBiometricProjection(activeOverlayBurstId, imageId);
  }

  async function toggleOverlayMarkDelete(imageId, isBest, currentlyApproved) {
    if (!activeOverlayBurstId) return;
    const approve = !currentlyApproved;
    if (approve && isBest) {
      const ok = await showConfirmDialog({
        title: "Mark the best shot for delete?",
        message: "This is the current best shot of the burst. If you still want to delete it, consider promoting another frame to best first.",
        confirmText: "Mark delete", variant: "warning" });
      if (!ok) return;
    }
    await markBurstDeletes(activeOverlayBurstId, [imageId], approve);
    await openBiometricProjection(activeOverlayBurstId, imageId);
  }

  async function reassignFace(faceId, opts) {
    try {
      await post(`/api/faces/${num(faceId)}/reassign`, opts || {});
      toast("Reassigned", "ok");
      await openBiometricProjection(activeOverlayBurstId, activeOverlayImageId);
    } catch (e) { toast(e.message, "err"); }
  }

  // Expose handlers to window for inline onclick attributes
  window.openBiometricProjection = openBiometricProjection;
  window.closeBiometricProjection = closeBiometricProjection;
  window.setOverlayLayout = setOverlayLayout;
  window.setFaceZoom = setFaceZoom;
  window.setSpeed = setSpeed;
  window.setHoveredFace = setHoveredFace;
  window.selectFaceInOverlay = selectFaceInOverlay;
  window.makeOverlayBestShot = makeOverlayBestShot;
  window.toggleOverlayMarkDelete = toggleOverlayMarkDelete;
  window.reassignFace = reassignFace;

  // Compatibility alias
  async function showFaces(imageId, host, filename) {
    if (currentBurstId) {
      await openBiometricProjection(currentBurstId, imageId);
    }
  }

  async function rejectBlinks(burstId, count) {
    const yes = await showConfirmDialog({
      title: "Reject all blinks?",
      message: `${fmtN(count)} alternate${count === 1 ? "" : "s"} with closed eyes will go to Junk review. The best shot stays where it is. No files are moved, and you can restore them.`,
      confirmText: "Reject blinks", variant: "warning" });
    if (!yes) return;
    try {
      const r = await post(`/api/bursts/${num(burstId)}/reject-blinks`, {});
      toast(r.moved ? `Sent ${fmtN(r.moved)} to Junk review` : (r.note || "Nothing to reject"), "ok");
      await showBurst(burstId);
    } catch (e) { toast(e.message, "err"); }
  }

  async function restoreBlinks(burstId) {
    try {
      const r = await post(`/api/bursts/${num(burstId)}/restore-blinks`, {});
      toast(`Restored ${fmtN(r.restored)}`, "ok");
      await showBurst(burstId);
    } catch (e) { toast(e.message, "err"); }
  }

  // One sentence for what the group matrix found in a burst; null when it is not a group shot.
  function groupSummary(group, nameOf) {
    if (!group || group.status === "not_group") return null;
    const best = nameOf(group.best_id);
    if (group.status === "perfect") {
      return { tone: "ok", text: `Everyone's eyes open in ${best}. This is the frame to keep.` };
    }
    if (group.status === "composite") {
      return { tone: "warn", text: `No single frame is perfect. Keep ${best} and ${nameOf(group.partner_id)}: between them everyone looks good, and Photocure can combine them into one photo for you.` };
    }
    return { tone: "warn", text: `No frame has everyone looking good. ${best} is the closest.` };
  }

  const BURST_ICONS = {
    star: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></svg>',
    check: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12l5 5 9-10"/></svg>',
    trash: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>',
    pick: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/></svg>',
    face: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M8 14s1.5 2 4 2 4-2 4-2M9 9h.01M15 9h.01"/></svg>',
  };

  const GM_WORDS = { open: "eyes open", possible: "eyes may be closing", closed: "eyes closed",
    unknown: "eyes not measured", missing: "not seen in this frame" };

  async function showBurst(id) {
    currentBurstId = id;
    closeBiometricProjection();
    clearPicks();
    $$(".burstrow", $("#burstlist")).forEach((r) => {
      const on = num(r.dataset.burst) === id;
      r.classList.toggle("on", on);
      r.classList.toggle("is-active", on);
    });
    const data = await api("/api/burst/" + num(id));
    state.burst = data;
    const group = data.group;
    const gmById = {};
    ((group && group.frames) || []).forEach((f) => { gmById[f.id] = f; });
    const nameById = Object.fromEntries(data.members.map((m) => [m.id, m.filename]));
    const summary = groupSummary(group, (i) => nameById[i] || "this frame");
    const banner = summary ? `<div class="gm-banner ${summary.tone}" role="status">${esc(summary.text)}</div>` : "";
    const gmBadge = (m) => {
      if (!summary) return "";
      if (group.status === "perfect" && m.id === group.best_id) return '<span class="tag-badge best gm-badge">All eyes open</span>';
      if (group.status === "composite" && (m.id === group.best_id || m.id === group.partner_id)) return '<span class="tag-badge best gm-badge">Composite pair</span>';
      if (group.status === "best_effort" && m.id === group.best_id) return '<span class="tag-badge warn gm-badge">Closest</span>';
      return "";
    };
    const gmDetail = (m) => {
      const f = gmById[m.id];
      if (!f) return "";
      const dots = f.subjects.map((s, i) => `<i class="gm-dot" data-s="${esc(s.state)}" title="Person ${i + 1}: ${esc(GM_WORDS[s.state] || GM_WORDS.unknown)}"></i>`).join("");
      return `<div class="gm-detail"><div class="gm-dots" role="img" aria-label="Faces in this frame">${dots}</div>
        <div class="gm-bar" title="Weakest face in this frame"><span style="width:${Math.round(num(f.min_quality) * 100)}%"></span></div></div>`;
    };
    $("#burstdetail").innerHTML = `
      <div class="burst-header anim-fade">
        <div class="detail-head">
          <h1 class="burst-title" id="title">${esc(data.burst)}</h1>
          <span class="count" id="count">${fmtN(data.members.length)} frames</span>
        </div>
        ${banner}
        <div class="toolbar" id="blinkbar"></div>
      </div>
      <section class="burstgrid grid anim-in" id="grid" aria-label="Frames">` +
      data.members.map((m) => {
        const sharp = m.sharpness_norm != null ? num(m.sharpness_norm) : 1.0;
        const face = m.face_confidence != null ? num(m.face_confidence) : (m.face_count ? 0.9 : 0.85);
        const eyesClosed = m.eye_closed != null ? num(m.eye_closed) : (m.has_blinks ? 0.5 : 0.0);
        const sc = Math.min(100, Math.max(1, Math.round(100 * (0.4 * sharp + 0.4 * face + 0.2 * (1 - eyesClosed)))));

        const sharpText = m.sharpness_norm != null ? sharp.toFixed(2) : "1.00";
        const faceText = m.face_confidence != null ? face.toFixed(2) : (m.face_count ? "0.90" : "—");



        const dim = m.width && m.height ? `${num(m.width)}×${num(m.height)}` : "";
        const time = m.taken_at ? String(m.taken_at).slice(11, 19) : "";
        const metaParts = [dim, time].filter(Boolean);
        if (m.face_count > 1) metaParts.push(`${num(m.face_count)} faces`);
        const metaText = metaParts.join(" · ");

        return `<article class="card-wrap${m.is_best_shot ? " is-best" : ""}${m.delete_approved ? " is-marked" : ""}" data-file="${esc(m.filename)}" data-id="${num(m.id)}">
          <figure class="card" data-id="${num(m.id)}">
            <div class="photo thumb" style="cursor:pointer;" title="View photo">
              <img loading="lazy" src="${thumbUrl(m.id, "medium")}" alt="${esc(m.filename)}" onerror="this.style.opacity=.15">
            </div>
            <div class="caption">
              <div class="burst-row-title">
                <div class="file" title="${esc(m.filename)}">${esc(m.filename)}</div>
                <span class="tag-badge ${m.is_best_shot ? "best" : (m.delete_approved ? "warn" : "")}">${m.is_best_shot ? "Best shot" : (m.delete_approved ? "Marked delete" : "Alternate")}</span>${gmBadge(m)}
              </div>
              <div class="meta">${esc(metaText)}</div>
              <div class="burst-stats">
                <span class="stat-pill" title="Overall quality score">Score <b>${sc}</b></span>
                <span class="stat-pill" title="Sharpness">Sharp <b>${sharpText}</b></span>
                <span class="stat-pill" title="Face quality">Face <b>${faceText}</b></span>
                <span class="stat-pill ${m.eye_closed > 0.2 || m.has_blinks ? "warn" : "ok"}">${m.eye_closed > 0.2 || m.has_blinks ? "Blink" : "Eyes open"}</span>
              </div>
            </div>
            <div class="actions actions-icon">
              <button class="btn btn-pick" data-pick="${num(m.id)}" aria-pressed="false" title="Select to keep" aria-label="Select ${esc(m.filename)} to keep">${BURST_ICONS.pick}</button>
              ${m.is_best_shot
                ? `<button class="btn btn-best is-best" disabled title="This is the best shot" aria-label="This is the best shot">${BURST_ICONS.check}</button>`
                : `<button class="btn btn-best" data-makebest="${num(m.id)}" title="Set as best shot" aria-label="Set as best shot">${BURST_ICONS.star}</button>`}
              <button class="btn btn-del${m.delete_approved ? " is-del" : ""}${m.is_best_shot ? " warn-best" : ""}" data-markdelete="${num(m.id)}" data-is-best="${m.is_best_shot ? "1" : "0"}" aria-pressed="${m.delete_approved ? "true" : "false"}" title="${m.delete_approved ? "Marked for delete (click to unmark)" : (m.is_best_shot ? "Mark the best shot for delete" : "Mark this frame for delete")}" aria-label="${m.delete_approved ? "Marked for delete (click to unmark)" : "Mark this frame for delete"}">${BURST_ICONS.trash}</button>
              ${m.face_count ? `<button class="btn btn-faces" data-faces="${num(m.id)}" data-burst="${num(id)}" data-count="${num(m.face_count)}" title="Inspect faces" aria-label="Inspect faces">${BURST_ICONS.face}</button>` : ""}
            </div>
          </figure>
        </article>`;
      }).join("") + `</section>`;

    const bar = $("#blinkbar");
    const addBtn = (label, fn, isPrimary = false) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "pill" + (isPrimary ? " primary" : "");
      b.textContent = label;
      b.onclick = fn;
      bar.appendChild(b);
    };
    if (group && group.status === "composite") {
      addBtn("Create fixed photo", () => makeComposite(id, "fixed"), true);
      addBtn("Export layered PSD", () => makeComposite(id, "layered"));
    }
    if (data.face_matrix && data.face_matrix.subjects && data.face_matrix.subjects.length > 0) {
      addBtn("Compare all faces (Matrix)", () => openBiometricProjection(id, data.members[0].id, "matrix"), true);
    }
    const others = data.members.filter((m) => !m.is_best_shot);
    const unmarked = others.filter((m) => !m.delete_approved).map((m) => m.id);
    const marked = others.filter((m) => m.delete_approved).map((m) => m.id);
    if (unmarked.length) addBtn(`Mark ${fmtN(unmarked.length)} other frame${unmarked.length === 1 ? "" : "s"} for delete`, () => markBurstDeletes(id, unmarked, true));
    if (marked.length) addBtn(`Clear ${fmtN(marked.length)} delete mark${marked.length === 1 ? "" : "s"}`, () => markBurstDeletes(id, marked, false));
    if (data.can_reject_blinks) addBtn("Reject all blinks", () => rejectBlinks(id, data.reject_count));
    if (data.rejected_blinks) addBtn(`Restore ${data.rejected_blinks} from Junk review`, () => restoreBlinks(id));
    $$("[data-makebest]", $("#burstdetail")).forEach((b) => {
      b.onclick = () => makeBestShot(id, num(b.dataset.makebest));
    });
    $$("[data-markdelete]", $("#burstdetail")).forEach((b) => {
      b.onclick = async () => {
        const approve = b.getAttribute("aria-pressed") !== "true";
        if (approve && b.dataset.isBest === "1") {
          const ok = await showConfirmDialog({
            title: "Mark the best shot for delete?",
            message: "This is the current best shot of the burst. If you still want to delete it, consider promoting another frame to best first.",
            confirmText: "Mark delete", variant: "warning" });
          if (!ok) return;
        }
        markBurstDeletes(id, [num(b.dataset.markdelete)], approve);
      };
    });
  }

  async function markBurstDeletes(burstId, ids, approved) {
    try {
      await post("/api/deletes/approve", { ids, approved });
      const undo = approved ? { label: "Undo", fn: async () => {
        await post("/api/deletes/approve", { ids, approved: false });
        await showBurst(burstId);
      } } : undefined;
      toast(approved
        ? `Marked ${fmtN(ids.length)} for delete. They are moved to the trash folder when you apply with the trash option.`
        : `Delete mark cleared on ${fmtN(ids.length)}`, "ok", undo);
      await showBurst(burstId);
    } catch (e) { toast(e.message, "err"); }
  }

  async function makeComposite(burstId, mode) {
    toast(mode === "fixed" ? "Combining the frames..." : "Building the layered file...", "ok");
    try {
      const r = await post(`/api/bursts/${num(burstId)}/composite`, { mode });
      const note = r.skipped && r.skipped.length ? ` (${fmtN(r.skipped.length)} face${r.skipped.length === 1 ? "" : "s"} left as they were)` : "";
      toast(`Saved ${r.path}${note}`, "ok");
      libLog(`Saved ${r.path}`, "ok");
    } catch (e) { toast(e.message, "err"); }
  }

  async function makeBestShot(burstId, imageId) {
    try {
      await post(`/api/bursts/${num(burstId)}/best-shot`, { image_id: num(imageId) });
      toast("Best shot updated", "ok");
      await renderBursts(burstId);
    } catch (e) { toast(e.message, "err"); }
  }

  // ------------------------------------------------------------------ events
  function clusterRow(c, on) {
    const title = c.label || "Event " + num(c.id);
    const sub = `${fmtN(c.image_count)} photos, ${esc((c.start || "").slice(0, 10))} to ${esc((c.end || "").slice(0, 10))}`;
    const thumb = c.cover_id != null
      ? `<img class="rowthumb" loading="lazy" src="${thumbUrl(c.cover_id)}" alt="">` : "";
    return `<div class="burstrow${on ? " on" : ""}" role="button" tabindex="0" data-cid="${num(c.id)}">
      <div class="rowmain">${thumb}<div style="min-width:0"><div class="name">${esc(title)}</div>
        <div class="info">${sub}</div></div></div>
      <button class="mini" data-rename="${num(c.id)}">Rename</button>
    </div>`;
  }

  async function loadEvents() {
    const listEl = $("#eventlist");
    const data = await api("/api/events");
    $("#eventcount").textContent = fmtN(data.count) + " events";
    if (!data.items.length) {
      listEl.innerHTML = "";
      grids.event.vg.showEmpty(emptyNode({ icon: "cal", title: "No events yet",
        text: "Choose the gap and distance above, then find events. Photos need dates to be grouped." }));
      return;
    }
    const selected = data.items.some((c) => c.id === state.eventId) ? state.eventId : data.items[0].id;
    state.eventId = selected;
    listEl.innerHTML = data.items.map((c) => clusterRow(c, c.id === selected)).join("");
    grids.event.reset();
    $$(".burstrow", listEl).forEach((row) => {
      const pick = (e) => {
        if (e.target.closest("[data-rename]")) return;
        $$(".burstrow", listEl).forEach((r) => r.classList.remove("on"));
        row.classList.add("on");
        state.eventId = num(row.dataset.cid);
        grids.event.reset();
      };
      row.onclick = pick;
      row.onkeydown = (e) => { if ((e.key === "Enter" || e.key === " ") && e.target === row) { e.preventDefault(); pick(e); } };
    });
    $$("[data-rename]", listEl).forEach((b) => {
      b.onclick = async () => {
        const row = b.closest(".burstrow");
        const curName = (row ? ($(".name", row) || {}).textContent : "") || "";
        const name = await showConfirmDialog({
          title: "Rename event",
          confirmText: "Rename",
          cancelText: "Cancel",
          input: {
            label: "Event name",
            value: curName,
            placeholder: "e.g. Summer Holiday",
          },
        });
        if (name == null) return;
        const trimmed = name.trim();
        if (!trimmed || trimmed === curName) return;
        try {
          await post(`/api/events/${num(b.dataset.rename)}/rename`, { name: trimmed });
          await loadEvents();
        } catch (e) { toast(e.message, "err"); }
      };
    });
  }

  // ------------------------------------------------------------- transactions
  async function loadTransactions() {
    const data = await api("/api/transactions");
    const sel = $("#txnselect");
    const cur = sel.value;
    sel.innerHTML = `<option value="">Latest${data.latest_undoable ? " (" + esc(data.latest_undoable) + ")" : ""}</option>` +
      data.items.map((t) =>
        `<option value="${esc(t.transaction_id)}">${esc(t.transaction_id)}: ${fmtN(t.committed)} applied, ${fmtN(t.rolled_back)} undone</option>`).join("");
    sel.value = cur;
  }

  async function followJob(jobId, label) {
    state.taskRunning = true;
    let since = 0;
    try {
      for (;;) {
        const j = await api(`/api/jobs/${encodeURIComponent(jobId)}?since=${since}`);
        j.log.forEach((l) => logLine(l.msg, l.level));
        since = j.next;
        if (j.status !== "running" && j.status !== "stopping") {
          if (j.status === "error") throw new Error(j.error || label + " failed");
          return j.result;
        }
        await new Promise((r) => setTimeout(r, 500));
      }
    } finally {
      state.taskRunning = false;
    }
  }

  // ------------------------------------------------------------------ plan UI
  function planBody() {
    return {
      operation: $("#op").value, mode: $("#orgmode").value,
      output_dir: $("#orgroot").value.trim(),
      only_category: $("#onlycat").value || null, trash: $("#trash").checked,
    };
  }

  const OPVERB = { move: "Move", copy: "Copy", hardlink: "Hard link", symlink: "Link" };

  function renderDiff(d, confirmable) {
    const el = $("#diff-panel");
    const cats = Object.entries(d.by_category || {}).sort((a, b) => b[1] - a[1]);
    const maxN = Math.max(1, ...cats.map((c) => c[1]));
    const notices = [];
    if (!d.enough_space) notices.push(["err", "There is not enough free space on the destination drive."]);
    if (d.missing_sources) notices.push(["err", `${fmtN(d.missing_sources)} source files are missing and will fail.`]);
    if (d.marked_for_delete) notices.push(["", `${fmtN(d.marked_for_delete)} photo${d.marked_for_delete === 1 ? " is" : "s are"} marked for delete and will be left out of the result${$("#trash").checked && d.operation === "move" ? ", moved to the trash folder" : ", staying where they are"}.`]);
    if (d.collisions) notices.push(["", `${fmtN(d.collisions)} files share a name at the destination. They will get a short suffix.`]);
    if (d.cross_device) notices.push(["", `${fmtN(d.cross_device)} files are on another drive. They are copied, then removed from the source.`]);
    const verb = OPVERB[d.operation] || "Apply";
    el.innerHTML = `<h3>${verb} ${fmtN(d.total)} photo${d.total === 1 ? "" : "s"}</h3>
      <p class="lede">Nothing has changed yet. This is what will happen if you continue.</p>
      <div class="diff-target">Destination: <b>${esc(d.organize_root)}</b></div>
      <div class="tiles">
        <div class="tile"><b>${fmtN(d.total)}</b><span>files</span></div>
        <div class="tile"><b>${fmtBytes(d.bytes_total)}</b><span>total size</span></div>
        <div class="tile"><b>${fmtBytes(d.bytes_to_write)}</b><span>new space needed</span></div>
        <div class="tile${d.enough_space ? "" : " bad"}"><b>${fmtBytes(d.free_bytes)}</b><span>free at destination</span></div>
        <div class="tile${d.collisions ? " warn" : ""}"><b>${fmtN(d.collisions)}</b><span>name clashes</span></div>
        <div class="tile${d.missing_sources ? " bad" : ""}"><b>${fmtN(d.missing_sources)}</b><span>missing sources</span></div>
      </div>
      <div class="bars">${cats.map(([c, n]) =>
        `<div class="bar" style="${catStyle(c)}"><span class="name">${esc(catLabel(c))}</span>
           <span class="track"><span class="fill" style="width:${Math.max(2, (n / maxN) * 100)}%"></span></span>
           <span class="n">${fmtN(n)}</span></div>`).join("")}</div>
      ${notices.map(([k, w]) => `<div class="notice ${k}">${icon("alert", "sm")}<span>${esc(w)}</span></div>`).join("")}
      ${(d.collision_samples || []).length ? `<p class="help">Example clashes: ${esc(d.collision_samples.slice(0, 3).join(", "))}</p>` : ""}
      ${confirmable ? `<div class="btn-row" style="margin-top:14px"><button id="diff-confirm" class="btn commit" ${d.enough_space ? "" : "disabled"}>${verb} ${fmtN(d.total)} photos</button>
        <button id="diff-cancel" class="btn">Cancel</button></div>`
        : `<div class="btn-row" style="margin-top:14px"><button id="diff-cancel" class="btn">Close</button></div>`}`;
    el.hidden = false;
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    $("#diff-cancel").onclick = () => { el.hidden = true; };
    if (confirmable) $("#diff-confirm").onclick = () => { el.hidden = true; startApply(); };
  }

  async function startApply() {
    try {
      const r = await post("/api/plan/apply", planBody());
      logLine("Apply started", "ok");
      const res = await followJob(r.job_id, "Apply");
      if (res && res.ok) {
        logLine(`Applied ${fmtN(res.executed)} files (change ${res.transaction_id})`, "ok");
        if (res.marked_left_out) logLine(`${fmtN(res.marked_left_out)} photos marked for delete were left out and stay where they are (turn on the trash option to move them)`, "warn");
        if (res.unapproved_junk) logLine(`${fmtN(res.unapproved_junk)} junk photos were not approved for deletion and stay in the junk review folder`, "warn");
        toast(`Organized ${fmtN(res.executed)} photos. You can undo this from this page.`, "ok");
        if (res.backup_warning) toast(res.backup_warning, "err");
        if (num(res.copy_fallbacks) > 0) {
          addBanner("copy", "Cross-filesystem link detected; files were copied instead of hardlinked.", [["Dismiss", (ev) => ev.currentTarget.parentNode.remove()]]);
        }
        if (num(res.skipped_collisions) > 0) {
          addBanner("skip", `${fmtN(res.skipped_collisions)} files were left in place because their names were already taken.`, [["Dismiss", (ev) => ev.currentTarget.parentNode.remove()]]);
        }
      } else if (res && res.error) {
        logLine(res.error, "err");
        toast(res.error, "err");
      } else if (res) {
        logLine(`Applied ${fmtN(res.executed)} files, ${fmtN(res.failures)} failed`, "err");
        toast(`Finished with ${fmtN(res.failures)} failures`, "err");
      }
    } catch (e) { logLine(e.message, "err"); toast(e.message, "err"); }
    await loadTransactions();
    resetVisibleGrids();
    loadStats();
  }

  async function startUndo() {
    const tid = $("#txnselect").value || null;
    const okUndo = await showConfirmDialog({
      title: "Undo this change?", message: "Files go back to where they were before.",
      confirmText: "Undo", variant: "warning" });
    if (!okUndo) return;
    try {
      const r = await post("/api/undo", { transaction_id: tid, verify: $("#undo-verify").checked });
      if (!r.job_id) { toast(r.error || "Nothing to undo", "err"); return; }
      logLine(`Undo started (change ${r.transaction_id})`, "ok");
      const res = await followJob(r.job_id, "Undo");
      if (res && res.ok) toast(`Restored ${fmtN(res.reversed)} files`, "ok");
      else toast("Some files could not be restored. See the activity log.", "err");
    } catch (e) { logLine(e.message, "err"); toast(e.message, "err"); }
    await loadTransactions();
    resetVisibleGrids();
    loadStats();
  }

  // ------------------------------------------------------------------ progress
  let lastActive = false;

  function renderProgress(p) {
    const card = $("#progress-card");
    const show = (el, on) => { if (el) el.hidden = !on; };
    const done = !p.active && !p.error && !p.stopped && p.percent === 100;
    const bad = !p.active && !!p.error;
    const stopped = !p.active && !!p.stopped;
    state.lastRun = p.run != null ? p.run : null;
    if (p.active) state.dismissedRun = null;
    const dismissed = !p.active && p.run != null && p.run === state.dismissedRun;
    const visible = (p.active || done || bad || stopped) && !dismissed;
    show(card, visible);
    // C8: the equalizer spectrum (M5 CSS) animates only while #progress-card
    // is .active, so gate the eq-pulse keyframes on the live running state.
    card.classList.toggle("active", p.active);
    document.body.classList.toggle("task-running", !!p.active);
    if (!visible) return;
    const pct = p.active ? Math.max(num(p.percent), 4) : 100;
    const cls = p.active ? "active" : (bad || stopped ? "error" : "done");
    show($("#progress-spinner"), p.active);
    [$("#progress-status-icon")].forEach((e) => {
      if (!e) return;
      e.hidden = p.active;
      e.innerHTML = icon(bad || stopped ? "alert" : "check", "sm");
      e.style.color = bad || stopped ? "var(--err)" : "var(--ok)";
      e.style.display = p.active ? "none" : "inline-flex";
    });
    $("#progress-title").textContent = p.active
      ? `${p.task || "Working"}${p.step ? ": " + p.step : ""}`
      : `${p.task || "Task"} ${bad ? "failed" : (stopped ? "stopped" : "finished")}`;
    $("#progress-pct").textContent = bad ? "Error" : (stopped ? "Stopped" : `${num(p.percent)}%`);
    $("#progress-fill").className = "progress-fill " + cls;
    $("#progress-fill").style.width = pct + "%";
    $("#progress-msg").textContent = bad ? p.error : (p.message || "Working");
    $("#progress-counts").textContent = p.detail || (p.total > 0 ? `${fmtN(p.done)} of ${fmtN(p.total)}` : "");
    // C8: live tabular ratio (e.g. "135 / 3,194"); empty when there's no total,
    // which the :empty CSS rule uses to hide the element.
    $("#progress-ratio").textContent = p.detail || (p.total > 0 ? `${fmtN(p.done)} / ${fmtN(p.total)}` : "");
    const phase = (String(p.message || "").match(/Phase\s+\d+\s*\/\s*\d+/i) || [""])[0];
    const phaseEl = $("#progress-phase");
    if (phaseEl) { phaseEl.textContent = phase; phaseEl.hidden = !phase || !p.active; }
    const track = $("#progress-card .progress-track");
    if (track) track.setAttribute("aria-valuenow", String(Math.round(p.active ? num(p.percent) : 100)));

    const isBusy = p.active || p.busy;
    [$("#btn-stop-progress")].forEach((btn) => {
      if (!btn) return;
      btn.style.display = isBusy ? "inline-flex" : "none";
      if (p.stopping) {
        btn.disabled = true;
        btn.textContent = "Stopping...";
      } else {
        btn.disabled = false;
        btn.textContent = "Stop";
      }
    });

    const dis = $("#btn-dismiss-progress");
    if (dis) dis.style.display = isBusy ? "none" : "inline-block";
  }

  function applyBusy(busy) {
    state.busy = busy;
    $$("[data-busy]").forEach((b) => {
      b.disabled = busy;
      b.classList.toggle("busy", busy);
      b.classList.toggle("loading", busy && b === state.busyBtn);
    });
    if (!busy) state.busyBtn = null;
    updateLearn();
  }

  let pollTimer = null;
  let pollSeq = 0;
  async function pollProgress() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
    const seq = ++pollSeq;
    let delay = 3000;
    try {
      const p = await api("/api/progress");
      if (seq !== pollSeq) return;
      renderProgress(p);
      applyBusy(state.taskRunning || !!p.busy || !!p.active);
      if (p.active || p.busy) delay = 600;
      if (lastActive && !p.active) {
        lastActive = false;
        refreshBackend();
        loadStats();
        loadTransactions();
        resetVisibleGrids();
      }
      if (p.active) lastActive = true;
    } catch (e) { /* server restarting; retry */ }
    if (seq === pollSeq) {
      pollTimer = setTimeout(pollProgress, delay);
    }
  }

  // ------------------------------------------------------------ backend banner
  // Keep only the install command from a hint that may include a sentence.
  function hintCommand(h) {
    const t = String(h || "");
    const m = t.match(/((?:python3? -m )?pip3? install[^\n]*)$/);
    return (m ? m[1] : t).trim();
  }

  // The command to show and copy: unset the switch, or install the model.
  function bannerCommand(b) {
    return b && b.forced_by_env ? "unset PCUR_BACKEND" : hintCommand(b && b.hint);
  }

  function renderBanner() {
    const b = state.backend;
    let html = "";
    if (b && b.semantic_is_stub && b.forced_by_env) {
      html = `<div class="banner warn" role="alert">${icon("alert")}<div class="b-body">
        <h3>The AI model is switched off</h3>
        <p>The model is installed, but the environment setting PCUR_BACKEND is forcing the placeholder, which cannot recognise photos. Remove the setting in the terminal that started the app, restart it, then analyze again.</p>
        <div class="b-row"><div class="codebox"><code>${esc(bannerCommand(b))}</code>
          <button class="btn sm" type="button" data-copy>Copy</button></div></div></div></div>`;
    } else if (b && b.semantic_is_stub) {
      html = `<div class="banner warn" role="alert">${icon("alert")}<div class="b-body">
        <h3>Photos cannot be recognised yet</h3>
        <p>No AI model is installed, so a placeholder is sorting your photos. It cannot tell a screenshot from a landscape, which is why most photos land in Finalize. Install the model, restart the app, then analyze again.</p>
        <div class="b-row"><div class="codebox"><code>${esc(hintCommand(b.hint))}</code>
          <button class="btn sm" type="button" data-copy>Copy</button></div>
          <a class="link" href="#settings" data-goto="settings">Device and worker settings</a></div></div></div>`;
    } else if (b && num(b.stub_rows) > 0) {
      html = `<div class="banner info" role="status">${icon("alert")}<div class="b-body">
        <h3>${fmtN(b.stub_rows)} photo${num(b.stub_rows) === 1 ? " was" : "s were"} analysed by the placeholder</h3>
        <p>Analyze again to improve them with the real model.</p>
        <div class="b-row"><button class="btn primary sm" type="button" data-reanalyze>Analyze again</button></div></div></div>`;
    }
    $$(".banner-host").forEach((h) => { h.innerHTML = html; });
  }

  async function refreshBackend() {
    try {
      const cfg = await api("/api/config");
      state.backend = cfg.backend || null;
      state.thresholds = cfg.thresholds || state.thresholds;
      renderBanner();
    } catch (e) { /* ignore */ }
    refreshSettings(false);
  }

  async function copyHint(btn) {
    await copyText(bannerCommand(state.backend), btn);
  }

  async function copyText(text, btn) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } catch (e2) { /* ignore */ }
      ta.remove();
    }
    btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = "Copy"; }, 1800);
  }

  // ------------------------------------------------------------ settings
  const accShort = (a) => String((a && a.label) || (a && a.id) || "CPU").replace(/\s*\(.*\)\s*$/, "");
  const KIND_TEXT = { discrete: "Dedicated GPU", integrated: "Integrated", cpu: "Processor" };

  function accelChoices() {
    const st = state.settings;
    const list = (st && st.accelerators) || [];
    const integratedOk = list.some((a) => a.kind === "integrated" && a.available);
    const ranked = list.map((a, i) => [a, i]).sort((x, y) =>
      (x[0].available ? 0 : 1) - (y[0].available ? 0 : 1) || x[1] - y[1]).map((x) => x[0]);
    return [
      { id: "auto", label: "Automatic", available: true, kind: "",
        sub: "Use the fastest device that works on this computer." },
      { id: "integrated", label: "Integrated graphics", available: integratedOk, kind: "integrated",
        sub: "Use the graphics built into the processor, such as Intel Iris Xe or Apple silicon.",
        reason: "No integrated graphics were found on this computer." },
    ].concat(ranked);
  }

  function renderChip() {
    const st = state.settings;
    const chip = $("#device-chip");
    if (!chip) return;
    if (!st) { $("#device-chip-text").textContent = "Checking device"; return; }
    const acc = (st.accelerators || []).find((a) => a.id === st.selected);
    const label = acc ? accShort(acc) : String(st.selected || "cpu").toUpperCase();
    const isGpu = !!st.selected && st.selected !== "cpu";
    $("#device-chip-text").textContent = isGpu ? "GPU" : "CPU";
    chip.classList.toggle("gpu", isGpu);
    chip.classList.toggle("on", !st.fallback_reason);
    chip.classList.toggle("warn", !!st.fallback_reason);
    const detail = st.fallback_reason ? `Using ${label}. ${st.fallback_reason}` : `Running on ${label}. Open Settings`;
    chip.title = detail;
    chip.setAttribute("aria-label", detail);
  }

  function fillDeviceSelect() {
    const sel = $("#infer-device");
    const st = state.settings;
    if (!sel || !st) return;
    const keep = sel.dataset.user ? sel.value : null;
    sel.innerHTML = accelChoices().map((a) =>
      `<option value="${esc(a.id)}"${a.available ? "" : " disabled"}>${esc(a.id === "auto" ? "Automatic" : accShort(a))}${a.available ? "" : " (not available)"}</option>`).join("");
    const want = keep || (st.settings && st.settings.device) || "auto";
    const ok = $$("option", sel).some((o) => o.value === want && !o.disabled);
    sel.value = ok ? want : "auto";
  }

  async function refreshSettings(render) {
    try { state.settings = await api("/api/settings"); } catch (e) { return; }
    renderChip();
    fillDeviceSelect();
    if (render) renderSettings();
  }

  function setupGpuIndex() {
    const st = state.settings;
    const field = $("#gpu-index-field");
    if (!st) { field.hidden = true; return; }
    const id = state.draft.device === "auto" ? st.selected : state.draft.device;
    const acc = (st.accelerators || []).find((a) => a.id === id);
    let names = null;
    if (acc && Array.isArray(acc.devices) && acc.devices.length > 1) names = acc.devices;
    else if (acc && num(acc.count) > 1) names = Array.from({ length: num(acc.count) }, (_, i) => "GPU " + i);
    field.hidden = !names;
    if (!names) return;
    $("#gpu-index").innerHTML = names.map((n, i) => `<option value="${i}">${i}: ${esc(n)}</option>`).join("");
    $("#gpu-index").value = String(Math.min(num(state.draft.device_index), names.length - 1));
  }

  function renderSettings() {
    const st = state.settings;
    if (!st) return;
    const saved = st.settings || {};
    state.draft = { device: saved.device || "auto", device_index: num(saved.device_index) };
    const accs = st.accelerators || [];
    const active = accs.find((a) => a.id === st.selected);
    const activeLabel = active ? accShort(active) : String(st.selected || "CPU").toUpperCase();
    const reqAcc = accs.find((a) => a.id === st.requested);
    const reqLabel = st.requested === "auto" ? "Automatic" : st.requested === "integrated"
      ? "Integrated graphics" : accShort(reqAcc || { id: st.requested });
    $("#active-device").innerHTML =
      `<div class="now"><i class="dot ${st.selected === "cpu" ? "cpu" : ""}"></i>
         <span>Running on <b>${esc(activeLabel)}</b>${active && active.name ? ", " + esc(active.name) : ""}${active && active.memory_gb ? ", " + esc(active.memory_gb) + " GB" : ""}</span></div>` +
      (st.fallback_reason
        ? `<div class="notice">${icon("alert", "sm")}<span>${esc(reqLabel)} could not be used: ${esc(st.fallback_reason)}. ${esc(activeLabel)} is being used instead.</span></div>` : "");

    $("#accel-list").innerHTML = accelChoices().map((a) => {
      const off = !a.available;
      const hint = String(a.install_hint || "");
      const cmd = /pip3? install/.test(hint) ? hintCommand(hint) : "";
      const sub = a.sub || [a.name, a.memory_gb ? a.memory_gb + " GB" : ""].filter(Boolean).join(", ");
      return `<label class="accel${off ? " off" : ""}" data-acc="${esc(a.id)}">
        <input type="radio" name="device" value="${esc(a.id)}"${off ? " disabled" : ""}${state.draft.device === a.id ? " checked" : ""}>
        <span class="radio"></span>
        <span class="title">${esc(a.id === "auto" || a.id === "integrated" ? a.label : a.label || a.id)}</span>
        <span class="tags">${a.id === st.selected ? '<span class="tag-pill active">In use</span>' : ""}${KIND_TEXT[a.kind] && a.id !== "auto" ? `<span class="tag-pill kind">${esc(KIND_TEXT[a.kind])}</span>` : ""}</span>
        ${sub ? `<span class="sub">${esc(sub)}</span>` : ""}
        ${off ? `<span class="why-off">Not available: ${esc(a.reason || "not detected")}.</span>` : ""}
        ${off && cmd ? `<div class="codebox"><code>${esc(cmd)}</code><button class="btn sm" type="button" data-copytext="${esc(cmd)}">Copy</button></div>`
          : off && hint && !cmd ? `<span class="sub">${esc(hint)}</span>` : ""}
      </label>`;
    }).join("");
    $$("#accel-list input[type=radio]").forEach((r) => {
      r.onchange = () => { state.draft.device = r.value; setupGpuIndex(); };
    });
    setupGpuIndex();
    $("#gpu-index").onchange = (e) => { state.draft.device_index = num(e.target.value); };
    state.draft.device_index = num(saved.device_index);

    const eff = st.effective || {};
    [["scan", "scan_workers"], ["decode", "decode_workers"], ["threads", "torch_threads"]].forEach(([k, key]) => {
      const input = $("#set-" + k);
      input.value = num(saved[key]) > 0 ? String(saved[key]) : "";
      input.classList.remove("input-error");
      $("#err-" + k).hidden = true;
      const show = () => {
        const v = input.value.trim();
        if (v !== "" && (!/^\d+$/.test(v) || Number(v) > 128)) { $("#eff-" + k).textContent = "Check this number"; return; }
        $("#eff-" + k).textContent = v === "" || v === "0"
          ? (num(saved[key]) > 0 || v !== "" ? "Auto" : `Auto uses ${fmtN(eff[key])}`)
          : `Using ${v}`;
      };
      $("#eff-" + k).textContent = num(saved[key]) > 0 ? `Using ${fmtN(eff[key])}` : `Auto uses ${fmtN(eff[key])}`;
      input.oninput = () => { input.classList.remove("input-error"); $("#err-" + k).hidden = true; show(); };
    });
    $("#set-collision").value = saved.collision_policy || "sequence";
    $("#settings-msg").textContent = "";
    $("#settings-msg").className = "help";
  }

  // Blank means auto (0); anything else must be a whole number from 0 to 128.
  function readWorkers() {
    const out = {};
    let bad = false;
    [["scan", "scan_workers", "Scan workers"], ["decode", "decode_workers", "Analysis decode workers"],
     ["threads", "torch_threads", "CPU threads"]].forEach(([k, key, label]) => {
      const input = $("#set-" + k);
      const v = input.value.trim();
      const err = $("#err-" + k);
      err.hidden = true;
      input.classList.remove("input-error");
      if (v === "") { out[key] = 0; return; }
      const n = Number(v);
      if (!/^\d+$/.test(v) || n > 128) {
        err.textContent = `${label} must be a whole number from 0 to 128, or empty for Auto.`;
        err.hidden = false;
        input.classList.add("input-error");
        bad = true;
        return;
      }
      out[key] = n;
    });
    return bad ? null : out;
  }

  async function saveSettings() {
    const msg = $("#settings-msg");
    const workers = readWorkers();
    if (!workers) {
      msg.textContent = "Fix the highlighted fields first.";
      msg.className = "help err";
      return;
    }
    const body = Object.assign({ device: state.draft.device, device_index: num(state.draft.device_index),
      collision_policy: $("#set-collision").value }, workers);
    try {
      state.settings = await post("/api/settings", body);
      renderChip();
      fillDeviceSelect();
      renderSettings();
      $("#settings-msg").textContent = "Saved. The next scan or analysis uses these settings.";
      $("#settings-msg").className = "help ok";
      toast("Settings saved", "ok");
    } catch (e) {
      msg.textContent = e.message;
      msg.className = "help err";
      toast(e.message, "err");
    }
  }

  // ------------------------------------------------------------ learn
  function updateLearn() {
    const btn = $("#btn-learn");
    if (!btn) return;
    const few = state.manualCount < 2;
    btn.disabled = state.busy || few;
    btn.classList.toggle("busy", state.busy);
    const help = $("#learn-help");
    if (help) {
      help.classList.toggle("warn", few);
      help.textContent = few
        ? `Choose a category for at least 2 photos first. You have chosen ${state.manualCount}.`
        : "Fix a few photos, then let the app re-sort the rest using what you chose.";
    }
  }

  async function learnNow() {
    try {
      const r = await post("/api/learn");
      logLine("Learning from your choices", "ok");
      const res = await followJob(r.job_id, "Learn");
      const moved = num(res && res.moved_out_of_uncertain);
      toast(moved
        ? `${fmtN(moved)} photo${moved === 1 ? "" : "s"} moved out of Finalize`
        : "No photos changed. Choose a few more examples and try again.", moved ? "ok" : "err");
      logLine(`Learned from ${fmtN(res && res.examples)} examples, updated ${fmtN(res && res.updated)} photos`, "ok");
    } catch (e) { toast(e.message, "err"); logLine(e.message, "err"); }
    await sortNow(true);
  }

  // --------------------------------------------------- overview and counts
  function renderComp(ov) {
    const total = ov.total;
    const cats = Object.entries(ov.by_category).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]);
    const active = ($("#catfilter") || {}).value || "";
    const dockTotal = $("#dock-total");
    if (dockTotal) {
      dockTotal.textContent = fmtN(total);
      $("#dock-label").textContent = total === 1 ? "photo in your library" : "photos in your library";
    }
    $$(".comp-host").forEach((host) => {
      const head = `<div class="comp-head total"><b class="comp-total" id="total">${fmtN(total)}</b>
        <span class="comp-label">${total === 1 ? "photo" : "photos"} in your library${total ? "" : ". Import a folder to see it sorted here."}</span></div>`;
      if (!total) {
        host.innerHTML = head + '<div class="comp-strip bar empty-strip"></div>';
        return;
      }
      const strip = cats.map(([c, n]) =>
        `<button style="${catStyle(c)};flex:${n} 1 0" data-cat="${esc(c === "unrouted" ? "" : c)}" title="${esc(catLabel(c))}: ${fmtN(n)}" aria-label="${esc(catLabel(c))}: ${fmtN(n)} photos"></button>`).join("");
      const legend = cats.map(([c, n]) => {
        const catVal = c === "unrouted" ? "" : c;
        const isOn = active === catVal;
        return `<button class="chipbtn${isOn ? " on" : ""}" style="${catStyle(c)}" data-cat="${esc(catVal)}"><span class="dot"></span><span>${esc(catLabel(c))}</span> <b>${fmtN(n)}</b></button>`;
      }).join("");
      const unsorted = cats.length === 1 && cats[0][0] === "unrouted";
      host.innerHTML = head + `<div class="comp-strip bar" id="bar" aria-hidden="true">${strip}</div>
        <div class="comp-legend legend" id="legend">${active ? '<button class="chipbtn" data-cat="">Show all</button>' : ""}${legend}</div>` +
        (unsorted ? `<p class="help" style="margin-top:12px">Photos are placed in categories when a plan is built. <button class="btn sm secondary" data-sortnow>Sort photos now</button></p>` : "");
    });
  }

  // Which import step is next: Scan, then Analyze (which also groups and sorts), then Scan again.
  function flowStates(total, pending, bursts, stored) {
    const inferred = total > 0 && pending === 0;
    const stale = stored === null ? bursts === 0 : String(total) !== stored;
    const next = !total ? "scan" : pending > 0 || stale ? "infer" : "scan";
    const cls = (id, rest) => (next === id ? "active" : rest);
    return { next, inferred, scan: cls("scan", "done"), infer: cls("infer", inferred ? "done" : "locked") };
  }

  // One plain sentence for what to do next, shown in the dock.
  function nextHint(flow, total) {
    if (!total) return "Start here: choose a folder, then press Scan.";
    if (flow.next === "infer") return "Next: press Analyze Now. It reads your photos, groups burst shots and sorts them into categories.";
    return "All caught up. Choose a folder and press Scan to add more photos.";
  }

  function markGrouped() {
    if (!state.overview) return;
    try { localStorage.setItem("pcur-grouped-total", String(state.overview.total)); } catch (e) { /* storage unavailable */ }
  }

  function renderCounts(ov) {
    const s = ov.stats || {};
    const set = (k, v, alert) => {
      const e = $(`[data-count="${k}"]`);
      if (!e) return;
      const next = v ? fmtN(v) : "";
      const had = e.textContent;
      e.textContent = next;
      e.classList.toggle("alert", !!(alert && v));
      if (had && had !== next) {
        e.classList.remove("tick");
        void e.offsetWidth;
        e.classList.add("tick");
        afterAnim(e, moMs("--mo-slow", 380) * 2).then(() => e.classList.remove("tick"));
      }
    };
    const junk = Object.entries(ov.by_category).reduce((a, [c, n]) => a + (state.junk.has(c) ? n : 0), 0);
    set("library", ov.total);
    set("categories", ov.total);
    set("bursts", s.burst_groups);
    set("people", s.people);
    set("uncertain", ov.by_category.uncertain || 0, true);
    set("junk", junk);
    set("utility", Object.entries(ov.by_category).reduce((a, [c, n]) => a + (state.utility.has(c) ? n : 0), 0));
    set("actions", s.planned_actions);

    const total = ov.total, pending = num(s.pending_inference), bursts = num(s.burst_groups);
    let stored = null;
    try {
      stored = localStorage.getItem("pcur-grouped-total");
      if (!total) localStorage.removeItem("pcur-grouped-total");
    } catch (e) { /* storage unavailable */ }
    const flow = flowStates(total, pending, bursts, stored);
    state.inferred = flow.inferred;
    const step = (id, cls, text) => {
      const li = $("#step-" + id);
      if (!li) return;
      li.classList.toggle("done", cls === "done");
      li.classList.toggle("active", cls === "active");
      li.classList.toggle("locked", cls === "locked");
      $("#state-" + id).textContent = text;
      const pipeLabel = $("#pipe-label-" + id);
      if (pipeLabel) pipeLabel.textContent = text;
    };
    step("scan", flow.scan, total > 0 ? `${fmtN(total)} indexed` : "Not started");
    step("infer", flow.infer, pending > 0 ? `${fmtN(pending)} waiting`
      : flow.next === "infer" ? "Needs grouping" : total > 0 ? "Done" : "Needs step 1");

    // Per-card hint text (beside each button)
    const scanHint = $("#scan-hint");
    if (scanHint) scanHint.textContent = total > 0 ? "Already-indexed photos are skipped." : "Choose a folder, then press Scan.";
    const inferHint = $("#infer-hint");
    if (inferHint) {
      inferHint.textContent = pending > 0
        ? `${fmtN(pending)} photo${pending === 1 ? "" : "s"} ready to analyze.`
        : flow.next === "infer" ? "Ready when you are."
        : total > 0 ? "Nothing waiting to analyze." : "Run Scan first.";
    }

    // Analysis stages pill row
    const INFER_STAGES = ["Faces", "Sharpness", "Exposure", "Labels", "Bursts", "Categories"];
    const stagesEl = $("#infer-stages");
    if (stagesEl) {
      const allDone = flow.infer === "done";
      const running = pending > 0;
      stagesEl.innerHTML = INFER_STAGES.map((s) =>
        `<span class="stage-pill${allDone ? " done" : running ? " cur" : ""}">${allDone ? "✓ " : ""}${s}</span>`
      ).join("");
    }

    const hint = $("#dock-hint");
    if (hint) hint.textContent = nextHint(flow, total);
  }

  async function loadStats() {
    try {
      const ov = await api("/api/overview");
      state.overview = ov;
      state.manualCount = num(ov.manual_count);
      updateLearn();
      renderComp(ov);
      renderCounts(ov);
    } catch (e) { /* ignore */ }
    loadFacets();
    refreshDupCount();
  }

  // ------------------------------------------------------------------- tabs
  function fillCatFilters() {
    const sorted = state.cats.slice().sort((a, b) => catLabel(a).localeCompare(catLabel(b)));
    const opts = sorted.map((c) => `<option value="${esc(c)}">${esc(catLabel(c))}</option>`).join("");
    $("#catfilter").innerHTML = '<option value="">All categories</option>' + opts;
    $("#onlycat").innerHTML = '<option value="">Every category</option>' + opts;
    $("#bulk-cat").innerHTML = '<option value="" disabled selected>Move to category</option>' + opts;
  }

  function showTab(name, keepPicks) {
    const changed = state.tab !== name;
    state.tab = name;
    state.triageGrid = null;
    if (!keepPicks) clearPicks();
    $$(".tab").forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("active", on);
      if (on) t.setAttribute("aria-current", "page"); else t.removeAttribute("aria-current");
    });
    $$(".panel").forEach((p) => {
      p.classList.toggle("active", p.id === "tab-" + name);
      p.classList.remove("enter");
    });
    const entering = $("#tab-" + name);
    if (changed && entering && !reducedMotion()) {
      void entering.offsetWidth;
      entering.classList.add("enter");
      const seq = entering._enterSeq = (entering._enterSeq || 0) + 1;
      afterAnim(entering, moMs("--mo-std", 260) + 40).then(() => { if (entering._enterSeq === seq) entering.classList.remove("enter"); });
    }
    positionNavIndicator();
    const [title, sub] = TITLES[name] || [name, ""];
    $("#page-title").textContent = title;
    $("#page-sub").textContent = sub;
    document.title = title + " - Photocure Studio";
    document.body.classList.remove("nav-open");
    try { history.replaceState(null, "", "#" + name); } catch (e) { /* ignore */ }
    window.scrollTo(0, 0);

    const gridTab = { categories: "cat", junk: "junk", utility: "utility", uncertain: "uncertain", highlights: "highlights" }[name];
    if (gridTab) {
      const g = grids[gridTab];
      if (g) {
        if (g.total == null || g.stale) { g.stale = false; g.reset(); }
        else { g.update(); }
      }
    }
    if (name === "search") {
      if (!$("#searchgrid").children.length) runSearch();
      setTimeout(() => $("#searchq").focus(), 0);
    }
    if (name === "library") refreshWatch();
    if (name === "duplicates") loadDuplicates(false);
    if (name === "highlights") {
      loadFeatures().catch(() => {});
      refreshTaste().catch(() => {});
      loadHighlightYears().catch(() => {});
      if (isEventView()) renderEventHighlights();
    }
    if (name === "people") {
      renderPeople().catch((e) => toast(e.message, "err"));
      updatePeopleSaveBar();
    } else {
      closeOverlay($("#people-savebar"));
    }
    if (name === "actions") loadTransactions().catch(() => {});
    if (name === "settings") { refreshSettings(true); refreshKeywords(); }
    if (name === "uncertain") renderLegend();
    if (state.overview) renderComp(state.overview);
  }

  // ------------------------------------------------------------------- init
  async function runTask(label, fn) {
    state.taskRunning = true;
    const started = document.activeElement;
    state.busyBtn = started && started.matches && started.matches("[data-busy]") ? started : null;
    applyBusy(true);
    libLog(label + "...");
    try {
      const res = await fn();
      if (res && res.stopped) {
        libLog(`${label} stopped by user`, "warn");
        toast(`${label} stopped`, "warn");
        return null;
      }
      return res;
    } catch (e) {
      libLog(`${label} failed: ${e.message}`, "err");
      toast(e.message, "err");
      return null;
    } finally {
      state.taskRunning = false;
      pollProgress();
    }
  }

  async function stopProcess() {
    const btns = [$("#btn-stop-progress")];
    btns.forEach((b) => {
      if (b) {
        b.disabled = true;
        b.textContent = "Stopping...";
      }
    });
    try {
      await post("/api/stop");
      toast("Stop requested", "ok");
    } catch (e) {
      toast("Failed to stop: " + e.message, "err");
      btns.forEach((b) => {
        if (b) {
          b.disabled = false;
          b.textContent = "Stop";
        }
      });
    }
  }

  async function resetLibrary() {
    const okReset = await showConfirmDialog({
      title: "Clear the indexed library?",
      message: "This removes every indexed photo, AI analysis, burst and plan from the Photocure database. Your original photo files are not deleted or modified.",
      confirmText: "Clear library", variant: "danger" });
    if (!okReset) return;
    const btns = [$("#btn-reset-library"), $("#btn-import-reset")];
    const saved = btns.map((b) => b && b.innerHTML);
    btns.forEach((b) => {
      if (b) {
        b.disabled = true;
        b.textContent = "Clearing...";
      }
    });
    try {
      await post("/api/reset");
      toast("Library cleared successfully", "ok");
      libLog("Library cleared: all indexed images and metadata removed.", "warn");
      const pathInput = $("#scan-path");
      if (pathInput) {
        pathInput.value = "";
        if (window.__syncScanPathDisplay) window.__syncScanPathDisplay();
      }
      ["#state-scan", "#state-infer"].forEach((sel) => {
        const el = $(sel);
        if (el) el.textContent = "";
      });
      renderProgress({ active: false, done: 0, total: 0, percent: 0, message: "" });
      await Promise.all([loadStats(), loadTransactions(), refreshBackend()]);
      resetVisibleGrids();
    } catch (e) {
      toast("Failed to reset library: " + e.message, "err");
    } finally {
      btns.forEach((b, i) => {
        if (b) {
          b.disabled = false;
          b.innerHTML = saved[i];
        }
      });
    }
  }

  function needPath() {
    const input = $("#scan-path");
    const path = input ? input.value.trim() : "";
    if (!path) {
      if (input) { input.classList.add("input-error"); }
      const bar = $(".folder-pick-bar");
      if (bar) {
        bar.classList.add("input-error");
        setTimeout(() => bar.classList.remove("input-error"), 2000);
      }
      libLog("Choose a photo folder first", "err");
      toast("Choose a photo folder first", "err");
      $("#btn-browse-scan").focus();
      return null;
    }
    return path;
  }

  // Placing photos in categories happens when a plan is built, so do it after analysis.
  async function sortNow(silent) {
    let ok = true;
    try {
      await post("/api/plan/rebuild", planBody());
      if (!silent) toast("Photos sorted into categories", "ok");
    } catch (e) {
      ok = false;
      if (!silent) toast(e.message, "err");
    }
    await loadStats();
    resetVisibleGrids();
    return ok;
  }

  async function afterTask(sort) {
    let ok = true;
    if (sort) ok = await sortNow(true);
    else { await loadStats(); resetVisibleGrids(); }
    refreshBackend();
    return ok;
  }

  // ------------------------------------------------------- lifecycle
  function showRestartOverlay() {
    const box = $("#restart-overlay");
    $("#restart-spinner").hidden = false;
    $("#restart-text").textContent = "Restarting Photocure...";
    box.hidden = false;
    const t0 = Date.now();
    const timer = setInterval(async () => {
      // The old process can still answer for a moment, so wait before trusting /healthz.
      try {
        const r = await fetch("/healthz", { cache: "no-store" });
        if (r.ok && Date.now() - t0 > 1000) { clearInterval(timer); window.location.reload(); return; }
      } catch (e) { /* server is down; keep polling */ }
      if (Date.now() - t0 > 60000) {
        clearInterval(timer);
        $("#restart-spinner").hidden = true;
        $("#restart-text").textContent = "Server did not restart within 60 seconds. Please run 'photocure start' from your terminal.";
      }
    }, 500);
  }

  async function restartServer() {
    const go = await showConfirmDialog({
      title: "Restart Photocure?", message: "The server restarts and this page reloads.",
      confirmText: "Restart", variant: "warning" });
    if (!go) return;
    try {
      await api("/api/restart?force=false", { method: "POST" });
    } catch (e) {
      const msg = (e.body && e.body.error) || e.message;
      if (e.status !== 409 || !/background task/.test(msg)) { toast(msg, "err"); return; }
      const force = await showConfirmDialog({
        title: "Restart anyway?", message: "A scan or analysis is running and will stop.",
        confirmText: "Restart anyway", variant: "danger" });
      if (!force) return;
      try { await api("/api/restart?force=true", { method: "POST" }); }
      catch (e2) { toast((e2.body && e2.body.error) || e2.message, "err"); return; }
    }
    showRestartOverlay();
  }

  function addBanner(kind, text, actions) {
    const box = $("#banners");
    const el = document.createElement("div");
    el.className = "banner";
    el.dataset.kind = kind;
    const span = document.createElement("span");
    span.textContent = text;
    el.appendChild(span);
    (actions || []).forEach(([label, fn]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn";
      b.textContent = label;
      b.onclick = fn;
      el.appendChild(b);
    });
    box.appendChild(el);
    return el;
  }

  async function checkOrphans() {
    let r;
    try { r = await api("/api/transactions/orphaned"); } catch (e) { return; }
    $$('#banners [data-kind="orphan"]').forEach((n) => n.remove());
    (r.transactions || []).forEach((t) => {
      const url = "/api/transactions/" + encodeURIComponent(t.transaction_id);
      const settle = async (path, label) => {
        try { await post(url + path); toast(label, "ok"); } catch (e) { toast(e.message, "err"); }
        checkOrphans();
      };
      addBanner("orphan",
        `Incomplete transaction from unexpected shutdown detected (${t.transaction_id}). Resume to finish, or Undo to roll back.`,
        [["Resume", () => settle("/resume", "Transaction settled")],
         ["Undo", async () => {
           const yes = await showConfirmDialog({
             title: "Roll back this transaction?", message: "Files moved so far go back to where they were.",
             confirmText: "Roll back", variant: "warning" });
           if (yes) settle("/rollback", "Transaction rolled back");
         }]]);
    });
  }

  // Open a tab and, when the link names an anchor, scroll that section into view.
  function gotoTarget(el) {
    showTab(el.dataset.goto);
    const anchor = el.dataset.anchor && $("#" + el.dataset.anchor);
    if (!anchor) return;
    requestAnimationFrame(() => {
      anchor.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
      anchor.focus({ preventScroll: true });
    });
  }

  async function loadAboutInfo() {
    try {
      const about = await api("/api/about");
      const versionBadge = $("#header-version-badge");
      if (versionBadge) versionBadge.textContent = about.version || "0.0.0";
      const sidebarVersion = $("#sidebar-version");
      if (sidebarVersion) sidebarVersion.textContent = about.version || "0.0.0";

      const aboutCard = $("#about-card");
      if (aboutCard) {
        const chips = (about.models || []).map((m) => `<span class="about-chip">${esc(m.name)}</span>`).join("");
        aboutCard.innerHTML = `
          <div class="about">
            <img class="about-logo" src="/static/logo.svg" alt="" width="64" height="64">
            <div class="about-main">
              <div class="about-title"><span class="about-name">Photocure Studio</span><span class="about-ver">v${esc(about.version || "0.0.0")}</span></div>
              <p class="about-tag">Offline photo curation. Your photos never leave this computer.</p>
              <dl class="about-facts">
                <div><dt>Acceleration</dt><dd>${esc(about.device || "CPU")}</dd></div>
                <div><dt>Python</dt><dd>${esc(about.python || "unknown")}</dd></div>
                <div><dt>License</dt><dd>${esc(about.license || "MIT License")}</dd></div>
                <div><dt>Author</dt><dd>${esc(about.author || "Unknown")}</dd></div>
              </dl>
              ${chips ? `<div class="about-models" aria-label="Active models">${chips}</div>` : ""}
              <div class="about-links">
                <a class="btn sm secondary" href="/licenses" target="_blank" rel="noopener">Licenses</a>
                <a class="btn sm ghost" href="https://github.com/safiyu/photocure" target="_blank" rel="noopener">GitHub</a>
              </div>
            </div>
          </div>`;
      }
    } catch (e) {
      /* ignore */
    }
  }

  async function init() {
    applyTheme(document.documentElement.dataset.theme || "light");
    await exchangeFromHash();
    checkOrphans();
    const btnRestart = $("#btn-restart");
    if (btnRestart) btnRestart.onclick = restartServer;
    try {
      const cfg = await api("/api/config");
      state.cats = sortCats(cfg.categories.map((c) => c.value));
      state.backend = cfg.backend || null;
      state.thresholds = cfg.thresholds || null;
      state.junk = new Set(cfg.categories.filter((c) => c.junk).map((c) => c.value));
      state.utility = new Set(cfg.categories.filter((c) => c.utility).map((c) => c.value));
      if (!$("#orgroot").value) $("#orgroot").value = cfg.default_output_dir;
    } catch (e) {
      state.cats = sortCats(["people", "food", "scenery", "animal", "screenshot", "document",
        "graphic", "accidental", "duplicate", "uncertain", "unknown"]);
      state.junk = new Set(["graphic", "accidental", "duplicate"]);
      state.utility = new Set(["utility", "screenshot", "document", "slide"]);
    }
    await loadAboutInfo();
    fillCatFilters();
    setupGrids();
    bindFilters();
    bindDelegates();
    bindLightbox();
    bindPicker();
    document.addEventListener("keydown", onKey);

    $$(".tab").forEach((t) => (t.onclick = () => showTab(t.dataset.tab)));
    $("#menu-btn").onclick = () => document.body.classList.toggle("nav-open");
    document.addEventListener("click", (e) => {
      if (document.body.classList.contains("nav-open") && !e.target.closest("#side, #menu-btn")) {
        document.body.classList.remove("nav-open");
      }
    });
    const themeToggle = $("#theme-toggle");
    const themeSwitch = $("#theme-switch");
    const toggleTheme = () =>
      applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    if (themeSwitch) themeSwitch.onchange = toggleTheme;
    if (themeToggle) {
      themeToggle.onclick = (e) => {
        if (e.target === themeSwitch) return;  // the native change already fired
        toggleTheme();
      };
      themeToggle.onkeydown = (e) => {
        if (e.target === themeSwitch) return;  // the input handles Space itself
        if (e.key === " " || e.key === "Enter") {
          e.preventDefault();
          toggleTheme();
        }
      };
    }
    $("#btn-help").onclick = openHelp;
    $("#help-close").onclick = closeHelp;
    $("#help").addEventListener("click", (e) => { if (e.target.id === "help") closeHelp(); });

    let saved = null;
    try { saved = localStorage.getItem("pcur-density"); } catch (e) { /* ignore */ }
    const curVal = saved || ($("#density") ? $("#density").value : "220") || "220";
    const setDens = (val) => {
      const numVal = Math.max(220, parseInt(val, 10) || 220);
      state.density = numVal;
      document.documentElement.style.setProperty("--thumb-min", numVal + "px");
      try { localStorage.setItem("pcur-density", String(numVal)); } catch (e) { /* ignore */ }
      $$(".density input:not(#people-density), #density, .density-slider:not(.people-density-slider)").forEach((s) => {
        if (s.value !== String(numVal)) s.value = numVal;
      });

      const newTier = gridThumbTier(numVal);
      const tierChanged = state.gridTier && state.gridTier !== newTier;
      state.gridTier = newTier;

      // Update sizes attribute and upgrade visible card images in real time
      const sizesAttr = `(max-width: 480px) 100vw, ${Math.round(numVal * 1.25)}px`;
      $$(".card .thumb img, .vgrid .thumb img").forEach((img) => {
        img.sizes = sizesAttr;
        const card = img.closest(".card");
        if (card && card.dataset.id) {
          const id = num(card.dataset.id);
          const expected = thumbUrl(id, newTier);
          if (newTier === "medium" && !img.src.includes("tier=medium")) {
            img.src = expected;
          } else if (newTier === "micro" && img.src.includes("tier=medium")) {
            img.src = expected;
          }
        }
      });

      allGrids().forEach((g) => {
        if (g && g.vg) {
          if (tierChanged) g.vg._clearCards();
          g.vg.update();
        }
      });
    };
    document.addEventListener("input", (e) => {
      if (e.target && e.target.matches && e.target.matches(".density input:not(#people-density), #density, .density-slider:not(.people-density-slider)")) {
        setDens(e.target.value);
      }
    });
    setDens(curVal);

    let savedPeopleDensity = null;
    try { savedPeopleDensity = localStorage.getItem("pcur-people-density"); } catch (e) { /* ignore */ }
    const curPeopleVal = savedPeopleDensity || ($("#people-density") ? $("#people-density").value : "210") || "210";
    const setPeopleDens = (val) => {
      const numVal = Math.min(340, Math.max(140, parseInt(val, 10) || 210));
      state.peopleDensity = numVal;
      document.documentElement.style.setProperty("--people-card-min", numVal + "px");
      try { localStorage.setItem("pcur-people-density", String(numVal)); } catch (e) { /* ignore */ }
      const slider = $("#people-density");
      if (slider && slider.value !== String(numVal)) slider.value = numVal;
    };
    document.addEventListener("input", (e) => {
      if (e.target && e.target.matches && e.target.matches("#people-density, .people-density-slider")) {
        setPeopleDens(e.target.value);
      }
    });
    setPeopleDens(curPeopleVal);

    ["catfilter", "manualonly", "neardup"].forEach((id) =>
      $("#" + id).addEventListener("change", () => {
        updateResetFiltersButton();
        grids.cat.reset();
        if (state.overview) renderComp(state.overview);
      }));
    let qt = null;
    $("#q").addEventListener("input", () => {
      clearTimeout(qt);
      updateResetFiltersButton();
      qt = setTimeout(() => grids.cat.reset(), 250);
    });
    $("#junkfilter").addEventListener("change", () => grids.junk.reset());
    if ($("#filter-uncertain-status")) {
      $("#filter-uncertain-status").addEventListener("change", () => grids.uncertain.reset());
    }
    if ($("#q-uncertain")) {
      let qtu = null;
      $("#q-uncertain").addEventListener("input", () => {
        clearTimeout(qtu);
        qtu = setTimeout(() => grids.uncertain.reset(), 250);
      });
    }
    $$("#tab-utility [data-ukind]").forEach((b) => b.addEventListener("click", () => {
      state.utilityKind = b.dataset.ukind;
      $$("#tab-utility [data-ukind]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      grids.utility.reset();
    }));
    $("#btn-rescore").onclick = async () => {
      try {
        const r = await post("/api/rescore", {});
        if (r.note) { toast(r.note, "err"); return; }
        const res = await followJob(r.job_id, "Re-check");
        toast(`Re-checked ${fmtN(res.updated)} photos`, "ok");
        await sortNow(true);
        grids.utility.reset();
      } catch (e) { toast(e.message, "err"); }
    };
    $("#minaes").addEventListener("change", () => {
      if (isEventView()) renderEventHighlights(); else grids.highlights.reset();
    });
    bindHighlights();
    bindDuplicates();
    bindActivity();
    bindKeywords();
    bindWatch();
    $("#search-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
    $("#search-dups").addEventListener("change", () => { if ($("#searchq").value.trim()) runSearch(); });
    $("#similar-close").onclick = () => { closeOverlay($("#similar-panel")); };
    window.addEventListener("resize", positionNavIndicator);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(positionNavIndicator);
    $("#btn-dismiss-progress").onclick = () => { state.dismissedRun = state.lastRun; $("#progress-card").hidden = true; };

    $("#btn-learn").onclick = learnNow;
    renderBanner();
    updateLearn();
    $("#bulk-clear").onclick = clearPicks;
    $("#bulk-merge").onclick = bulkMergePeople;
    $("#bulk-name").onclick = () => {
      const [p] = selectedPeople();
      if (p) $(".people-name-pill-wrapper", p.card).click();
    };
    $("#bulk-del-people").onclick = bulkDeletePeople;
    $("#bulk-lock").onclick = () => bulkLock();
    $("#bulk-delete").onclick = () => bulkMarkDelete();
    $("#bulk-junk").onclick = bulkJunk;
    $("#bulk-best").onclick = async () => {
      const bi = burstPickInfo();
      if (!bi || bi.picked.length !== 1) return;
      const bid = currentBurstId, imageId = num(bi.picked[0].id);
      clearPicks();
      await makeBestShot(bid, imageId);
    };
    $("#bulk-rest").onclick = async () => {
      const bi = burstPickInfo();
      if (!bi || !bi.rest.length) return;
      await markBurstDeletes(currentBurstId, bi.rest.map((m) => num(m.id)), true);
    };
    $("#bulk-clean").onclick = openCleanExport;
    $("#ce-scan").onclick = ceScan;
    $("#ce-go").onclick = ceExport;
    $("#ce-close").onclick = $("#ce-cancel").onclick = () => closeOverlay($("#cleanexport"));
    $("#ce-browse").onclick = () => openPicker({
      title: "Choose where the clean copies go", mode: "output", initial: $("#ce-dest").value.trim(),
      okLabel: "Use this folder", onPick: (p) => { $("#ce-dest").value = p; },
    });
    $("#bulk-cat").onchange = (e) => { if (e.target.value) bulkReassign(e.target.value); e.target.selectedIndex = 0; };

    const scanInput = $("#scan-path");
    const scanDisplay = $("#scan-path-display");
    function syncScanPathDisplay() {
      if (!scanInput || !scanDisplay) return;
      const val = scanInput.value.trim();
      if (val) {
        scanDisplay.textContent = val;
        scanDisplay.classList.add("has-path");
        scanDisplay.title = val;
      } else {
        scanDisplay.textContent = scanDisplay.dataset.empty || "No folder selected";
        scanDisplay.classList.remove("has-path");
        scanDisplay.title = "";
      }
    }
    window.__syncScanPathDisplay = syncScanPathDisplay;
    const valDesc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
    if (valDesc && scanInput) {
      Object.defineProperty(scanInput, "value", {
        get() { return valDesc.get.call(this); },
        set(v) {
          valDesc.set.call(this, v);
          syncScanPathDisplay();
        }
      });
    }
    if (scanInput) {
      scanInput.addEventListener("input", () => {
        scanInput.classList.remove("input-error");
        const bar = $(".folder-pick-bar");
        if (bar) bar.classList.remove("input-error");
        syncScanPathDisplay();
      });
      scanInput.addEventListener("change", syncScanPathDisplay);
      scanInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); $("#btn-scan").click(); }
      });
    }
    const folderPickBar = $(".folder-pick-bar");
    if (folderPickBar) {
      folderPickBar.addEventListener("click", (e) => {
        if (!e.target.closest("#btn-browse-scan")) {
          $("#btn-browse-scan").click();
        }
      });
      folderPickBar.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          $("#btn-browse-scan").click();
        }
      });
    }
    const scanLabel = $('label[for="scan-path"]');
    if (scanLabel) {
      scanLabel.addEventListener("click", (e) => {
        e.preventDefault();
        $("#btn-browse-scan").click();
      });
    }
    syncScanPathDisplay();

    $("#btn-scan").onclick = async () => {
      const path = needPath();
      if (!path) return;
      const r = await runTask(`Scanning ${path}`, () => post("/api/scan", { path }));
      if (r) {
        libLog(`Scan finished: ${fmtN(r.scanned)} files seen, ${fmtN(r.new)} new, ${fmtN(r.duplicates)} duplicates in ${r.duration_s}s`, "ok");
        toast(`Indexed ${fmtN(r.new)} new photos`, "ok");
        await afterTask();
      }
    };
    $("#btn-infer").onclick = async () => {
      const device = $("#infer-device").value;
      const r = await runTask(`Analyzing photos (${device})`, async () => {
        const inferred = await post("/api/infer", { device });
        if (inferred && inferred.stopped) return inferred;
        if (!inferred.new_ids_count) return { inferred, grouped: null };
        const grouped = await post("/api/cluster", { only_new: true });
        if (grouped && grouped.stopped) return grouped;
        return { inferred, grouped };
      });
      if (!r) return;
      if (!r.grouped) {
        libLog("Nothing new to analyze", "ok");
        toast("Nothing new to analyze. Use Reanalyse to re-check photos you already analyzed.", "ok");
        await afterTask();
        return;
      }
      libLog(`Analysis finished: ${fmtN(r.inferred.processed)} photos, ${fmtN(r.inferred.faces_total)} faces in ${r.inferred.duration_s}s`, "ok");
      libLog(`Grouping finished: ${fmtN(r.grouped.burst_groups)} burst groups`, "ok");
      toast(`Analyzed ${fmtN(r.inferred.processed)} new photos`, "ok");
      markGrouped();
      const sorted = await afterTask(true);
      libLog(sorted ? "Photos sorted into categories" : "Could not sort photos into categories", sorted ? "ok" : "warn");
      if (sorted) await approveProposedDeletes(true);
    };
    $("#btn-recheck").onclick = async () => {
      const r = await runTask("Re-checking rules", async () => {
        const job = await post("/api/rescore", {});
        if (job.note) return { note: job.note };
        const res = await followJob(job.job_id, "Re-check");
        await sortNow(true);
        return res;
      });
      if (!r) return;
      if (r.note) { toast(r.note, "err"); return; }
      toast(`Re-checked ${fmtN(r.updated)} photos`, "ok");
      libLog(`Re-checked ${fmtN(r.updated)} photos and re-sorted them`, "ok");
      grids.utility.reset();
    };
    $("#btn-reanalyse-all").onclick = async () => {
      const total = state.overview ? state.overview.total : 0;
      const ok = await showConfirmDialog({
        title: "Reanalyse everything?",
        message: `This runs the AI on all ${fmtN(total)} photos again and can take a long time. Photos you already sorted by hand are kept.`,
        confirmText: "Reanalyse", variant: "warning" });
      if (!ok) return;
      const device = $("#infer-device").value;
      const r = await runTask(`Reanalysing photos (${device})`, async () => {
        const inferred = await post("/api/infer", { device, reanalyze: true });
        if (inferred && inferred.stopped) return inferred;
        const grouped = await post("/api/cluster", {});
        if (grouped && grouped.stopped) return grouped;
        return { inferred, grouped };
      });
      if (!r) return;
      libLog(`Reanalysis finished: ${fmtN(r.inferred.processed)} photos, ${fmtN(r.grouped.burst_groups)} burst groups`, "ok");
      toast(`Reanalysed ${fmtN(r.inferred.processed)} photos`, "ok");
      markGrouped();
      await afterTask(true);
    };
    $("#btn-ingest-all").onclick = async () => {
      const path = needPath();
      if (!path) return;
      const r = await runTask(`Importing ${path}`, () => post("/api/ingest", { path }));
      if (r) {
        libLog(`Import finished: ${fmtN(r.scan.new)} new, ${fmtN(r.infer.processed)} analyzed, ${fmtN(r.cluster.burst_groups)} burst groups`, "ok");
        toast("Import finished", "ok");
        markGrouped();
        if (await afterTask(true)) await approveProposedDeletes();
      }
    };

    $("#btn-events").onclick = async () => {
      try {
        const r = await post("/api/events/cluster", {
          gap_hours: num($("#ev-gap").value) || 2, gps_km: num($("#ev-km").value) || 1,
        });
        toast(`Found ${fmtN(r.events)} events`, "ok");
        await loadEvents();
      } catch (e) { toast(e.message, "err"); }
    };

    $("#btn-rebuild").onclick = async () => {
      try {
        const r = await post("/api/plan/rebuild", planBody());
        const n = Object.keys(r.by_category).length;
        $("#plan-summary").innerHTML =
          `<div class="status-line">${icon("check", "sm")}<span><b>${fmtN(r.total)}</b> photos planned across <b>${n}</b> categories` +
          `${r.burst_count ? `, ${fmtN(r.burst_count)} burst groups` : ""}. Destination: <b>${esc(r.organize_root)}</b></span></div>`;
        logLine(`Plan rebuilt: ${fmtN(r.total)} files`, "ok");
        toast("Plan rebuilt", "ok");
        loadStats();
        resetVisibleGrids();
      } catch (e) { toast(e.message, "err"); }
    };
    $("#btn-diff").onclick = async () => {
      try { renderDiff(await post("/api/plan/diff", planBody()), false); }
      catch (e) { toast(e.message, "err"); }
    };
    $("#btn-apply").onclick = async () => {
      try { renderDiff(await post("/api/plan/diff", planBody()), true); }
      catch (e) { toast(e.message, "err"); }
    };
    $("#btn-undo").onclick = startUndo;
    $("#btn-verify").onclick = async () => {
      const tid = $("#txnselect").value || "latest";
      try {
        const r = await api(`/api/transactions/${encodeURIComponent(tid)}/verify`);
        logLine(`Verified ${r.transaction_id}: ${JSON.stringify(r.report).slice(0, 400)}`, "ok");
        toast("Verification finished. See the activity log.", "ok");
      } catch (e) { logLine(e.message, "err"); toast(e.message, "err"); }
    };
    $("#btn-repair").onclick = async () => {
      try {
        const r = await post("/api/repair", {});
        logLine(`Repaired ${fmtN(r.repaired)} image paths`, "ok");
        toast(`Repaired ${fmtN(r.repaired)} paths`, "ok");
        resetVisibleGrids();
      } catch (e) { toast(e.message, "err"); }
    };

    window.addEventListener("hashchange", () => {
      const h = (location.hash || "").slice(1);
      if (TITLES[h] && h !== state.tab) showTab(h);
    });
    window.addEventListener("beforeunload", (ev) => {
      if (pendingPeopleDeletions.size > 0) {
        ev.preventDefault();
        ev.returnValue = "";
      }
    });
    $("#btn-save-settings").onclick = saveSettings;
    $("#btn-reset-settings").onclick = () => { renderSettings(); };
    $("#btn-stop-progress").onclick = stopProcess;
    const btnResetLib = $("#btn-reset-library");
    if (btnResetLib) btnResetLib.onclick = resetLibrary;
    const btnImportReset = $("#btn-import-reset");
    if (btnImportReset) btnImportReset.onclick = resetLibrary;
    $("#infer-device").addEventListener("change", (e) => { e.target.dataset.user = "1"; });
    await Promise.all([loadStats(), loadTransactions(), refreshSettings(false)]);
    const start = (location.hash || "").slice(1);
    showTab(TITLES[start] ? start : "library");
    pollProgress();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
