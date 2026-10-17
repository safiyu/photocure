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
    tab: "library", cats: [], junk: new Set(), byId: {}, busy: false, taskRunning: false,
    inferred: false,
    triageSel: -1, triageGrid: null, undo: [], picks: new Set(), lastPick: null,
    overview: null, eventId: null,
    backend: null, thresholds: null, manualCount: 0,
    settings: null, draft: { device: "auto", device_index: 0 },
  };

  const TITLES = {
    library: ["Import", "Choose a folder and let Photocure sort what it finds."],
    categories: ["Library", "Every photo, grouped by the category it was sorted into."],
    bursts: ["Bursts", "Shots taken seconds apart. The best one of each group is kept."],
    highlights: ["Highlights", "Your most appealing photos first."],
    events: ["Events", "Photos grouped by when and where they were taken."],
    junk: ["Junk review", "Screenshots, documents, duplicates and accidents. Move anything that does not belong."],
    uncertain: ["Needs a look", "The analysis was not sure about these. Choose a category for each."],
    settings: ["Settings", "Choose what runs the analysis and how many workers it uses."],
    actions: ["Apply and undo", "Review the plan, then organize your files. Every change can be undone."],
  };

  // ------------------------------------------------------------------ utils
  function toast(msg, cls) {
    const box = $("#toasts");
    const el = document.createElement("div");
    el.className = "toast " + (cls || "");
    el.innerHTML = (cls === "err" ? icon("alert", "sm") : cls === "ok" ? icon("check", "sm") : "") +
      `<span>${esc(msg)}</span>`;
    box.appendChild(el);
    while (box.children.length > 4) box.firstChild.remove();
    setTimeout(() => el.remove(), cls === "err" ? 6000 : 3200);
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
    ok.className = "btn " + variant;
    const previous = document.activeElement;
    modal.hidden = false;

    return new Promise((resolve) => {
      let releaseTrap = null;
      const finish = (value) => {
        if (releaseTrap) {
          releaseTrap();
          releaseTrap = null;
        }
        modal.hidden = true;
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
  // Tiers: micro for the grid, medium for burst inspection, preview for the lightbox.
  const thumbUrl = (id, tier) => {
    const it = state.byId[id];
    return "/thumbnail/" + num(id) + "?tier=" + (tier || "micro") + (it && it.fp ? "&v=" + it.fp : "");
  };
  const catKey = (c) => String(c || "unrouted").replace(/[^a-z_]/g, "") || "unknown";
  // Most useful first, so the number keys 1-9 cover the categories people assign most.
  const CAT_ORDER = ["people", "food", "scenery", "animal", "screenshot", "document", "graphic",
    "illustration", "slide", "architecture", "vehicle", "plant", "product", "accidental",
    "uncertain", "duplicate", "near_duplicate", "burst_alternate", "unreadable", "custom", "unknown"];
  const TRIAGE_SKIP = new Set(["unknown", "duplicate", "near_duplicate", "uncertain",
    "burst_alternate", "unreadable", "custom"]);
  const sortCats = (cats) => cats.slice().sort((a, b) => {
    const ia = CAT_ORDER.indexOf(a), ib = CAT_ORDER.indexOf(b);
    return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib);
  });
  const catStyle = (c) => `--cat:var(--c-${catKey(c)}, var(--c-unknown))`;
  const catLabel = (c) => {
    const t = String(c || "Not sorted yet").replace(/_/g, " ");
    return t.charAt(0).toUpperCase() + t.slice(1);
  };
  const fmtBytes = (n) => {
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0, v = num(n);
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return v.toFixed(i ? 1 : 0) + " " + u[i];
  };

  function logTo(boxSel, msg, cls) {
    const box = $(boxSel);
    if (!box) return;
    $$(":scope > .logline:not([data-t])", box).forEach((e) => e.remove());
    // Pill tint per log type: success=green, error/warn=muted, progress/info=blue.
    const pillCls = cls === "ok" ? "ok" : (cls === "err" || cls === "warn") ? "muted" : "info";
    // A completed phase gets the green checkmark row. Detected by text (not cls, since
    // start messages like "Apply started" are also cls=ok) and never on err/warn rows.
    const completion =
      cls !== "err" && cls !== "warn" &&
      /\b(finished|done|rebuilt|verified|repaired|learned|applied)\b/i.test(msg);
    const el = document.createElement("div");
    el.className = "logline" + (cls ? " " + cls : "") + (completion ? " active-highlight" : "");
    el.dataset.t = "1";
    const pill = document.createElement("span");
    pill.className = "time-pill " + pillCls;
    pill.textContent = new Date().toLocaleTimeString();
    const body = document.createElement("span");
    body.textContent = msg;
    el.appendChild(pill);
    el.appendChild(body);
    box.prepend(el);
    while (box.children.length > 200) box.lastChild.remove();
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
    const add = (cls, text) => out.push({ cls, text });
    if (it.locked) add("lock", "Locked");
    if (num(it.rating) === -1) add("warn", "Rejected");
    if (it.is_favorite) add("fav", "\u2665 Favourite");
    if (num(it.rating) > 0) add("", "\u2605" + num(it.rating));
    if (it.is_best_shot) add("best", "Best shot");
    if (it.is_duplicate) add("warn", "Duplicate");
    if (it.near_dup_of != null) add("warn", "Near duplicate");
    if (it.is_alternate) add("", "Alternate");
    if (it.has_blinks) add("warn", "Blink");
    if (it.is_blurry) add("", "Blurry");
    if (it.is_too_dark) add("", "Dark");
    if (it.is_overexposed) add("", "Overexposed");
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
    const cls = ["card", it.locked ? "locked" : "", picked ? "picked" : "", num(it.rating) === -1 ? "rejected" : ""]
      .filter(Boolean).join(" ");
    const img = el("img", { loading: "lazy", decoding: "async", src: thumbUrl(id), alt: it.filename || "" });
    img.onerror = () => { img.style.opacity = ".15"; };
    const lockTitle = (it.locked ? "Unlock" : "Lock") + " (l)";
    const thumb = el("div", { class: "thumb" }, img,
      el("button", { class: "pick", "data-pick": id, "aria-label": "Select " + (it.filename || ""),
        "aria-pressed": String(picked), tabindex: -1, html: icon("check") }),
      el("div", { class: "hover-actions" },
        el("button", { "data-lock": id, title: lockTitle, "aria-label": it.locked ? "Unlock" : "Lock", tabindex: -1, html: icon("lock", "sm") }),
        el("button", { "data-similar": id, title: "Find similar photos", "aria-label": "Find similar photos", tabindex: -1, html: icon("copy", "sm") })),
      el("div", { class: "flags" }, flagList(it).slice(0, 3).map((f) => el("span", { class: "badge " + f.cls, text: f.text }))));
    const showReason = it.uncertain_reason && !it.manual_bucket;
    const cap = el("figcaption", { class: "caption" },
      el("div", { class: "fn", title: it.relative_path || it.filename, text: it.filename }),
      showReason ? el("div", { class: "reason", title: it.uncertain_reason.text, text: shortReason(it) }) : null,
      el("div", { class: "meta", text: meta.join(", ") }),
      el("div", { class: "catline", style: catStyle(rawCat) },
        el("button", { class: "catpick", type: "button", "data-catpick": id, title: "Change category",
          "aria-haspopup": "listbox", "aria-label": `Category ${catLabel(rawCat)}, change`, tabindex: -1 },
        el("i"), el("span", { class: "t", text: catLabel(rawCat) })),
        it.manual_bucket ? [el("span", { class: "tag", text: "edited" }),
          el("button", { class: "x", "data-clear": id, title: "Back to the AI choice",
            "aria-label": "Back to the AI choice", tabindex: -1, html: icon("x", "sm") })] : null));
    return el("figure", { class: cls, "data-id": id, tabindex: -1, role: "gridcell" }, thumb, cap);
  }

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
    return d;
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
      renderCard: cardNode,
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
    g.reset = () => g.vg.reset();
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
        scrubParams(p);
        p.set("sort", "taken");  // oldest first, so the date scrubber means something
        return p.toString();
      },
      matches(it) {
        if (val("catfilter") && eff(it) !== val("catfilter")) return false;
        if (chk("manualonly") && !it.manual_bucket) return false;
        if (chk("f-fav") && !it.is_favorite) return false;
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
      params() {
        const c = val("junkfilter");
        return c ? "category=" + encodeURIComponent(c) : "group=junk";
      },
      matches(it) {
        const c = val("junkfilter");
        return c ? eff(it) === c : state.junk.has(eff(it));
      },
      empty: { icon: "trash", title: "No junk to review", text: "Screenshots, documents and duplicates show up here after you analyze your photos." },
    });
    makeGrid("uncertain", {
      el: "#uncertaingrid", more: "#uncertaingrid-more", count: "#uncertaincount",
      params: () => "category=uncertain",
      matches: (it) => eff(it) === "uncertain",
      empty: { icon: "check", title: "Nothing needs a look", text: "Every photo was sorted with confidence." },
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
    ["f-camera", "f-rating", "f-country"].forEach((id) => { const el = $("#" + id); if (el) el.value = ""; });
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
  function applyFacets(f) {
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
    ["f-camera", "f-rating", "f-country", "f-fav"].forEach((id) => $("#" + id).addEventListener("change", again));
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
      const b = $(".pick", c);
      if (b) b.setAttribute("aria-pressed", String(on));
    });
    const n = state.picks.size;
    document.body.classList.toggle("has-picks", n > 0);
    $("#bulkbar").hidden = n === 0;
    $("#bulk-count").textContent = `${fmtN(n)} selected`;
  }

  function clearPicks() { state.picks.clear(); state.lastPick = null; applyPicks(); }

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
    let ok = 0;
    for (const id of ids) {
      try {
        const prev = await reassign(id, bucket);
        state.undo.push({ id, prev });
        ok++;
      } catch (e) { toast(e.message, "err"); break; }
    }
    if (ok) toast(`Moved ${fmtN(ok)} photo${ok === 1 ? "" : "s"} to ${catLabel(bucket)}`, "ok");
    clearPicks();
    loadStats();
  }

  async function bulkLock() {
    const ids = Array.from(state.picks);
    for (const id of ids) {
      try { await setLock(id, true); } catch (e) { toast(e.message, "err"); break; }
    }
    toast(`Locked ${fmtN(ids.length)} photos`, "ok");
    clearPicks();
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
    panel.hidden = false;
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

  function setCategoryFilter(cat) {
    $("#catfilter").value = cat || "";
    updateResetFiltersButton();
    showTab("categories", true);
    grids.cat.reset();
  }

  function bindDelegates() {
    document.addEventListener("click", async (ev) => {
      const t = ev.target.closest && ev.target.closest(
        "[data-pick],[data-catpick],[data-clear],[data-lock],[data-similar],[data-more],[data-goto],[data-clearfilters],[data-cat],[data-sortnow],[data-copy],[data-copytext],[data-reanalyze],.thumb");
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
        else if (t.dataset.similar) {
          closeLightbox();
          await showSimilar(num(t.dataset.similar));
        } else if (t.dataset.more) {
          const g = grids[t.dataset.more];
          if (g) await g.loadMore();
        } else if (t.dataset.goto) { ev.preventDefault(); showTab(t.dataset.goto); }
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
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        const goto = ev.target.closest && ev.target.closest("[data-goto]");
        if (goto && goto.dataset.goto) {
          ev.preventDefault();
          showTab(goto.dataset.goto);
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
      lb.ids = $$(".card", card.parentElement).map((c) => num(c.dataset.id));
      lb.i = Math.max(0, lb.ids.indexOf(id));
    }
    lbShow(card);
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
    $("#lightbox").hidden = false;
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
    $("#lightbox").hidden = true;
    document.body.style.overflow = "";
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
    img.src = thumbUrl(id, "micro");  // already cached from the grid: shows instantly
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
    $("#picker").hidden = false;
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
    $("#picker").hidden = true;
    document.body.style.overflow = "";
    if (pk.opener && document.contains(pk.opener)) {
      pk.opener.focus({ preventScroll: true });
    }
  }

  function bindPicker() {
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
      onPick: (p) => { const i = $("#scan-path"); i.value = p; i.classList.remove("input-error"); },
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
    const key = { categories: "cat", highlights: "highlights", events: "event", junk: "junk", uncertain: "uncertain" }[state.tab];
    return key ? grids[key] || null : null;
  }

  const isTriage = (g) => g === grids.uncertain || g === grids.junk;
  const MODALS = ["#picker", "#help"];
  const SELF_MANAGED = ["#confirm-modal", "#auth-modal", "#restart-overlay"];  // they handle their own keys
  const open = (sel) => { const e = $(sel); return !!(e && !e.hidden); };
  const anyModal = () => MODALS.some(open);

  let helpOpener = null;
  let helpReleaseTrap = null;

  function openHelp() {
    helpOpener = document.activeElement;
    $("#help").hidden = false;
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
    $("#help").hidden = true;
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
      else if (lb.open) closeLightbox();
      else if (!$("#similar-panel").hidden) $("#similar-panel").hidden = true;
      else if (state.picks.size) clearPicks();
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
  async function renderBursts() {
    const data = await api("/api/bursts");
    const list = $("#burstlist");
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
      `<div class="burstrow" role="button" tabindex="0" data-burst="${num(b.id)}">
        <div><div class="name">${esc(b.name)}</div>
             <div class="info">${fmtN(b.image_count)} photos, best: ${esc(b.best_shot || "none")}</div></div>
        ${icon("right", "sm")}
      </div>`).join("");
    $$(".burstrow", list).forEach((row) => {
      const go = () => showBurst(num(row.dataset.burst));
      row.onclick = go;
      row.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } };
    });
    await showBurst(data.items[0].id);
  }

  // Tight face crops with the eye state, built from DOM nodes only.
  async function showFaces(imageId, host) {
    const data = await api(`/api/images/${num(imageId)}/faces`);
    host.replaceChildren();
    if (!data.items.length) {
      const p = document.createElement("p");
      p.className = "help";
      p.textContent = "No faces were found in this photo.";
      host.appendChild(p);
      return;
    }
    const wording = { open: "Eyes open", possible: "Eyes may be closing", closed: "Eyes closed", unknown: "Eyes not measured" };
    data.items.forEach((f) => {
      const chip = document.createElement("figure");
      chip.className = "facechip" + (f.is_blink ? " blink" : "");
      if (f.url) {
        const img = document.createElement("img");
        img.loading = "lazy";
        img.alt = "Face crop";
        img.src = f.url;
        chip.appendChild(img);
      }
      const cap = document.createElement("figcaption");
      const score = f.eye_closed_score != null ? " (" + num(f.eye_closed_score).toFixed(2) + ")" : "";
      cap.textContent = (wording[f.eye_status] || wording.unknown) + score;
      chip.appendChild(cap);
      host.appendChild(chip);
    });
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

  async function showBurst(id) {
    $$(".burstrow", $("#burstlist")).forEach((r) => r.classList.toggle("on", num(r.dataset.burst) === id));
    const data = await api("/api/burst/" + num(id));
    $("#burstdetail").innerHTML = `<h3>${esc(data.burst)}</h3>
      <div class="btn-row" id="blinkbar"></div>
      <div class="burstgrid">` +
      data.members.map((m) => {
        const bits = [];
        if (m.sharpness_norm != null) bits.push("sharpness " + num(m.sharpness_norm).toFixed(2));
        if (m.face_confidence != null) bits.push("face " + num(m.face_confidence).toFixed(2));
        if (m.eye_closed != null) bits.push("eyes closed " + num(m.eye_closed).toFixed(2));
        return `<figure class="card${m.is_best_shot ? " best" : ""}" data-id="${num(m.id)}" tabindex="0">
          <div class="thumb">
            <img loading="lazy" src="${thumbUrl(m.id)}" alt="${esc(m.filename)}" onerror="this.style.opacity=.15">
            <div class="flags">${m.is_best_shot ? '<span class="badge best">Best shot</span>' : '<span class="badge">Alternate</span>'}${m.has_blinks ? '<span class="badge warn">Blink</span>' : ""}${m.manual_bucket === "junk_blinks" ? '<span class="badge">In Junk review</span>' : ""}</div>
          </div>
          <figcaption class="caption">
            <div class="fn">${esc(m.filename)}</div>
            <div class="meta">${num(m.width)}×${num(m.height)}${m.taken_at ? ", " + esc(String(m.taken_at).slice(11, 19)) : ""}</div>
            <div class="meta">${esc(bits.join(", "))}</div>
            ${m.face_count ? `<button class="mini" data-faces="${num(m.id)}">Inspect faces</button>` : ""}
          </figcaption>
        </figure>`;
      }).join("") + `</div><div id="facepanel" class="facepanel"></div>`;
    const bar = $("#blinkbar");
    const addBtn = (label, fn) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn";
      b.textContent = label;
      b.onclick = fn;
      bar.appendChild(b);
    };
    if (data.can_reject_blinks) addBtn("Reject all blinks", () => rejectBlinks(id, data.reject_count));
    if (data.rejected_blinks) addBtn(`Restore ${data.rejected_blinks} from Junk review`, () => restoreBlinks(id));
    $$("[data-faces]", $("#burstdetail")).forEach((b) => {
      b.onclick = () => showFaces(num(b.dataset.faces), $("#facepanel")).catch((e) => toast(e.message, "err"));
    });
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
      ${confirmable ? `<div class="btn-row" style="margin-top:14px"><button id="diff-confirm" class="btn primary" ${d.enough_space ? "" : "disabled"}>${verb} ${fmtN(d.total)} photos</button>
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
    const visible = p.active || done || bad || stopped;
    show(card, visible);
    // C8: the equalizer spectrum (M5 CSS) animates only while #progress-card
    // is .active, so gate the eq-pulse keyframes on the live running state.
    card.classList.toggle("active", p.active);
    show($("#hero-gauge"), visible);
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
    $("#progress-ratio").textContent = p.total > 0 ? `${fmtN(p.done)} / ${fmtN(p.total)}` : "";
    // Gauge: the arc is 100 units long (pathLength), so the dash length is the percent.
    const gaugePct = p.active ? Math.max(num(p.percent), 2) : (done ? 100 : Math.max(num(p.percent), 0));
    const phase = (String(p.message || "").match(/Phase\s+\d+\s*\/\s*\d+/i) || [""])[0];
    const gauge = $("#hero-gauge");
    gauge.setAttribute("aria-valuenow", String(Math.round(gaugePct)));
    $("#gauge-arc").style.strokeDasharray = `${gaugePct} 100`;
    $("#gauge-arc").setAttribute("class", "gauge-arc " + cls);
    $("#gauge-phase").textContent = bad ? "Error" : (stopped ? "Stopped" : (phase || p.task || "Working"));
    $("#gauge-pct").textContent = bad || stopped ? "-" : `${num(p.percent)}%`;

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
      b.disabled = busy || (b.id === "btn-cluster" && !state.inferred);
      b.classList.toggle("busy", busy);
    });
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
        <p>No AI model is installed, so a placeholder is sorting your photos. It cannot tell a screenshot from a landscape, which is why most photos land in Needs a look. Install the model, restart the app, then analyze again.</p>
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
        ? `${fmtN(moved)} photo${moved === 1 ? "" : "s"} moved out of Needs a look`
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
      $("#btn-sort-dock").disabled = !total;
    }
    $$(".comp-host").forEach((host) => {
      const head = `<div class="comp-head"><span class="comp-total">${fmtN(total)}</span>
        <span class="comp-label">${total === 1 ? "photo" : "photos"} in your library${total ? "" : ". Import a folder to see it sorted here."}</span></div>`;
      if (!total) {
        host.innerHTML = head + '<div class="comp-strip empty-strip"></div>';
        return;
      }
      const strip = cats.map(([c, n]) =>
        `<button style="${catStyle(c)};flex:${n} 1 0" data-cat="${esc(c === "unrouted" ? "" : c)}" title="${esc(catLabel(c))}: ${fmtN(n)}" aria-label="${esc(catLabel(c))}: ${fmtN(n)} photos"></button>`).join("");
      const legend = cats.map(([c, n]) =>
        `<button style="${catStyle(c)}" data-cat="${esc(c === "unrouted" ? "" : c)}"><i></i><span>${esc(catLabel(c))}</span> <b>${fmtN(n)}</b></button>`).join("");
      const unsorted = cats.length === 1 && cats[0][0] === "unrouted";
      host.innerHTML = head + `<div class="comp-strip">${strip}</div>
        <div class="comp-legend">${active ? '<button data-cat="">Show all</button>' : ""}${legend}</div>` +
        (unsorted ? `<p class="help" style="margin-top:12px">Photos are placed in categories when a plan is built. <button class="btn sm" data-sortnow>Sort photos now</button></p>` : "");
    });
  }

  function renderCounts(ov) {
    const s = ov.stats || {};
    const set = (k, v, alert) => {
      const e = $(`[data-count="${k}"]`);
      if (!e) return;
      e.textContent = v ? fmtN(v) : "";
      e.classList.toggle("alert", !!(alert && v));
    };
    const junk = Object.entries(ov.by_category).reduce((a, [c, n]) => a + (state.junk.has(c) ? n : 0), 0);
    set("library", ov.total);
    set("categories", ov.total);
    set("bursts", s.burst_groups);
    set("uncertain", ov.by_category.uncertain || 0, true);
    set("junk", junk);
    set("actions", s.planned_actions);

    const total = ov.total, pending = num(s.pending_inference), bursts = num(s.burst_groups);
    const inferred = total > 0 && pending === 0;
    state.inferred = inferred;
    const step = (id, cls, text) => {
      const li = $("#step-" + id);
      if (!li) return;
      li.classList.toggle("done", cls === "done");
      li.classList.toggle("active", cls === "active");
      li.classList.toggle("locked", cls === "locked");
      $("#state-" + id).textContent = text;
    };
    // Step 1 — done once indexed, otherwise the actionable focus (never locked).
    step("scan", total > 0 ? "done" : "active", total > 0 ? `${fmtN(total)} indexed` : "Not started");
    // Step 2 — done once inferred, the active focus while photos await analysis, locked with no library.
    step("infer", inferred ? "done" : total > 0 ? "active" : "locked",
      pending > 0 ? `${fmtN(pending)} waiting` : total > 0 ? "Done" : "Needs step 1");
    // Step 3 — done once grouped, active when analysis is ready, locked until step 2 completes.
    step("cluster", bursts > 0 ? "done" : inferred ? "active" : "locked",
      bursts > 0 ? `${fmtN(bursts)} groups` : inferred ? "Not run yet" : "Needs step 2");
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
  }

  // ------------------------------------------------------------------- tabs
  function fillCatFilters() {
    const opts = state.cats.map((c) => `<option value="${esc(c)}">${esc(catLabel(c))}</option>`).join("");
    $("#catfilter").innerHTML = '<option value="">All categories</option>' + opts;
    $("#onlycat").innerHTML = '<option value="">Every category</option>' + opts;
    $("#bulk-cat").innerHTML = '<option value="" disabled selected>Move to category</option>' + opts;
  }

  function showTab(name, keepPicks) {
    state.tab = name;
    state.triageGrid = null;
    if (!keepPicks) clearPicks();
    $$(".tab").forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("active", on);
      if (on) t.setAttribute("aria-current", "page"); else t.removeAttribute("aria-current");
    });
    $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
    const [title, sub] = TITLES[name] || [name, ""];
    $("#page-title").textContent = title;
    $("#page-sub").textContent = sub;
    document.title = title + " - Photocure";
    document.body.classList.remove("nav-open");
    try { history.replaceState(null, "", "#" + name); } catch (e) { /* ignore */ }
    window.scrollTo(0, 0);

    const gridTab = { categories: "cat", junk: "junk", uncertain: "uncertain", highlights: "highlights" }[name];
    if (gridTab) {
      const g = grids[gridTab];
      if (g) {
        if (g.total == null || g.stale) { g.stale = false; g.reset(); }
        else { g.update(); }
      }
    }
    if (name === "bursts") renderBursts().catch((e) => toast(e.message, "err"));
    if (name === "events") loadEvents().catch((e) => toast(e.message, "err"));
    if (name === "actions") loadTransactions().catch(() => {});
    if (name === "settings") refreshSettings(true);
    if (name === "uncertain") renderLegend();
    if (state.overview) renderComp(state.overview);
  }

  // ------------------------------------------------------------------- init
  async function runTask(label, fn) {
    state.taskRunning = true;
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
      if (pathInput) pathInput.value = "";
      ["#state-scan", "#state-infer", "#state-cluster"].forEach((sel) => {
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
      libLog("Choose a photo folder first", "err");
      toast("Choose a photo folder first", "err");
      $("#btn-browse-scan").focus();
      return null;
    }
    return path;
  }

  // Placing photos in categories happens when a plan is built, so do it after analysis.
  async function sortNow(silent) {
    try {
      await post("/api/plan/rebuild", planBody());
      if (!silent) toast("Photos sorted into categories", "ok");
    } catch (e) {
      if (!silent) toast(e.message, "err");
    }
    await loadStats();
    resetVisibleGrids();
  }

  async function afterTask(sort) {
    if (sort) await sortNow(true);
    else { await loadStats(); resetVisibleGrids(); }
    refreshBackend();
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

  async function loadAboutInfo() {
    try {
      const about = await api("/api/about");
      const versionBadge = $("#header-version-badge");
      if (versionBadge) versionBadge.textContent = about.version || "0.0.0";
      const sidebarVersion = $("#sidebar-version");
      if (sidebarVersion) sidebarVersion.textContent = about.version || "0.0.0";

      const aboutCard = $("#about-card");
      if (aboutCard) {
        const modelsList = (about.models || []).length > 0
          ? `<div style="margin-top:12px;"><strong>Active models:</strong><ul style="margin:6px 0;padding-left:20px;">${about.models.map(m => `<li>${esc(m.name)}</li>`).join("")}</ul></div>`
          : "";
        aboutCard.innerHTML = `
          <div style="font-size:13px;line-height:1.6;color:var(--mut);">
            <p style="margin:0 0 8px;">Version: <strong>${esc(about.version || "0.0.0")}</strong></p>
            <p style="margin:0 0 8px;">By <strong>${esc(about.author || "Unknown")}</strong> • ${esc(about.license || "MIT")}</p>
            <p style="margin:0 0 8px;">Acceleration: <strong>${esc(about.device || "CPU")}</strong></p>
            <p style="margin:0 0 8px;">Python: ${esc(about.python || "unknown")}</p>
            ${modelsList}
            <p style="margin:10px 0 0;"><a href="/licenses" target="_blank" class="link">View full licenses</a></p>
          </div>
        `;
      }
    } catch (e) {
      console.warn("Could not load about info:", e);
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
      if (!$("#orgroot").value) $("#orgroot").value = cfg.default_output_dir;
    } catch (e) {
      state.cats = sortCats(["people", "food", "scenery", "animal", "screenshot", "document",
        "graphic", "accidental", "duplicate", "uncertain", "unknown"]);
      state.junk = new Set(["screenshot", "document", "graphic", "accidental", "duplicate"]);
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

    const dens = $("#density");
    let saved = null;
    try { saved = localStorage.getItem("pcur-density"); } catch (e) { /* ignore */ }
    if (saved) dens.value = saved;
    const setDens = () => {
      document.documentElement.style.setProperty("--thumb-min", dens.value + "px");
      try { localStorage.setItem("pcur-density", dens.value); } catch (e) { /* ignore */ }
      allGrids().forEach((g) => g.vg.update());  // card size changed, so the rows are different
    };
    dens.addEventListener("input", setDens);
    setDens();

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
    $("#minaes").addEventListener("change", () => grids.highlights.reset());
    $("#similar-close").onclick = () => { $("#similar-panel").hidden = true; };
    $("#btn-dismiss-progress").onclick = () => { $("#progress-card").hidden = true; };

    $("#btn-learn").onclick = learnNow;
    renderBanner();
    updateLearn();
    $("#bulk-clear").onclick = clearPicks;
    $("#bulk-lock").onclick = () => bulkLock();
    $("#bulk-cat").onchange = (e) => { if (e.target.value) bulkReassign(e.target.value); e.target.selectedIndex = 0; };

    const scanInput = $("#scan-path");
    scanInput.addEventListener("input", () => scanInput.classList.remove("input-error"));
    scanInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); $("#btn-scan").click(); }
    });

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
      const r = await runTask(`Analyzing photos (${device})`, () => post("/api/infer", { device }));
      if (r) {
        libLog(`Analysis finished: ${fmtN(r.processed)} photos, ${fmtN(r.faces_total)} faces in ${r.duration_s}s`, "ok");
        toast(`Analyzed ${fmtN(r.processed)} photos`, "ok");
        await afterTask(true);
      }
    };
    $("#btn-cluster").onclick = async () => {
      const r = await runTask("Grouping bursts", () => post("/api/cluster", {}));
      if (r) {
        libLog(`Grouping finished: ${fmtN(r.burst_groups)} burst groups`, "ok");
        toast(`Found ${fmtN(r.burst_groups)} burst groups`, "ok");
        await afterTask(true);
      }
    };
    $("#btn-ingest-all").onclick = async () => {
      const path = needPath();
      if (!path) return;
      const r = await runTask(`Importing ${path}`, () => post("/api/ingest", { path }));
      if (r) {
        libLog(`Import finished: ${fmtN(r.scan.new)} new, ${fmtN(r.infer.processed)} analyzed, ${fmtN(r.cluster.burst_groups)} burst groups`, "ok");
        toast("Import finished", "ok");
        await afterTask(true);
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
