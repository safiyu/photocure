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

  const TOKEN = (document.querySelector('meta[name="pcur-token"]') || {}).content || "";
  const PAGE = 200;
  const state = {
    tab: "library", cats: [], junk: new Set(), byId: {}, busy: false, taskRunning: false,
    triageSel: -1, triageGrid: null, undo: [], picks: new Set(), lastPick: null,
    overview: null, eventId: null, personId: null,
    backend: null, thresholds: null, manualCount: 0,
    settings: null, draft: { device: "auto", device_index: 0 },
  };

  const TITLES = {
    library: ["Import", "Choose a folder and let Photocure sort what it finds."],
    categories: ["Library", "Every photo, grouped by the category it was sorted into."],
    bursts: ["Bursts", "Shots taken seconds apart. The best one of each group is kept."],
    highlights: ["Highlights", "Your most appealing photos first."],
    events: ["Events", "Photos grouped by when and where they were taken."],
    people: ["People", "Faces grouped by the person they belong to."],
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

  // Accessible confirmation dialog with focus trap and Escape key.
  async function showConfirmDialog(options) {
    return new Promise((resolve) => {
      const opts = options || {};
      const variant = opts.variant || "primary";  // primary, warning, danger
      const cancelBtn = opts.cancelText || "Cancel";
      const confirmBtn = opts.confirmText || "OK";
      const modal = document.createElement("div");
      modal.setAttribute("role", "alertdialog");
      modal.setAttribute("aria-modal", "true");
      modal.setAttribute("aria-labelledby", "confirm-title");
      modal.setAttribute("aria-describedby", "confirm-message");
      modal.className = `confirm-modal confirm-${variant}`;
      modal.innerHTML = `
        <div class="confirm-backdrop"></div>
        <div class="confirm-content">
          <h2 id="confirm-title">${esc(opts.title || "Confirm")}</h2>
          <p id="confirm-message">${esc(opts.message || "")}</p>
          <div class="confirm-buttons">
            <button class="btn btn-secondary" data-action="cancel">${esc(cancelBtn)}</button>
            <button class="btn btn-${variant}" data-action="confirm">${esc(confirmBtn)}</button>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
      const backdrop = modal.querySelector(".confirm-backdrop");
      const cancelButton = modal.querySelector('[data-action="cancel"]');
      const confirmButton = modal.querySelector('[data-action="confirm"]');
      const focusableElems = [cancelButton, confirmButton];
      let focusIdx = variant === "danger" || variant === "warning" ? 0 : 1;  // Default focus on Cancel for danger
      focusableElems[focusIdx]?.focus();
      const close = (result) => {
        modal.remove();
        resolve(result);
      };
      const handleKey = (e) => {
        if (e.key === "Escape") {
          e.preventDefault();
          close(false);
        } else if (e.key === "Tab") {
          focusIdx = (focusIdx + (e.shiftKey ? -1 : 1)) % focusableElems.length;
          focusableElems[focusIdx]?.focus();
          e.preventDefault();
        }
      };
      cancelButton?.addEventListener("click", () => close(false));
      confirmButton?.addEventListener("click", () => close(true));
      backdrop?.addEventListener("click", () => close(false));
      modal.addEventListener("keydown", handleKey);
    });
  }

  async function api(path, opts) {
    opts = opts || {};
    const headers = Object.assign({}, opts.headers || {});
    const needsToken = (opts.method && opts.method !== "GET") || path.startsWith("/api/browse");
    if (needsToken) headers["X-PCUR-Token"] = TOKEN;
    const res = await fetch(path, Object.assign({}, opts, { headers }));
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const j = await res.json();
        detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail || j);
      } catch (e) { /* keep status text */ }
      throw new Error(detail || "HTTP " + res.status);
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
  const thumbUrl = (id) => "/api/thumbnail/" + num(id);
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
    const el = document.createElement("div");
    el.className = "logline " + (cls || "");
    el.dataset.t = "1";
    el.textContent = new Date().toLocaleTimeString() + "  " + msg;
    box.prepend(el);
    while (box.children.length > 200) box.lastChild.remove();
  }
  const logLine = (m, c) => logTo("#action-log", m, c);
  const libLog = (m, c) => logTo("#library-log", m, c);

  // ------------------------------------------------------------------ theme
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("pcur-theme", t); } catch (e) { /* private mode */ }
    const b = $("#theme-toggle");
    if (b) b.innerHTML = icon(t === "dark" ? "sun" : "moon");
  }

  // ------------------------------------------------------------------ cards
  function flagBadges(it) {
    const out = [];
    if (it.locked) out.push('<span class="badge lock">Locked</span>');
    if (it.is_best_shot) out.push('<span class="badge best">Best shot</span>');
    if (it.is_duplicate) out.push('<span class="badge warn">Duplicate</span>');
    if (it.near_dup_of != null) out.push('<span class="badge warn">Near duplicate</span>');
    if (it.is_alternate) out.push('<span class="badge">Alternate</span>');
    if (it.is_blurry) out.push('<span class="badge">Blurry</span>');
    if (it.is_too_dark) out.push('<span class="badge">Dark</span>');
    if (it.is_overexposed) out.push('<span class="badge">Overexposed</span>');
    return out.slice(0, 3).join("");
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

  function cardHTML(it) {
    const id = num(it.id);
    const rawCat = eff(it);
    const meta = [];
    if (it.width && it.height) meta.push(num(it.width) + "×" + num(it.height));
    if (it.taken_at) meta.push(String(it.taken_at).slice(0, 10));
    if (it.score != null) meta.push("match " + num(it.score).toFixed(2));
    const cls = ["card", it.locked ? "locked" : "", state.picks.has(id) ? "picked" : ""].join(" ");
    return `<figure class="${cls}" data-id="${id}" tabindex="0">
      <div class="thumb">
        <img loading="lazy" src="${thumbUrl(id)}" alt="${esc(it.filename)}" onerror="this.style.opacity=.15">
        <button class="pick" data-pick="${id}" aria-label="Select ${esc(it.filename)}" aria-pressed="${state.picks.has(id)}">${icon("check")}</button>
        <div class="hover-actions">
          <button data-lock="${id}" title="${it.locked ? "Unlock" : "Lock"} (l)" aria-label="${it.locked ? "Unlock" : "Lock"}">${icon("lock", "sm")}</button>
          <button data-similar="${id}" title="Find similar photos" aria-label="Find similar photos">${icon("copy", "sm")}</button>
        </div>
        <div class="flags">${flagBadges(it)}</div>
      </div>
      <figcaption class="caption">
        <div class="fn" title="${esc(it.relative_path || it.filename)}">${esc(it.filename)}</div>
        ${it.uncertain_reason && !it.manual_bucket ? `<div class="reason" title="${esc(it.uncertain_reason.text)}">${esc(shortReason(it))}</div>` : ""}
        <div class="meta">${esc(meta.join(", "))}</div>
        <div class="catline" style="${catStyle(rawCat)}"><i></i>
          <span class="t">${esc(catLabel(rawCat))}</span>
          ${it.manual_bucket ? `<span class="tag">edited</span><button class="x" data-clear="${id}" title="Back to the AI choice" aria-label="Back to the AI choice">${icon("x", "sm")}</button>` : ""}
        </div>
      </figcaption>
    </figure>`;
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

  function makeGrid(key, cfg) {
    const g = {
      key, cfg, items: [], total: null, loading: false, gen: 0,
      el: $(cfg.el), more: $(cfg.more), countEl: cfg.count ? $(cfg.count) : null,
    };
    g.reset = async function () {
      g.gen++;
      g.items = [];
      g.total = null;
      g.loading = false;
      g.el.innerHTML = skeletons(12);
      g.more.innerHTML = "";
      await g.loadMore();
    };
    g.updateMore = function () {
      const left = g.total == null ? 0 : g.total - g.items.length;
      g.more.innerHTML = left > 0
        ? `<button class="btn" data-more="${esc(key)}">Show more (${fmtN(left)} left)</button>` : "";
      if (g.countEl) {
        g.countEl.textContent = g.total == null ? "" : `${fmtN(g.items.length)} of ${fmtN(g.total)}`;
      }
      if (g.total === 0) {
        const e = typeof cfg.empty === "function" ? cfg.empty() : (cfg.empty || {});
        g.el.innerHTML = emptyHTML(Object.assign({ title: "No photos here" }, e));
      }
    };
    g.loadMore = async function () {
      if (g.loading || (g.total != null && g.items.length >= g.total)) return;
      g.loading = true;
      const gen = g.gen;
      try {
        const qs = cfg.params();
        const data = await api(`/api/images?${qs}${qs ? "&" : ""}limit=${PAGE}&offset=${g.items.length}`);
        if (gen !== g.gen) return;
        if (g.items.length === 0) g.el.innerHTML = "";
        data.items.forEach((it) => { state.byId[it.id] = it; });
        g.items = g.items.concat(data.items);
        g.total = data.total;
        g.el.insertAdjacentHTML("beforeend", data.items.map(cardHTML).join(""));
        g.updateMore();
        g.applySel();
      } catch (e) {
        g.el.innerHTML = emptyHTML({ icon: "alert", title: "Could not load photos", text: e.message });
        toast(e.message, "err");
      } finally {
        if (gen === g.gen) g.loading = false;
      }
    };
    g.cards = () => $$(".card", g.el);
    g.applySel = function () {
      if (state.triageGrid !== g) return;
      g.cards().forEach((c, i) => c.classList.toggle("sel", i === state.triageSel));
    };
    g.rerender = function (it) {
      const old = $(`.card[data-id="${num(it.id)}"]`, g.el);
      if (!old) return;
      const idx = g.items.findIndex((x) => x.id === it.id);
      if (idx >= 0) g.items[idx] = it;
      if (cfg.matches && !cfg.matches(it)) {
        old.remove();
        if (idx >= 0) g.items.splice(idx, 1);
        if (g.total != null) g.total = Math.max(0, g.total - 1);
        g.updateMore();
      } else {
        old.outerHTML = cardHTML(it);
      }
      g.applySel();
    };
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((es) => {
        if (es.some((e) => e.isIntersecting) && g.el.offsetParent !== null) g.loadMore();
      }, { rootMargin: "500px" }).observe(g.more);
    }
    grids[key] = g;
    return g;
  }

  const allGrids = () => Object.values(grids);
  function resetVisibleGrids() {
    allGrids().forEach((g) => { if (g.el.offsetParent !== null) g.reset(); else g.stale = true; });
  }
  function itemUpdated(it) {
    state.byId[it.id] = it;
    allGrids().forEach((g) => g.rerender(it));
    if (lb.open && lb.ids[lb.i] === it.id) lbRender();
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
        return p.toString();
      },
      matches(it) {
        if (val("catfilter") && eff(it) !== val("catfilter")) return false;
        if (chk("manualonly") && !it.manual_bucket) return false;
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
    makeGrid("person", {
      el: "#peopledetail", more: "#peopledetail-more",
      params: () => "person_id=" + num(state.personId),
      empty: { icon: "users", title: "This person has no photos" },
    });
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

  function togglePick(id, shift) {
    const card = $(`.card[data-id="${id}"]`);
    const box = card ? card.parentElement : null;
    if (shift && state.lastPick && box && state.lastPick.box === box) {
      const ids = $$(".card", box).map((c) => num(c.dataset.id));
      const a = ids.indexOf(state.lastPick.id), b = ids.indexOf(id);
      if (a >= 0 && b >= 0) ids.slice(Math.min(a, b), Math.max(a, b) + 1).forEach((i) => state.picks.add(i));
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
    $("#similar-grid").innerHTML = skeletons(8);
    panel.hidden = false;
    try {
      const data = await api(`/api/images/${num(id)}/similar?k=24`);
      data.items.forEach((x) => { state.byId[x.id] = x; });
      $("#similar-grid").innerHTML = data.items.length
        ? data.items.map(cardHTML).join("")
        : emptyHTML({ icon: "copy", title: "No similar photos", text: "Analyze the library first, then try again." });
    } catch (e) {
      $("#similar-grid").innerHTML = emptyHTML({ icon: "alert", title: "Search failed", text: e.message });
    }
  }

  function setCategoryFilter(cat) {
    $("#catfilter").value = cat || "";
    showTab("categories", true);
    grids.cat.reset();
  }

  function bindDelegates() {
    document.addEventListener("click", async (ev) => {
      const t = ev.target.closest && ev.target.closest(
        "[data-pick],[data-clear],[data-lock],[data-similar],[data-more],[data-goto],[data-clearfilters],[data-cat],[data-sortnow],[data-copy],[data-copytext],[data-reanalyze],.thumb");
      if (!t) return;
      try {
        if (t.dataset.pick) togglePick(num(t.dataset.pick), ev.shiftKey);
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
          $("#q").value = ""; $("#catfilter").value = "";
          $("#manualonly").checked = false; $("#neardup").checked = false;
          grids.cat.reset();
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
      if ((ev.key === "Enter" || ev.key === " ") && ev.target.classList && ev.target.classList.contains("card")) {
        ev.preventDefault();
        openLightbox(ev.target);
      }
    });
  }

  // ------------------------------------------------------------- lightbox
  const lb = { open: false, ids: [], i: 0, token: 0 };

  function openLightbox(card) {
    const box = card.parentElement;
    lb.ids = $$(".card", box).map((c) => num(c.dataset.id));
    lb.i = Math.max(0, lb.ids.indexOf(num(card.dataset.id)));
    lb.open = true;
    lb.opener = card;
    $("#lightbox").hidden = false;
    document.body.style.overflow = "hidden";
    lbRender();
    $("#lb-close").focus();
  }

  function closeLightbox() {
    if (!lb.open) return;
    lb.open = false;
    $("#lightbox").hidden = true;
    document.body.style.overflow = "";
    if (lb.opener && document.contains(lb.opener)) lb.opener.focus({ preventScroll: true });
  }

  function lbStep(d) {
    const n = lb.i + d;
    if (n < 0 || n >= lb.ids.length) return;
    lb.i = n;
    lbRender();
  }

  function lbRender() {
    const id = lb.ids[lb.i];
    const card = $(`.card[data-id="${id}"]`);
    const it = state.byId[id] || { id, filename: card ? ($(".fn", card) || {}).textContent : "Photo #" + id, _basic: true };
    const img = $("#lb-img");
    const my = ++lb.token;
    img.alt = it.filename || "";
    img.src = thumbUrl(id);
    $("#lb-loading").hidden = false;
    const big = new Image();
    big.onload = () => { if (my === lb.token) { img.src = big.src; $("#lb-loading").hidden = true; } };
    big.onerror = () => { if (my === lb.token) $("#lb-loading").hidden = true; };
    big.src = "/api/preview/" + num(id);

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
    flagBadges(it).replace(/<[^>]+>/g, "|").split("|").filter(Boolean).forEach((f, k) => { if (k === 0) add("Notes", f); });
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
    $("#lb-next").disabled = lb.i >= lb.ids.length - 1;
    $("#lb-count").textContent = `${lb.i + 1} of ${lb.ids.length}`;
    const nb = lb.ids[lb.i + 1];
    if (nb) new Image().src = "/api/preview/" + num(nb);
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
    $("#lightbox").addEventListener("click", (e) => {
      if (e.target.id === "lightbox" || e.target.classList.contains("lb-stage")) closeLightbox();
    });
    $("#lb-cat").onchange = async (e) => {
      const id = lb.ids[lb.i];
      try {
        const prev = await reassign(id, e.target.value);
        state.undo.push({ id, prev });
        toast("Moved to " + catLabel(e.target.value), "ok");
        loadStats();
      } catch (err) { toast(err.message, "err"); }
    };
    $("#lb-lock").onclick = () => toggleLock(lb.ids[lb.i]).catch((e) => toast(e.message, "err"));
    $("#lb-similar").onclick = () => { const id = lb.ids[lb.i]; closeLightbox(); showSimilar(id); };
    $("#lb-clear").onclick = async () => {
      const id = lb.ids[lb.i];
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
    pk.onPick = opts.onPick;
    pk.mode = opts.mode || "photos";
    $("#picker-title").textContent = opts.title || "Choose a folder";
    $("#picker-ok").textContent = opts.okLabel || "Use this folder";
    $("#picker").hidden = false;
    document.body.style.overflow = "hidden";
    pickerLoad(opts.initial || null).then(() => $("#picker-ok").focus());
  }

  function closePicker() {
    $("#picker").hidden = true;
    document.body.style.overflow = "";
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
      const name = prompt("Name for the new folder:");
      if (!name) return;
      try {
        const r = await post("/api/browse/mkdir", { parent: pk.data.path, name });
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

  function activeTriageGrid() {
    if (state.tab === "uncertain") return grids.uncertain;
    if (state.tab === "junk") return grids.junk;
    return null;
  }

  function moveSel(g, delta) {
    const cs = g.cards();
    if (!cs.length) return;
    state.triageGrid = g;
    state.triageSel = Math.min(cs.length - 1, Math.max(0, state.triageSel + delta));
    g.applySel();
    cs[state.triageSel].scrollIntoView({ block: "nearest" });
    if (state.triageSel >= cs.length - 3) g.loadMore();
  }

  const anyModal = () => !$("#picker").hidden || !$("#help").hidden;

  async function onKey(ev) {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const tag = (ev.target.tagName || "").toLowerCase();
    const typing = tag === "input" || tag === "select" || tag === "textarea";

    if (ev.key === "Escape") {
      if (!$("#picker").hidden) closePicker();
      else if (!$("#help").hidden) $("#help").hidden = true;
      else if (lb.open) closeLightbox();
      else if (!$("#similar-panel").hidden) $("#similar-panel").hidden = true;
      else if (state.picks.size) clearPicks();
      else document.body.classList.remove("nav-open");
      return;
    }
    if (typing || anyModal()) return;
    if (ev.key === "?") { $("#help").hidden = false; ev.preventDefault(); return; }

    if (lb.open) {
      if (ev.key === "ArrowLeft") lbStep(-1);
      else if (ev.key === "ArrowRight") lbStep(1);
      else if (ev.key.toLowerCase() === "l") toggleLock(lb.ids[lb.i]).catch((e) => toast(e.message, "err"));
      else return;
      ev.preventDefault();
      return;
    }

    const g = activeTriageGrid();
    if (!g) return;
    state.triageGrid = g;
    const k = ev.key.toLowerCase();
    try {
      if (k === "j") moveSel(g, 1);
      else if (k === "k") moveSel(g, -1);
      else if (k === " " || k === "enter") {
        const card = g.cards()[state.triageSel];
        if (!card) return;
        openLightbox(card);
      } else if (/^[1-9]$/.test(k)) {
        const card = g.cards()[state.triageSel];
        const cat = triageCats()[Number(k) - 1];
        if (!card || !cat) return;
        const id = num(card.dataset.id);
        const prev = await reassign(id, cat);
        state.undo.push({ id, prev });
        toast(`Moved to ${catLabel(cat)}`, "ok");
        const left = g.cards().length;
        if (state.triageSel >= left) state.triageSel = left - 1;
        g.applySel();
        loadStats();
      } else if (k === "u") {
        const last = state.undo.pop();
        if (!last) { toast("Nothing to undo", "err"); return; }
        await reassign(last.id, last.prev);
        toast("Change undone", "ok");
        await g.reset();
      } else if (k === "l") {
        const card = g.cards()[state.triageSel];
        if (card) await toggleLock(num(card.dataset.id));
      } else return;
      ev.preventDefault();
    } catch (e) { toast(e.message, "err"); }
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

  async function showBurst(id) {
    $$(".burstrow", $("#burstlist")).forEach((r) => r.classList.toggle("on", num(r.dataset.burst) === id));
    const data = await api("/api/burst/" + num(id));
    $("#burstdetail").innerHTML = `<h3>${esc(data.burst)}</h3><div class="burstgrid">` +
      data.members.map((m) => {
        const bits = [];
        if (m.sharpness_norm != null) bits.push("sharpness " + num(m.sharpness_norm).toFixed(2));
        if (m.face_confidence != null) bits.push("face " + num(m.face_confidence).toFixed(2));
        if (m.ear != null) bits.push("eyes " + num(m.ear).toFixed(2));
        return `<figure class="card${m.is_best_shot ? " best" : ""}" data-id="${num(m.id)}" tabindex="0">
          <div class="thumb">
            <img loading="lazy" src="${thumbUrl(m.id)}" alt="${esc(m.filename)}" onerror="this.style.opacity=.15">
            <div class="flags">${m.is_best_shot ? '<span class="badge best">Best shot</span>' : '<span class="badge">Alternate</span>'}</div>
          </div>
          <figcaption class="caption">
            <div class="fn">${esc(m.filename)}</div>
            <div class="meta">${num(m.width)}×${num(m.height)}${m.taken_at ? ", " + esc(String(m.taken_at).slice(11, 19)) : ""}</div>
            <div class="meta">${esc(bits.join(", "))}</div>
          </figcaption>
        </figure>`;
      }).join("") + "</div>";
  }

  // ------------------------------------------------------- events and people
  function clusterRow(kind, c, on) {
    const title = kind === "event"
      ? (c.label || "Event " + num(c.id)) : (c.name || "Person " + num(c.id));
    const sub = kind === "event"
      ? `${fmtN(c.image_count)} photos, ${esc((c.start || "").slice(0, 10))} to ${esc((c.end || "").slice(0, 10))}`
      : `${fmtN(c.face_count)} faces`;
    const thumb = c.cover_id != null
      ? `<img class="rowthumb" loading="lazy" src="${thumbUrl(c.cover_id)}" alt="">` : "";
    return `<div class="burstrow${on ? " on" : ""}" role="button" tabindex="0" data-kind="${kind}" data-cid="${num(c.id)}">
      <div class="rowmain">${thumb}<div style="min-width:0"><div class="name">${esc(title)}</div>
        <div class="info">${sub}</div></div></div>
      <button class="mini" data-rename="${kind}:${num(c.id)}">Rename</button>
    </div>`;
  }

  async function loadClusters(kind) {
    const url = kind === "event" ? "/api/events" : "/api/people";
    const listEl = $(kind === "event" ? "#eventlist" : "#peoplelist");
    const cntEl = $(kind === "event" ? "#eventcount" : "#peoplecount");
    const gridKey = kind === "event" ? "event" : "person";
    const data = await api(url);
    cntEl.textContent = fmtN(data.count) + (kind === "event" ? " events" : " people");
    if (!data.items.length) {
      listEl.innerHTML = "";
      grids[gridKey].el.innerHTML = emptyHTML(kind === "event"
        ? { icon: "cal", title: "No events yet", text: "Choose the gap and distance above, then find events. Photos need dates to be grouped." }
        : { icon: "users", title: "No people yet", text: "Faces are grouped after analysis. Select Find people to start." });
      grids[gridKey].more.innerHTML = "";
      return;
    }
    const cur = kind === "event" ? state.eventId : state.personId;
    const selected = data.items.some((c) => c.id === cur) ? cur : data.items[0].id;
    if (kind === "event") state.eventId = selected; else state.personId = selected;
    listEl.innerHTML = data.items.map((c) => clusterRow(kind, c, c.id === selected)).join("");
    grids[gridKey].reset();
    $$(".burstrow", listEl).forEach((row) => {
      const pick = (e) => {
        if (e.target.closest("[data-rename]")) return;
        $$(".burstrow", listEl).forEach((r) => r.classList.remove("on"));
        row.classList.add("on");
        if (kind === "event") state.eventId = num(row.dataset.cid); else state.personId = num(row.dataset.cid);
        grids[gridKey].reset();
      };
      row.onclick = pick;
      row.onkeydown = (e) => { if ((e.key === "Enter" || e.key === " ") && e.target === row) { e.preventDefault(); pick(e); } };
    });
    $$("[data-rename]", listEl).forEach((b) => {
      b.onclick = async () => {
        const [k, id] = b.dataset.rename.split(":");
        const name = prompt("New name:");
        if (name == null || !name.trim()) return;
        try {
          await post(`/api/${k === "event" ? "events" : "people"}/${num(id)}/rename`, { name });
          await loadClusters(k);
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
        if (j.status !== "running") {
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
    if (!confirm("Undo this change? Files go back to where they were before.")) return;
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
    const topbar = $("#topbar-progress");
    const show = (el, on) => { if (el) el.hidden = !on; };
    const done = !p.active && !p.error && !p.stopped && p.percent === 100;
    const bad = !p.active && !!p.error;
    const stopped = !p.active && !!p.stopped;
    const visible = p.active || done || bad || stopped;
    show(card, visible);
    show(topbar, visible);
    if (!visible) return;
    const pct = p.active ? Math.max(num(p.percent), 4) : 100;
    const cls = p.active ? "active" : (bad || stopped ? "error" : "done");
    show($("#progress-spinner"), p.active);
    show($("#topbar-spinner"), p.active);
    [$("#progress-status-icon"), $("#topbar-status-icon")].forEach((e) => {
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
    $("#topbar-progress-text").textContent = `${p.task || "Task"} ${bad ? "failed" : (stopped ? "stopped" : num(p.percent) + "%")}`;
    $("#topbar-fill").className = "topbar-fill " + (p.active ? "" : cls);
    $("#topbar-fill").style.width = pct + "%";

    const isBusy = p.active || p.busy;
    [$("#btn-stop-progress"), $("#btn-stop-topbar")].forEach((btn) => {
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
    });
    updateLearn();
  }

  async function pollProgress() {
    let delay = 3000;
    try {
      const p = await api("/api/progress");
      renderProgress(p);
      applyBusy(!!p.busy || p.active);
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
    setTimeout(pollProgress, delay);
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
    $("#device-chip-text").textContent = "Running on " + label;
    chip.classList.toggle("gpu", !!st.selected && st.selected !== "cpu");
    chip.classList.toggle("warn", !!st.fallback_reason);
    chip.title = st.fallback_reason ? `Using ${label}. ${st.fallback_reason}` : "Open Settings";
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
    const body = Object.assign({ device: state.draft.device, device_index: num(state.draft.device_index) }, workers);
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
    const step = (id, done, text) => {
      const li = $("#step-" + id);
      if (!li) return;
      li.classList.toggle("done", done);
      $("#state-" + id).textContent = text;
    };
    step("scan", total > 0, total > 0 ? `${fmtN(total)} indexed` : "Not started");
    const inferred = total > 0 && pending === 0;
    step("infer", inferred, pending > 0 ? `${fmtN(pending)} waiting` : total > 0 ? "Done" : "Needs step 1");
    step("cluster", bursts > 0, bursts > 0 ? `${fmtN(bursts)} groups` : inferred ? "Not run yet" : "Needs step 2");
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
    state.triageSel = -1;
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
      if (g.total == null || g.stale) { g.stale = false; g.reset(); }
    }
    if (name === "bursts") renderBursts().catch((e) => toast(e.message, "err"));
    if (name === "events") loadClusters("event").catch((e) => toast(e.message, "err"));
    if (name === "people") loadClusters("person").catch((e) => toast(e.message, "err"));
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
    }
  }

  async function stopProcess() {
    const btns = [$("#btn-stop-progress"), $("#btn-stop-topbar")];
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
    if (!confirm("Are you sure you want to clear all indexed photos, AI analysis, bursts, and plans?\n\nThis will remove everything from the Photocure database.\nYour original photo files on disk will NOT be deleted or modified.")) {
      return;
    }
    const btns = [$("#btn-reset-library"), $("#btn-import-reset")];
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
      btns.forEach((b) => {
        if (b) {
          b.disabled = false;
          b.textContent = b.id === "btn-import-reset" ? "Reset library" : "Clear indexed library";
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

  async function init() {
    applyTheme(document.documentElement.dataset.theme || "light");
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
    fillCatFilters();
    setupGrids();
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
    $("#theme-toggle").onclick = () =>
      applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    $("#btn-help").onclick = () => { $("#help").hidden = false; };
    $("#help-close").onclick = () => { $("#help").hidden = true; };
    $("#help").addEventListener("click", (e) => { if (e.target.id === "help") $("#help").hidden = true; });

    const dens = $("#density");
    let saved = null;
    try { saved = localStorage.getItem("pcur-density"); } catch (e) { /* ignore */ }
    if (saved) dens.value = saved;
    const setDens = () => {
      document.documentElement.style.setProperty("--thumb-min", dens.value + "px");
      try { localStorage.setItem("pcur-density", dens.value); } catch (e) { /* ignore */ }
    };
    dens.addEventListener("input", setDens);
    setDens();

    ["catfilter", "manualonly", "neardup"].forEach((id) =>
      $("#" + id).addEventListener("change", () => { grids.cat.reset(); if (state.overview) renderComp(state.overview); }));
    let qt = null;
    $("#q").addEventListener("input", () => { clearTimeout(qt); qt = setTimeout(() => grids.cat.reset(), 250); });
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
        await loadClusters("event");
      } catch (e) { toast(e.message, "err"); }
    };
    $("#btn-people").onclick = async () => {
      try {
        const r = await post("/api/people/cluster", {});
        toast(`Found ${fmtN(r.people)} people`, "ok");
        await loadClusters("person");
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
    $("#btn-stop-topbar").onclick = stopProcess;
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
