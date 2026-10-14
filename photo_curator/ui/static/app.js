/* Photo Curator review dashboard (vanilla JS). */
(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const CATS = ["people", "food", "scenery", "animal", "screenshot", "document",
    "graphic", "accidental", "duplicate", "uncertain", "unknown"];
  const JUNK = new Set(["screenshot", "document", "graphic", "accidental", "duplicate"]);

  const state = { images: [], byId: {}, tab: "categories" };

  // ------------------------------------------------------------------ utils
  function toast(msg, cls) {
    const el = $("#toast");
    el.textContent = msg;
    el.className = "toast " + (cls || "");
    el.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => (el.hidden = true), 3200);
  }

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) {}
      throw new Error(detail || "HTTP " + res.status);
    }
    return res.json();
  }

  function thumbUrl(id) { return "/api/thumbnail/" + id; }

  function flagBadges(it) {
    let s = "";
    if (it.is_duplicate) s += '<span class="badge b dup">DUP</span>';
    if (it.is_alternate) s += '<span class="badge b alt">ALT</span>';
    if (it.is_best_shot) s += '<span class="badge b best">BEST</span>';
    if (it.face_count > 0) s += '<span class="badge b face">face</span>';
    if (it.is_blurry) s += '<span class="badge b blur">blurry</span>';
    if (it.is_too_dark) s += '<span class="badge b dark">dark</span>';
    if (it.is_overexposed) s += '<span class="badge b over">over</span>';
    return s;
  }

  function cardHTML(it) {
    const cat = it.manual_bucket || it.category || "unrouted";
    const meta = [];
    if (it.width) meta.push(it.width + "×" + it.height);
    if (it.taken_at) meta.push(it.taken_at.slice(0, 10));
    if (it.top1) meta.push(it.top1 + " " + it.top1_score);
    if (it.quality_score != null) meta.push("q " + it.quality_score);
    const manual = it.manual_bucket ? ` (manual: ${esc(it.manual_bucket)})` : "";
    return `<div class="card" data-id="${it.id}">
      <div class="thumb">
        <img loading="lazy" src="${thumbUrl(it.id)}" alt="" onerror="this.style.opacity=.15">
        <div class="flags">${flagBadges(it)}</div>
      </div>
      <div class="cardbody">
        <div class="fn" title="${esc(it.relative_path)}">${esc(it.filename)}</div>
        <div class="meta">${esc(meta.join(" · "))}</div>
        <span class="catchip">${esc(cat)}</span>${manual}
        <div class="actions">
          ${CATS.map((c) =>
            `<button class="mini" data-reassign="${c}" data-id="${it.id}">${c}</button>`
          ).join("")}
          <button class="mini danger" data-clear="${it.id}">clear</button>
        </div>
      </div>
    </div>`;
  }

  // ------------------------------------------------------------------ data
  async function loadImages() {
    const manual = $("#manualonly") ? $("#manualonly").checked : false;
    const data = await api("/api/images?limit=5000" + (manual ? "&manual=true" : ""));
    state.images = data.items;
    state.byId = {};
    state.images.forEach((it) => (state.byId[it.id] = it));
  }

  function fillCatFilter() {
    const sel = $("#catfilter");
    const cur = sel.value;
    const present = {};
    state.images.forEach((it) => {
      const c = it.manual_bucket || it.category;
      if (c) present[c] = 1;
    });
    sel.innerHTML = '<option value="">All categories</option>' +
      CATS.filter((c) => present[c]).map((c) =>
        `<option value="${c}">${c}</option>`).join("");
    sel.value = cur && present[cur] ? cur : "";
  }

  function renderGrid(el, items) {
    if (!items.length) {
      el.innerHTML = '<div class="empty">No images here.</div>';
      return;
    }
    el.innerHTML = items.map(cardHTML).join("");
    bindCardActions(el);
  }

  function bindCardActions(root) {
    $$("[data-reassign]", root).forEach((btn) => {
      btn.onclick = async () => {
        const id = btn.dataset.id;
        const bucket = btn.dataset.reassign;
        try {
          await api(`/api/images/${id}/reassign`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ bucket }),
          });
          toast(`#${id} → ${bucket}`, "ok");
          await refresh();
        } catch (e) { toast(e.message, "err"); }
      };
    });
    $$("[data-clear]", root).forEach((btn) => {
      btn.onclick = async () => {
        const id = btn.dataset.clear;
        try {
          await api(`/api/images/${id}/reassign`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ bucket: null }),
          });
          toast(`#${id} override cleared`, "ok");
          await refresh();
        } catch (e) { toast(e.message, "err"); }
      };
    });
  }

  // ------------------------------------------------------------------ tabs
  function renderCategories() {
    const q = ($("#q").value || "").toLowerCase();
    const cat = $("#catfilter").value;
    const manualOnly = $("#manualonly").checked;
    let items = state.images.filter((it) => {
      if (manualOnly && !it.manual_bucket) return false;
      if (cat && (it.manual_bucket || it.category) !== cat) return false;
      if (q && !(it.filename.toLowerCase().includes(q) ||
                 (it.relative_path || "").toLowerCase().includes(q))) return false;
      return true;
    });
    $("#rowcount").textContent = items.length + " / " + state.images.length;
    renderGrid($("#grid"), items);
  }

  function renderJunk() {
    const cat = $("#junkfilter").value;
    let items = state.images.filter((it) => {
      const c = it.manual_bucket || it.category;
      if (cat) return c === cat;
      return JUNK.has(c);
    });
    $("#junkcount").textContent = items.length + " items";
    renderGrid($("#junkgrid"), items);
  }

  function renderUncertain() {
    renderGrid($("#uncertaingrid"),
      state.images.filter((it) =>
        (it.manual_bucket || it.category) === "uncertain"));
  }

  // ------------------------------------------------------------------ bursts
  async function renderBursts() {
    const data = await api("/api/bursts");
    const list = $("#burstlist");
    if (!data.items.length) {
      list.innerHTML = '<div class="empty">No burst groups. Run `photo-curator cluster`.</div>';
      $("#burstdetail").innerHTML = "";
      return;
    }
    list.innerHTML = data.items.map((b) =>
      `<div class="burstrow" data-burst="${b.id}">
        <div><div class="name">${esc(b.name)}</div>
             <div class="info">${b.image_count} photos · best: ${esc(b.best_shot || "—")}</div></div>
        <div class="muted">›</div>
      </div>`).join("");
    $$(".burstrow", list).forEach((row) => {
      row.onclick = () => showBurst(parseInt(row.dataset.burst, 10));
    });
    if (data.items.length) await showBurst(data.items[0].id);
  }

  async function showBurst(id) {
    $$(".burstrow").forEach((r) =>
      r.style.borderColor = r.dataset.burst == id ? "var(--acc)" : "");
    const data = await api("/api/burst/" + id);
    const members = data.members;
    $("#burstdetail").innerHTML = `<h3>${esc(data.burst)}</h3><div class="burstgrid">` +
      members.map((m) => {
        const best = m.is_best_shot ? " best" : "";
        const score = m.sharpness_norm != null ? "q " + m.sharpness_norm : "";
        const face = m.face_confidence != null ? "· face " + m.face_confidence : "";
        const ear = m.ear != null ? "· EAR " + m.ear.toFixed(2) : "";
        return `<div class="card${best}" data-id="${m.id}">
          <div class="thumb">
            <img loading="lazy" src="${thumbUrl(m.id)}" alt="" onerror="this.style.opacity=.15">
            <div class="flags">${m.is_best_shot ? '<span class="badge b best">BEST SHOT</span>' : '<span class="badge b alt">ALT</span>'}</div>
          </div>
          <div class="cardbody">
            <div class="fn">${esc(m.filename)}</div>
            <div class="meta">${m.width}×${m.height}${m.taken_at ? " · " + esc(m.taken_at.slice(0,19).replace("T"," ")) : ""}</div>
            <div class="meta">${score}${face}${ear}</div>
          </div>
        </div>`;
      }).join("") + "</div>";
  }

  // ------------------------------------------------------------------ actions
  function logLine(msg, cls) {
    const el = document.createElement("div");
    el.className = "logline " + (cls || "");
    el.textContent = new Date().toLocaleTimeString() + "  " + msg;
    const box = $("#action-log");
    box.prepend(el);
    if (box.children.length > 60) box.lastChild.remove();
  }

  async function loadTransactions() {
    const data = await api("/api/transactions");
    const sel = $("#txnselect");
    sel.innerHTML = '<option value="">Latest</option>' +
      data.items.map((t) =>
        `<option value="${esc(t.transaction_id)}">${esc(t.transaction_id)}
         (${t.committed} committed / ${t.rolled_back} undone)</option>`).join("");
  }

  async function refresh() {
    await loadImages();
    fillCatFilter();
    renderCategories();
    renderJunk();
    renderUncertain();
  }

  // ------------------------------------------------------------------ tabs nav
  function showTab(name) {
    state.tab = name;
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
    if (name === "bursts") renderBursts();
    if (name === "categories") renderCategories();
    if (name === "junk") renderJunk();
    if (name === "uncertain") renderUncertain();
    if (name === "actions") loadTransactions();
  }

  // ------------------------------------------------------------------ init
  async function loadStats() {
    try {
      const s = await api("/api/stats");
      const labels = {
        total_images: "Images", pending_inference: "Pending",
        duplicates: "Dupes", planned_actions: "Planned", burst_groups: "Bursts",
      };
      $("#topstats").innerHTML = Object.keys(labels)
        .map((k) => `<div class="topstat"><b>${s[k] || 0}</b><span>${labels[k]}</span></div>`)
        .join("");
    } catch (e) { /* ignore */ }
  }

  async function init() {
    $$(".tab").forEach((t) => (t.onclick = () => showTab(t.dataset.tab)));
    ["q", "catfilter", "manualonly"].forEach((id) =>
      $("#" + id).addEventListener("input", renderCategories));
    $("#junkfilter").addEventListener("change", renderJunk);

    $("#btn-rebuild").onclick = async () => {
      try {
        const r = await api("/api/plan/rebuild", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            operation: $("#op").value, mode: $("#orgmode").value,
            output_dir: $("#orgroot").value.trim(),
          }),
        });
        const cats = Object.entries(r.by_category)
          .map(([c, n]) => `${c}:${n}`).join(" · ");
        $("#plan-summary").textContent =
          `Plan ${r.plan_id}: ${r.total} files across ${Object.keys(r.by_category).length} ` +
          `categories, ${r.burst_count} burst groups → ${r.organize_root}  (${cats})`;
        logLine(`Plan rebuilt: ${r.total} files`, "ok");
        await refresh();
      } catch (e) { toast(e.message, "err"); }
    };

    $("#btn-apply").onclick = async () => {
      if (!confirm("Apply the plan now? This modifies files on disk.")) return;
      try {
        const r = await api("/api/plan/apply", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            operation: $("#op").value, mode: $("#orgmode").value,
            output_dir: $("#orgroot").value.trim(),
          }),
        });
        if (r.ok) {
          logLine(`Applied: ${r.executed} moved (txn ${r.transaction_id})`, "ok");
          toast(`Applied ${r.executed} files`, "ok");
        } else {
          logLine(`Apply failed: ${r.error || r.failures} failures`, "err");
          toast("Apply had failures", "err");
        }
        await loadTransactions();
        await refresh();
      } catch (e) { toast(e.message, "err"); }
    };

    $("#btn-undo").onclick = async () => {
      const tid = $("#txnselect").value || null;
      if (!confirm("Undo this transaction? Files will be moved back to their original paths.")) return;
      try {
        const r = await api("/api/undo", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ transaction_id: tid }),
        });
        if (r.ok) {
          logLine(`Undid ${r.reversed} actions (txn ${r.transaction_id})`, "ok");
          toast(`Reversed ${r.reversed} files`, "ok");
        } else {
          logLine(`Undo had ${r.failures} failures`, "err");
          toast("Undo had failures", "err");
        }
        await loadTransactions();
        await refresh();
      } catch (e) { toast(e.message, "err"); }
    };

    await Promise.all([loadStats(), loadImages(), loadTransactions()]);
    fillCatFilter();
    renderCategories();
    renderJunk();
    renderUncertain();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
