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

  const state = { images: [], byId: {}, tab: "library" };

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
    const rawCat = it.manual_bucket || it.category;
    const isUnrouted = !rawCat;
    const catDisplay = it.manual_bucket
      ? `✏️ ${it.manual_bucket}`
      : (it.category || "Unrouted (Pending AI)");
    const meta = [];
    if (it.width && it.height) meta.push(it.width + "×" + it.height);
    if (it.taken_at) meta.push(it.taken_at.slice(0, 10));
    if (it.top1) meta.push(it.top1 + " " + Math.round((it.top1_score || 0) * 100) + "%");
    if (it.quality_score != null) meta.push("Q:" + Number(it.quality_score).toFixed(2));

    return `<div class="card" data-id="${it.id}">
      <div class="thumb">
        <img loading="lazy" src="${thumbUrl(it.id)}" alt="" onerror="this.style.opacity=.15">
        <div class="flags">${flagBadges(it)}</div>
      </div>
      <div class="cardbody">
        <div class="fn" title="${esc(it.relative_path || it.filename)}">${esc(it.filename)}</div>
        <div class="meta">${esc(meta.join(" · "))}</div>
        <div class="catchip-row">
          <span class="catchip ${isUnrouted ? 'unrouted' : (it.manual_bucket ? 'manual' : 'active')}">${esc(catDisplay)}</span>
          ${it.manual_bucket ? `<button class="mini danger clear-btn" data-clear="${it.id}" title="Reset manual override">✕</button>` : ""}
        </div>
        <div class="card-reassign">
          <select class="cat-select" data-id="${it.id}" title="Manually reassign category">
            <option value="" disabled selected>Move to category…</option>
            ${CATS.map((c) =>
              `<option value="${c}" ${c === rawCat ? "disabled" : ""}>${c}</option>`
            ).join("")}
          </select>
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
    $$(".cat-select", root).forEach((sel) => {
      sel.onchange = async () => {
        const id = sel.dataset.id;
        const bucket = sel.value;
        if (!bucket) return;
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

  function logLine(msg, cls) {
    const el = document.createElement("div");
    el.className = "logline " + (cls || "");
    el.textContent = new Date().toLocaleTimeString() + "  " + msg;
    const box = $("#action-log");
    box.prepend(el);
    if (box.children.length > 60) box.lastChild.remove();
  }

  function libLog(msg, cls) {
    const el = document.createElement("div");
    el.className = "logline " + (cls || "");
    el.textContent = new Date().toLocaleTimeString() + "  " + msg;
    const box = $("#library-log");
    if (box) {
      box.prepend(el);
      if (box.children.length > 60) box.lastChild.remove();
    }
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

  let progressTimer = null;
  let isPollingInFlight = false;
  let isTaskRunning = false;
  let lastWasActive = false;

  async function pollProgress() {
    if (isPollingInFlight) return;
    isPollingInFlight = true;
    try {
      const p = await api("/api/progress");
      const card = $("#progress-card");
      const fill = $("#progress-fill");
      const title = $("#progress-title");
      const msg = $("#progress-msg");
      const pct = $("#progress-pct");
      const counts = $("#progress-counts");
      const spinner = $("#progress-spinner");
      const statusIcon = $("#progress-status-icon");
      const btnDismiss = $("#btn-dismiss-progress");
      const topbar = $("#topbar-progress");
      const topbarSpinner = $("#topbar-spinner");
      const topbarIcon = $("#topbar-status-icon");
      const topbarText = $("#topbar-progress-text");
      const topbarFill = $("#topbar-fill");

      if (p.active) {
        lastWasActive = true;
        // Automatically start the timer if not already running (e.g. after page reload/refresh)
        if (!progressTimer) {
          progressTimer = setInterval(pollProgress, 500);
        }

        if (card) card.hidden = false;
        if (btnDismiss) btnDismiss.style.display = "none";
        if (spinner) {
          spinner.hidden = false;
          spinner.style.display = "inline-block";
        }
        if (statusIcon) statusIcon.hidden = true;
        if (title) title.textContent = `${p.task || "Processing"} — ${p.step || ""}`;
        if (pct) pct.textContent = `${p.percent}%`;
        if (fill) {
          fill.className = "progress-fill active";
          fill.style.width = `${Math.max(p.percent, 4)}%`;
        }
        if (msg) msg.textContent = p.message || "Working…";
        if (counts) {
          counts.textContent = p.detail || (p.total > 0 ? `${p.done} / ${p.total}` : "");
        }
        if (topbar) {
          topbar.hidden = false;
          if (topbarSpinner) {
            topbarSpinner.hidden = false;
            topbarSpinner.style.display = "inline-block";
          }
          if (topbarIcon) topbarIcon.hidden = true;
          if (topbarText) topbarText.textContent = `${p.task || "Working"} ${p.percent}%`;
          if (topbarFill) {
            topbarFill.className = "topbar-fill";
            topbarFill.style.width = `${Math.max(p.percent, 4)}%`;
          }
        }
      } else {
        // p.active is false
        if (isTaskRunning) {
          // A task action is currently in-flight on the server.
          // Keep polling until the action completes!
          return p;
        }

        // Action finished:
        if (spinner) {
          spinner.hidden = true;
          spinner.style.display = "none";
        }
        if (btnDismiss) btnDismiss.style.display = "inline-block";

        if (p.error) {
          if (card) card.hidden = false;
          if (statusIcon) {
            statusIcon.hidden = false;
            statusIcon.textContent = "✕";
            statusIcon.style.color = "var(--err)";
            statusIcon.style.fontWeight = "bold";
          }
          if (title) title.textContent = `${p.task || "Task"} Failed`;
          if (fill) {
            fill.className = "progress-fill error";
            fill.style.width = "100%";
          }
          if (msg) msg.textContent = p.error;
          if (pct) pct.textContent = "Error";
          if (topbar) {
            topbar.hidden = false;
            if (topbarSpinner) {
              topbarSpinner.hidden = true;
              topbarSpinner.style.display = "none";
            }
            if (topbarIcon) {
              topbarIcon.hidden = false;
              topbarIcon.textContent = "✕";
              topbarIcon.style.color = "var(--err)";
            }
            if (topbarText) topbarText.textContent = `${p.task || "Task"} Error`;
            if (topbarFill) {
              topbarFill.className = "topbar-fill error";
              topbarFill.style.width = "100%";
            }
          }
          stopProgressPolling();
        } else if (p.percent === 100) {
          if (card) card.hidden = false;
          if (statusIcon) {
            statusIcon.hidden = false;
            statusIcon.textContent = "✓";
            statusIcon.style.color = "var(--ok)";
            statusIcon.style.fontWeight = "bold";
          }
          if (title) title.textContent = `${p.task || "Task"} Finished`;
          if (fill) {
            fill.className = "progress-fill done";
            fill.style.width = "100%";
          }
          if (msg) msg.textContent = p.message || "Completed successfully.";
          if (pct) pct.textContent = "100%";
          if (counts) counts.textContent = p.detail || (p.total > 0 ? `${p.done} / ${p.total}` : "");
          if (topbar) {
            topbar.hidden = false;
            if (topbarSpinner) {
              topbarSpinner.hidden = true;
              topbarSpinner.style.display = "none";
            }
            if (topbarIcon) {
              topbarIcon.hidden = false;
              topbarIcon.textContent = "✓";
              topbarIcon.style.color = "var(--ok)";
            }
            if (topbarText) topbarText.textContent = `${p.task || "Task"} 100%`;
            if (topbarFill) {
              topbarFill.className = "topbar-fill done";
              topbarFill.style.width = "100%";
            }
          }

          // Automatically reload stats and grid if we just finished an active task
          if (lastWasActive) {
            lastWasActive = false;
            loadStats();
            loadImages().then(() => {
              fillCatFilter();
              if (state.tab === "categories") renderCategories();
              if (state.tab === "junk") renderJunk();
              if (state.tab === "uncertain") renderUncertain();
            });
          }

          stopProgressPolling();
        } else {
          if (card) card.hidden = true;
          if (topbar) topbar.hidden = true;
          stopProgressPolling();
        }
      }
      return p;
    } catch (e) {
      if (!isTaskRunning && !lastWasActive) {
        stopProgressPolling();
      }
    } finally {
      isPollingInFlight = false;
    }
  }

  function startProgressPolling() {
    isTaskRunning = true;
    lastWasActive = true;
    stopProgressPolling();
    pollProgress();
    progressTimer = setInterval(pollProgress, 500);
  }

  function finishProgressPolling() {
    isTaskRunning = false;
    pollProgress();
  }

  function stopProgressPolling() {
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
  }

  async function init() {
    $$(".tab").forEach((t) => (t.onclick = () => showTab(t.dataset.tab)));
    ["q", "catfilter", "manualonly"].forEach((id) => {
      const el = $("#" + id);
      if (el) el.addEventListener("input", renderCategories);
    });
    const junkF = $("#junkfilter");
    if (junkF) junkF.addEventListener("change", renderJunk);

    const btnDismiss = $("#btn-dismiss-progress");
    if (btnDismiss) {
      btnDismiss.onclick = () => {
        const card = $("#progress-card");
        if (card) card.hidden = true;
      };
    }

    const scanInput = $("#scan-path");
    if (scanInput) {
      scanInput.addEventListener("input", () => scanInput.classList.remove("input-error"));
      scanInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          const btn = $("#btn-scan");
          if (btn) btn.click();
        }
      });
    }

    $$(".quick-btn").forEach((btn) => {
      btn.onclick = () => {
        if (scanInput) {
          scanInput.value = btn.dataset.path || "";
          scanInput.classList.remove("input-error");
          scanInput.focus();
        }
      };
    });

    function setBusy(busy) {
      ["btn-scan", "btn-infer", "btn-cluster", "btn-ingest-all"].forEach((id) => {
        const btn = $("#" + id);
        if (btn) btn.disabled = busy;
      });
    }

    const btnScan = $("#btn-scan");
    if (btnScan) {
      btnScan.onclick = async () => {
        const path = scanInput ? scanInput.value.trim() : "";
        if (!path) {
          if (scanInput) {
            scanInput.classList.add("input-error");
            scanInput.focus();
          }
          libLog("⚠️ Please enter a folder path to scan (or click one of the quick fill buttons)", "err");
          toast("Please enter a folder path to scan", "err");
          return;
        }
        setBusy(true);
        startProgressPolling();
        libLog(`Scanning folder: ${path}…`);
        try {
          const r = await api("/api/scan", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path }),
          });
          libLog(`Scan complete: ${r.scanned} files seen, ${r.new} new indexed, ${r.duplicates} duplicates in ${r.duration_s}s`, "ok");
          toast(`Indexed ${r.new} new photos`, "ok");
          await Promise.all([loadStats(), loadImages()]);
          if (state.tab === "categories") renderCategories();
        } catch (e) {
          libLog(`Scan error: ${e.message}`, "err");
          toast(e.message, "err");
        } finally {
          finishProgressPolling();
          setBusy(false);
        }
      };
    }

    const btnInfer = $("#btn-infer");
    if (btnInfer) {
      btnInfer.onclick = async () => {
        const device = $("#infer-device") ? $("#infer-device").value : "auto";
        setBusy(true);
        startProgressPolling();
        libLog(`Running AI inference on pending images (device=${device})…`);
        try {
          const r = await api("/api/infer", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device }),
          });
          libLog(`Inference complete: ${r.processed} processed, ${r.faces_total} faces, ${r.semantic_computed} semantics in ${r.duration_s}s`, "ok");
          toast(`Inferred ${r.processed} photos`, "ok");
          await Promise.all([loadStats(), loadImages()]);
          if (state.tab === "categories") renderCategories();
        } catch (e) {
          libLog(`Inference error: ${e.message}`, "err");
          toast(e.message, "err");
        } finally {
          finishProgressPolling();
          setBusy(false);
        }
      };
    }

    const btnCluster = $("#btn-cluster");
    if (btnCluster) {
      btnCluster.onclick = async () => {
        setBusy(true);
        startProgressPolling();
        libLog("Clustering burst shots…");
        try {
          const r = await api("/api/cluster", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          });
          libLog(`Burst clustering complete: ${r.burst_groups} groups found`, "ok");
          toast(`Clustered ${r.burst_groups} bursts`, "ok");
          await Promise.all([loadStats(), loadImages()]);
          if (state.tab === "categories") renderCategories();
        } catch (e) {
          libLog(`Clustering error: ${e.message}`, "err");
          toast(e.message, "err");
        } finally {
          finishProgressPolling();
          setBusy(false);
        }
      };
    }

    const btnIngest = $("#btn-ingest-all");
    if (btnIngest) {
      btnIngest.onclick = async () => {
        const path = scanInput ? scanInput.value.trim() : "";
        if (!path) {
          if (scanInput) {
            scanInput.classList.add("input-error");
            scanInput.focus();
          }
          libLog("⚠️ Please enter a folder path to scan (or click one of the quick fill buttons)", "err");
          toast("Please enter a folder path to scan", "err");
          return;
        }
        const device = $("#infer-device") ? $("#infer-device").value : "auto";
        setBusy(true);
        startProgressPolling();
        libLog(`Starting full ingest for ${path}…`);
        try {
          libLog(`[1/3] Scanning folder…`);
          const s = await api("/api/scan", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path }),
          });
          libLog(`[1/3] Scanned: ${s.new} new indexed (${s.scanned} files seen)`, "ok");

          libLog(`[2/3] Running AI inference (device=${device})…`);
          const i = await api("/api/infer", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device }),
          });
          libLog(`[2/3] Inference: ${i.processed} processed, ${i.faces_total} faces found`, "ok");

          libLog(`[3/3] Clustering bursts…`);
          const c = await api("/api/cluster", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          });
          libLog(`[3/3] Clustered: ${c.burst_groups} burst groups detected`, "ok");

          toast("Full ingest pipeline complete!", "ok");
          await Promise.all([loadStats(), loadImages()]);
          if (state.tab === "categories") renderCategories();
        } catch (e) {
          libLog(`Ingest pipeline failed: ${e.message}`, "err");
          toast(e.message, "err");
        } finally {
          finishProgressPolling();
          setBusy(false);
        }
      };
    }

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
          logLine(`Applied: ${r.executed} moved, ${r.failures} failed (txn ${r.transaction_id})`, "err");
          toast(`Applied ${r.executed} files (${r.failures} failed)`, "err");
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

    await Promise.all([loadStats(), loadImages(), loadTransactions(), pollProgress()]);
    fillCatFilter();
    renderCategories();
    renderJunk();
    renderUncertain();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
