/* Virtualized photo grid. Only the rows in (or just outside) the viewport exist in the DOM.
 *
 * The pure functions at the top have no DOM dependency so Node can test them
 * (node tests/virtual_grid.test.mjs). VirtualGrid below wires them to a real page.
 */

export function computeWindow({ scrollTop, gridTop, viewportH, rowH, cols, total, overscanRows = 2 }) {
  if (total === 0 || cols === 0 || rowH === 0) {
    return { startIdx: 0, endIdx: 0, offsetY: 0, totalHeight: 0 };
  }
  const totalRows = Math.ceil(total / cols);
  const totalHeight = totalRows * rowH;
  // Past the end (a filter just shrank the list) show the last screenful instead of nothing.
  const relScroll = Math.min(Math.max(0, scrollTop - gridTop), Math.max(0, totalHeight - viewportH));
  const startRow = Math.max(0, Math.floor(relScroll / rowH) - overscanRows);
  const endRow = Math.min(totalRows, Math.ceil((relScroll + viewportH) / rowH) + overscanRows);
  return {
    startIdx: startRow * cols,
    endIdx: Math.min(total, endRow * cols),
    offsetY: startRow * rowH,
    totalHeight,
  };
}

/** How many columns of at least `minCard` px fit in `width` with `gap` between them. */
export function columnsFor(width, minCard, gap) {
  if (!(width > 0)) return 0;
  return Math.max(1, Math.floor((width + gap) / (minCard + gap)));
}

/** Height of one card (thumbnail at `aspect` = height/width, plus a fixed caption) and of one row. */
export function rowMetrics({ width, cols, gap, rowGap, aspect, captionH }) {
  if (cols <= 0) return { cardW: 0, cardH: 0, rowH: 0 };
  const cardW = (width - gap * (cols - 1)) / cols;
  const cardH = Math.round(cardW * aspect + captionH);
  return { cardW, cardH, rowH: cardH + rowGap };
}

/** Offsets of the pages that cover [startIdx, endIdx) and still have unloaded items. */
export function pagesToLoad({ startIdx, endIdx, pageSize, isLoaded, total }) {
  const out = [];
  if (endIdx <= startIdx) return out;
  const last = total == null ? endIdx : Math.min(endIdx, total);
  for (let page = Math.floor(startIdx / pageSize); page * pageSize < last; page++) {
    const from = page * pageSize;
    const to = Math.min(from + pageSize, total == null ? from + pageSize : total);
    let missing = false;
    for (let i = Math.max(from, startIdx); i < Math.min(to, last); i++) {
      if (!isLoaded(i)) { missing = true; break; }
    }
    if (missing) out.push(from);
  }
  return out;
}

/** Where to scroll so row `row` is fully visible, or null if it already is. */
export function scrollTarget({ row, rowH, gridTop, scrollTop, viewportH, topInset = 0, margin = 12 }) {
  const top = gridTop + row * rowH;
  const bottom = top + rowH;
  if (top - margin < scrollTop + topInset) return Math.max(0, top - topInset - margin);
  if (bottom + margin > scrollTop + viewportH) return Math.max(0, bottom + margin - viewportH);
  return null;
}

/** Move a keyboard cursor by `delta` items, staying inside [0, total). Returns the new index. */
export function moveIndex(current, delta, total) {
  if (total <= 0) return -1;
  const from = current < 0 ? (delta >= 0 ? -1 : total) : current;
  return Math.min(total - 1, Math.max(0, from + delta));
}

export class VirtualGrid {
  /**
   * host        element that becomes the grid
   * fetchPage   async (offset, limit) => { items, total }
   * renderCard  (item) => Element, never touching innerHTML with data from the library
   * renderSkeleton () => Element for a photo that has not loaded yet
   */
  constructor({
    host, fetchPage, renderCard, renderSkeleton, pageSize = 200, minCard = 200, colGap = 16,
    rowGap = 22, aspect = 0.75, captionH = 92, overscanRows = 2, topInset = 84,
    onChange = () => {}, onError = () => {}, win = window,
  }) {
    Object.assign(this, { host, fetchPage, renderCard, renderSkeleton, pageSize, minCard, colGap, rowGap,
      aspect, captionH, overscanRows, topInset, onChange, onError, win });
    this.items = [];
    this.total = null;
    this.cursorId = null;
    this.cards = new Map();      // index -> element currently in the DOM
    this.epoch = 0;              // bumped whenever loaded pages stop being trustworthy
    this.inflight = new Set();
    this.cols = 0;
    this.rowH = 0;
    this.raf = 0;
    host.classList.add("vgrid");
    if (typeof host.setAttribute === "function") {
      host.setAttribute("role", "grid");
      host.setAttribute("aria-label", "Photos");
    }
    host.replaceChildren();
    this.spacer = document.createElement("div");
    this.spacer.className = "vg-spacer";
    if (typeof this.spacer.setAttribute === "function") {
      this.spacer.setAttribute("role", "presentation");
    }
    this.window = document.createElement("div");
    this.window.className = "vg-window";
    if (typeof this.window.setAttribute === "function") {
      this.window.setAttribute("role", "row");
    }
    this.empty = document.createElement("div");
    this.empty.className = "vg-empty";
    this.empty.hidden = true;
    this.spacer.appendChild(this.window);
    host.append(this.spacer, this.empty);
    this._onFocusIn = (e) => {
      const card = e.target && e.target.closest ? e.target.closest(".card") : null;
      if (card && card.dataset && card.dataset.id) {
        const id = Number(card.dataset.id);
        if (id && this.cursorId !== id) {
          this.cursorId = id;
          this.applyCursor();
        }
      }
    };
    if (typeof host.addEventListener === "function") {
      host.addEventListener("focusin", this._onFocusIn);
    }
    this._schedule = () => {
      if (this.raf) return;
      this.raf = win.requestAnimationFrame(() => { this.raf = 0; this.update(); });
    };
    win.addEventListener("scroll", this._schedule, { passive: true });
    win.addEventListener("resize", this._schedule);
    if (typeof ResizeObserver !== "undefined") {
      this.resizer = new ResizeObserver(() => this._schedule());
      this.resizer.observe(host);
    }
  }

  destroy() {
    if (typeof this.host.removeEventListener === "function") {
      this.host.removeEventListener("focusin", this._onFocusIn);
    }
    this.win.removeEventListener("scroll", this._schedule);
    this.win.removeEventListener("resize", this._schedule);
    if (this.resizer) this.resizer.disconnect();
  }

  get renderedCount() { return this.cards.size; }
  get visible() { return this.host.offsetParent !== null; }
  idAt(i) { const it = this.items[i]; return it ? it.id : null; }
  getItem(i) { return this.items[i]; }
  indexOfId(id) {
    for (let i = 0; i < this.items.length; i++) if (this.items[i] && this.items[i].id === id) return i;
    return -1;
  }

  /** Throw away everything and load the first page again (filters changed). */
  async reset() {
    this.epoch++;
    this.items = [];
    this.total = null;
    this.inflight.clear();
    this._clearCards();
    this.empty.hidden = true;
    this.spacer.hidden = false;
    if (this.host.getBoundingClientRect().top < 0) this._scrollToTop();
    this._bootSkeletons();
    await this._load(0, this.pageSize);
    this.update();
  }

  /** A few placeholders so the grid is not blank while the first page loads. */
  _bootSkeletons() {
    this.window.style.transform = "";
    this.window.replaceChildren(...Array.from({ length: 12 }, () => {
      const el = this.renderSkeleton();
      el.dataset.boot = "1";
      return el;
    }));
  }

  _scrollToTop() {
    const top = this.host.getBoundingClientRect().top + this.win.scrollY - this.topInset;
    this.win.scrollTo(0, Math.max(0, top));
  }

  showEmpty(node) {
    this.empty.replaceChildren(node);
    this.empty.hidden = false;
    this.spacer.hidden = true;
  }

  _clearCards() {
    this.cards.forEach((el) => el.remove());
    this.cards.clear();
  }

  /** Card size and caption height come from the stylesheet (--thumb-min, --vg-caption-h). */
  _css(name, fallback) {
    const v = parseFloat(this.win.getComputedStyle(this.host).getPropertyValue(name));
    return v > 0 ? v : fallback;
  }

  _measure() {
    const width = this.host.clientWidth;
    this.cols = columnsFor(width, this._css("--thumb-min", this.minCard), this.colGap);
    const m = rowMetrics({ width, cols: this.cols, gap: this.colGap, rowGap: this.rowGap,
      aspect: this.aspect, captionH: this._css("--vg-caption-h", this.captionH) });
    if (this.rowH !== m.rowH || this._cols !== this.cols) this._clearCards();  // layout changed
    this.rowH = m.rowH;
    this._cols = this.cols;
    this.host.style.setProperty("--vg-cols", String(this.cols));
    this.host.style.setProperty("--vg-card-h", m.cardH + "px");
    this.host.style.setProperty("--vg-row-gap", this.rowGap + "px");
    this.host.style.setProperty("--vg-col-gap", this.colGap + "px");
  }

  /** Recompute which cards should exist, add the missing ones, drop the rest, fetch what is blank. */
  update() {
    if (this.total == null || !this.visible) return;
    this._measure();
    const gridTop = this.host.getBoundingClientRect().top + this.win.scrollY;
    const w = computeWindow({
      scrollTop: this.win.scrollY, gridTop, viewportH: this.win.innerHeight, rowH: this.rowH,
      cols: this.cols, total: this.total,
      overscanRows: this.cols > 10 ? 1 : this.overscanRows,
    });
    this.spacer.style.height = w.totalHeight + "px";
    this.window.style.transform = `translateY(${w.offsetY}px)`;
    this.window.style.setProperty("--vg-cols", String(this.cols));

    this.window.querySelectorAll("[data-boot]").forEach((el) => el.remove());
    for (const [idx, el] of this.cards) {
      if (idx < w.startIdx || idx >= w.endIdx) { el.remove(); this.cards.delete(idx); }
      else if (el.dataset.skeleton && this.items[idx]) { el.remove(); this.cards.delete(idx); }
    }
    // Walk backwards so each new card is inserted before the one that follows it.
    let ref = null;
    for (let idx = w.endIdx - 1; idx >= w.startIdx; idx--) {
      let el = this.cards.get(idx);
      if (!el) {
        const it = this.items[idx];
        el = it ? this.renderCard(it) : this.renderSkeleton();
        if (!it) el.dataset.skeleton = "1";
        el.dataset.idx = String(idx);
        this.window.insertBefore(el, ref);
        this.cards.set(idx, el);
      }
      ref = el;
    }
    this.applyCursor();
    this._ensure(w.startIdx, w.endIdx);
  }

  _ensure(startIdx, endIdx) {
    const offsets = pagesToLoad({ startIdx, endIdx, pageSize: this.pageSize, total: this.total,
      isLoaded: (i) => !!this.items[i] });
    offsets.forEach((o) => this._load(o, this.pageSize).then(() => this._schedule()));
  }

  async ensureLoaded(startIdx, endIdx) {
    const offsets = pagesToLoad({ startIdx, endIdx, pageSize: this.pageSize, total: this.total,
      isLoaded: (i) => !!this.items[i] });
    await Promise.all(offsets.map((o) => this._load(o, this.pageSize)));
  }

  async _load(offset, limit) {
    const key = offset + ":" + this.epoch;
    if (this.inflight.has(key)) return;
    this.inflight.add(key);
    const epoch = this.epoch;
    try {
      const data = await this.fetchPage(offset, limit);
      if (epoch !== this.epoch) return;           // the list changed while this was in flight
      this.total = data.total;
      data.items.forEach((it, k) => { if (!this.items[offset + k]) this.items[offset + k] = it; });
      this.onChange(this);
      if (this.total === 0) { this._clearCards(); this.spacer.style.height = "0px"; }
    } catch (e) {
      if (epoch === this.epoch) this.onError(e);
    } finally {
      this.inflight.delete(key);
    }
  }

  /** Replace the card for an item whose data changed. */
  rerenderItem(item) {
    const idx = this.indexOfId(item.id);
    if (idx < 0) return;
    this.items[idx] = item;
    const old = this.cards.get(idx);
    if (!old) return;
    const el = this.renderCard(item);
    el.dataset.idx = String(idx);
    old.replaceWith(el);
    this.cards.set(idx, el);
    this.applyCursor();
  }

  /** Drop an item (it no longer matches the filter). Later positions shift up, as they do on the server. */
  removeItem(id) {
    const idx = this.indexOfId(id);
    if (idx < 0) return;
    this.epoch++;                 // pages in flight were computed before this removal
    this.inflight.clear();
    this.items.splice(idx, 1);
    this.total = Math.max(0, this.total - 1);
    this._clearCards();
    this.onChange(this);
    this.update();
  }

  setCursor(id) { this.cursorId = id; this.applyCursor(); }

  applyCursor() {
    let rovingCard = null;
    if (this.cards.size > 0) {
      if (this.cursorId != null) {
        for (const [, el] of this.cards) {
          if (Number(el.dataset && el.dataset.id) === this.cursorId) {
            rovingCard = el;
            break;
          }
        }
      }
      if (!rovingCard) {
        let minIdx = Infinity;
        for (const idx of this.cards.keys()) {
          if (idx < minIdx) minIdx = idx;
        }
        if (minIdx !== Infinity) rovingCard = this.cards.get(minIdx);
      }
    }

    for (const [, el] of this.cards) {
      const isSel = this.cursorId != null && Number(el.dataset && el.dataset.id) === this.cursorId;
      if (el.classList && typeof el.classList.toggle === "function") {
        el.classList.toggle("sel", isSel);
      }
      const isRoving = el === rovingCard;
      if (typeof el.setAttribute === "function") {
        el.setAttribute("tabindex", isRoving ? "0" : "-1");
        el.setAttribute("role", "gridcell");
      }
      const btns = typeof el.querySelectorAll === "function" ? el.querySelectorAll("button, [role='button'], a") : [];
      for (const b of btns) {
        if (typeof b.setAttribute === "function") {
          b.setAttribute("tabindex", isRoving ? "0" : "-1");
        }
      }
    }
  }

  /** Scroll until item `idx` is on screen, loading its page first if needed. */
  async scrollToIndex(idx) {
    if (idx < 0 || this.total == null || idx >= this.total) return;
    this._measure();
    const gridTop = this.host.getBoundingClientRect().top + this.win.scrollY;
    const y = scrollTarget({ row: Math.floor(idx / this.cols), rowH: this.rowH, gridTop,
      scrollTop: this.win.scrollY, viewportH: this.win.innerHeight, topInset: this.topInset });
    if (y != null) this.win.scrollTo(0, y);
    await this.ensureLoaded(idx, idx + 1);
    this.update();
  }

  /** Move the cursor; unrendered and even unloaded items are scrolled into view first. */
  async moveCursor(delta) {
    if (!this.total) return null;
    const next = moveIndex(this.indexOfId(this.cursorId), delta, this.total);
    await this.ensureLoaded(next, next + 1);
    const it = this.items[next];
    if (!it) return null;
    this.cursorId = it.id;
    await this.scrollToIndex(next);
    this.applyCursor();
    const card = this.cards.get(next);
    if (card && typeof card.focus === "function") {
      card.focus({ preventScroll: false });
    }
    return it;
  }
}

// app.js is a classic script, so hand it what it needs.
if (typeof window !== "undefined") window.PCUR_GRID = { VirtualGrid, computeWindow };
