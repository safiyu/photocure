// node tests/virtual_grid.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import {
  computeWindow, columnsFor, rowMetrics, pagesToLoad, scrollTarget, moveIndex,
} from "../photocure/ui/static/virtual_grid.mjs";

const base = { gridTop: 100, viewportH: 800, rowH: 250, cols: 5, total: 5000, overscanRows: 2 };

test("initial viewport: starts at row 0 and renders only a screenful plus overscan", () => {
  const w = computeWindow({ ...base, scrollTop: 0 });
  assert.equal(w.startIdx, 0);
  assert.equal(w.offsetY, 0);
  assert.equal(w.totalHeight, 1000 * 250);                 // 5000 / 5 = 1000 rows
  // (800 / 250 = 3.2 -> 4 rows) + 2 overscan = 6 rows = 30 cards
  assert.equal(w.endIdx, 6 * 5);
});

test("scrolled above the grid behaves like the top", () => {
  assert.deepEqual(computeWindow({ ...base, scrollTop: 40 }), computeWindow({ ...base, scrollTop: 0 }));
});

test("middle of the list: window is anchored to whole rows and contains the viewport", () => {
  const scrollTop = 100 + 250 * 400 + 30;                 // inside row 400
  const w = computeWindow({ ...base, scrollTop });
  assert.equal(w.startIdx % base.cols, 0);
  assert.equal(w.offsetY, (400 - 2) * 250);
  assert.ok(w.startIdx <= 400 * 5 && w.endIdx >= 404 * 5);
  assert.ok(w.endIdx - w.startIdx <= 100);                 // the DOM budget
});

test("bottom of the list: never runs past the last item", () => {
  const w = computeWindow({ ...base, scrollTop: 100 + 1000 * 250 });
  assert.equal(w.endIdx, 5000);
  assert.ok(w.startIdx < 5000 && w.startIdx >= 5000 - 5 * 10);
});

test("scrolled far past the end still shows the last rows, never an inverted range", () => {
  const w = computeWindow({ ...base, scrollTop: 1e9 });
  assert.equal(w.endIdx, 5000);
  assert.ok(w.startIdx < w.endIdx && w.startIdx >= 5000 - 5 * 10);
  assert.ok(w.offsetY <= w.totalHeight);
});

test("a total that does not fill the last row still ends exactly at total", () => {
  const w = computeWindow({ ...base, total: 5003, scrollTop: 1e9 });
  assert.equal(w.endIdx, 5003);
  assert.equal(w.totalHeight, Math.ceil(5003 / 5) * 250);
});

test("resize that changes the column count keeps the same photo on screen", () => {
  // Photo 2000 sits in row 400 at 5 columns. At 8 columns it sits in row 250.
  const before = computeWindow({ ...base, scrollTop: 100 + 400 * 250 });
  assert.ok(before.startIdx <= 2000 && 2000 < before.endIdx);
  const after = computeWindow({ ...base, cols: 8, scrollTop: 100 + 250 * 250 });
  assert.ok(after.startIdx <= 2000 && 2000 < after.endIdx);
  assert.equal(after.totalHeight, 625 * 250);              // 5000 / 8 rows
  assert.equal(after.startIdx % 8, 0);
});

test("single column and one-row lists", () => {
  assert.equal(computeWindow({ ...base, cols: 1, total: 3, scrollTop: 0 }).endIdx, 3);
  const one = computeWindow({ ...base, cols: 6, total: 4, scrollTop: 0 });
  assert.deepEqual([one.startIdx, one.endIdx, one.totalHeight], [0, 4, 250]);
});

test("empty dataset and degenerate layouts give an empty window", () => {
  const zero = { startIdx: 0, endIdx: 0, offsetY: 0, totalHeight: 0 };
  assert.deepEqual(computeWindow({ ...base, total: 0, scrollTop: 500 }), zero);
  assert.deepEqual(computeWindow({ ...base, cols: 0, scrollTop: 0 }), zero);
  assert.deepEqual(computeWindow({ ...base, rowH: 0, scrollTop: 0 }), zero);
});

test("rendered count stays within budget at every scroll position", () => {
  let worst = 0;
  for (let y = 0; y <= 100 + 1000 * 250 + 800; y += 37) {
    const w = computeWindow({ ...base, scrollTop: y });
    worst = Math.max(worst, w.endIdx - w.startIdx);
    assert.ok(w.startIdx >= 0 && w.endIdx <= base.total && w.startIdx <= w.endIdx);
  }
  assert.ok(worst <= 100, `rendered ${worst}`);
});

test("columnsFor", () => {
  assert.equal(columnsFor(1416, 200, 16), 6);
  assert.equal(columnsFor(432, 200, 16), 2);
  assert.equal(columnsFor(150, 200, 16), 1);               // narrower than one card: still one column
  assert.equal(columnsFor(0, 200, 16), 0);
});

test("rowMetrics", () => {
  const m = rowMetrics({ width: 1416, cols: 6, gap: 16, rowGap: 22, aspect: 0.75, captionH: 92 });
  assert.equal(m.cardW, (1416 - 80) / 6);
  assert.equal(m.cardH, Math.round(m.cardW * 0.75 + 92));
  assert.equal(m.rowH, m.cardH + 22);
  assert.deepEqual(rowMetrics({ width: 100, cols: 0, gap: 1, rowGap: 1, aspect: 1, captionH: 1 }), { cardW: 0, cardH: 0, rowH: 0 });
});

test("pagesToLoad asks only for pages that have gaps inside the wanted range", () => {
  const loaded = new Set();
  for (let i = 0; i < 200; i++) loaded.add(i);             // page 0 is in
  const isLoaded = (i) => loaded.has(i);
  assert.deepEqual(pagesToLoad({ startIdx: 0, endIdx: 150, pageSize: 200, isLoaded, total: 1000 }), []);
  assert.deepEqual(pagesToLoad({ startIdx: 150, endIdx: 450, pageSize: 200, isLoaded, total: 1000 }), [200, 400]);
  assert.deepEqual(pagesToLoad({ startIdx: 900, endIdx: 1100, pageSize: 200, isLoaded, total: 1000 }), [800]);
  assert.deepEqual(pagesToLoad({ startIdx: 5, endIdx: 5, pageSize: 200, isLoaded, total: 1000 }), []);
  assert.deepEqual(pagesToLoad({ startIdx: 0, endIdx: 10, pageSize: 200, isLoaded: () => false, total: null }), [0]);
});

test("scrollTarget leaves visible rows alone and brings hidden ones into view", () => {
  const v = { rowH: 250, gridTop: 100, viewportH: 800, topInset: 80 };
  assert.equal(scrollTarget({ ...v, row: 1, scrollTop: 0 }), null);          // fully inside the viewport
  const down = scrollTarget({ ...v, row: 10, scrollTop: 0 });
  assert.ok(down > 0 && down + 800 >= 100 + 11 * 250);     // bottom edge now visible
  const up = scrollTarget({ ...v, row: 1, scrollTop: 5000 });
  assert.ok(up < 5000 && up <= 100 + 250 - 80);            // top edge clears the sticky bar
});

test("moveIndex clamps at both ends and starts sensibly with no cursor", () => {
  assert.equal(moveIndex(10, 1, 100), 11);
  assert.equal(moveIndex(0, -1, 100), 0);
  assert.equal(moveIndex(99, 5, 100), 99);
  assert.equal(moveIndex(-1, 1, 100), 0);                  // first press selects the first photo
  assert.equal(moveIndex(-1, -1, 100), 99);
  assert.equal(moveIndex(3, 1, 0), -1);
});
